"""
BAIT模型训练脚本
"""

import os
import time
import argparse
import json
import pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from bait_model import BAIT, BAITLoss
from data_generation import create_dataloaders
from metrics import TrackingMetrics
from config import load_config

# 🔥 导入带交叉场景的数据生成器
try:
    from data_generation_with_crossing import create_dataloaders_with_crossing
    CROSSING_AVAILABLE = True
except ImportError:
    CROSSING_AVAILABLE = False
    print("⚠️ Warning: data_generation_with_crossing not found, using standard data generation")


def parse_args():
    """解析命令行参数（仅保留必要参数）"""
    parser = argparse.ArgumentParser(description='Train BAIT model')
    
    # 配置文件路径
    parser.add_argument('--config', type=str, default='config_example.json',
                        help='Path to config file')
    
    # 运行时参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # 交叉数据参数
    parser.add_argument('--use-crossing', action='store_true',
                        help='Use crossing trajectory scenarios in training data')
    parser.add_argument('--crossing-prob', type=float, default=0.5,
                        help='Probability of generating crossing scenarios (0.0-1.0)')
    
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_one_step(model, batch, criterion, optimizer, device, grad_clip=1.0):
    """训练一步"""
    model.train()
    
    # 将数据移到设备
    past_states = batch['past_states'].to(device)
    current_measurements = batch['current_measurements'].to(device)
    gt_associations = batch['gt_associations'].to(device)
    gt_states = batch['gt_states'].to(device)
    num_past_targets = batch['num_past_targets'].to(device)
    num_current_measurements = batch['num_current_measurements'].squeeze(-1).to(device)
    num_current_targets = batch['num_current_targets'].squeeze(-1).to(device)
    
    # 前向传播
    match_prob_matrix, filtered_states, existence_probs = model(
        past_states, current_measurements, num_past_targets, num_current_measurements
    )
    
    # 计算损失
    total_loss, loss_dict = criterion(
        match_prob_matrix, filtered_states,
        gt_associations, gt_states,
        num_current_measurements, num_current_targets
    )
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    
    # 梯度裁剪
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    optimizer.step()
    
    return loss_dict


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    
    total_loss = 0.0
    total_association_loss = 0.0
    total_filtering_loss = 0.0
    total_association_acc = 0.0
    total_pos_error_m = 0.0
    num_batches = 0
    
    metrics = TrackingMetrics(c=1.0, p=1)
    
    with torch.no_grad():
        for batch in val_loader:
            # 将数据移到设备
            past_states = batch['past_states'].to(device)
            current_measurements = batch['current_measurements'].to(device)
            gt_associations = batch['gt_associations'].to(device)
            gt_states = batch['gt_states'].to(device)
            num_past_targets = batch['num_past_targets'].to(device)
            num_current_measurements = batch['num_current_measurements'].squeeze(-1).to(device)
            num_current_targets = batch['num_current_targets'].squeeze(-1).to(device)
            
            # 前向传播
            match_prob_matrix, filtered_states, existence_probs = model(
                past_states, current_measurements, num_past_targets, num_current_measurements
            )
            
            # 计算损失
            _, loss_dict = criterion(
                match_prob_matrix, filtered_states,
                gt_associations, gt_states,
                num_current_measurements, num_current_targets
            )
            
            total_loss += loss_dict['total_loss']
            total_association_loss += loss_dict['association_loss']
            total_filtering_loss += loss_dict['filtering_loss']
            total_association_acc += loss_dict['association_acc']
            total_pos_error_m += loss_dict['pos_error_m']
            num_batches += 1
            
            # 计算OSPA指标
            batch_size = filtered_states.size(0)
            for i in range(batch_size):
                num_targets = num_current_targets[i].item()
                # 提取有效的预测和真实状态
                pred_states = filtered_states[i, :num_targets].cpu().numpy()
                true_states = gt_states[i, :num_targets].cpu().numpy()
                
                if len(pred_states) > 0 and len(true_states) > 0:
                    metrics.update_ospa(pred_states, true_states)
    
    n = num_batches if num_batches > 0 else 1
    avg_loss = total_loss / n
    avg_association_loss = total_association_loss / n
    avg_filtering_loss = total_filtering_loss / n
    avg_association_acc = total_association_acc / n
    avg_pos_error_m = total_pos_error_m / n

    ospa_stats = metrics.get_ospa_stats()
    
    return {
        'loss': avg_loss,
        'association_loss': avg_association_loss,
        'filtering_loss': avg_filtering_loss,
        'association_acc': avg_association_acc,
        'pos_error_m': avg_pos_error_m,
        'ospa_mean': ospa_stats['mean'],
        'ospa_loc_mean': ospa_stats['loc_mean'],
        'ospa_card_mean': ospa_stats['card_mean']
    }


def main():
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    print(f"\n使用配置文件: {args.config}")
    print(f"配置内容:\n{json.dumps(config.to_dict(), indent=2, ensure_ascii=False)}\n")
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 从config获取参数
    save_dir = config.get('logging', 'save_dir', default='checkpoints')
    log_dir = config.get('logging', 'log_dir', default='logs')
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 保存配置
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_save_path = os.path.join(save_dir, f'config_{timestamp}.json')
    config.save_config(config_save_path)
    
    # 创建数据加载器
    print("创建数据加载器...")
    # 🔥 根据参数选择数据生成器
    if args.use_crossing and CROSSING_AVAILABLE:
        crossing_prob = config.get('data', 'crossing_probability', default=args.crossing_prob)
        print(f"\n{'='*60}")
        print(f"使用包含两两交叉场景的训练数据（3D雷达）")
        print(f"  交叉场景概率: {crossing_prob * 100:.0f}%")
        print(f"{'='*60}\n")

        train_loader, val_loader, test_loader = create_dataloaders_with_crossing(
            num_train_scenarios=config.get('data', 'num_train_scenarios', default=800),
            num_val_scenarios=config.get('data', 'num_val_scenarios', default=100),
            num_test_scenarios=config.get('data', 'num_test_scenarios', default=100),
            batch_size=config.get('data', 'batch_size', default=16),
            tau=config.get('data', 'tau', default=4),
            max_targets=config.get('model', 'max_targets', default=20),
            max_measurements=config.get('data', 'max_measurements', default=30),
            task_type=config.get('data', 'task_type', default=1),
            num_workers=config.get('data', 'num_workers', default=0),
            crossing_probability=crossing_prob,
        )
    else:
        if args.use_crossing and not CROSSING_AVAILABLE:
            print("⚠️ Warning: --use-crossing specified but crossing module not available, using standard data")
        train_loader, val_loader, test_loader = create_dataloaders(
            num_train_scenarios=config.get('data', 'num_train_scenarios', default=800),
            num_val_scenarios=config.get('data', 'num_val_scenarios', default=100),
            num_test_scenarios=config.get('data', 'num_test_scenarios', default=100),
            batch_size=config.get('data', 'batch_size', default=16),
            tau=config.get('data', 'tau', default=4),
            max_targets=config.get('model', 'max_targets', default=20),
            max_measurements=config.get('data', 'max_measurements', default=30),
            task_type=config.get('data', 'task_type', default=1),
            num_workers=config.get('data', 'num_workers', default=0)
        )
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print(f"测试批次数: {len(test_loader)}")

    # 将测试集原始场景保存到磁盘，供 evaluate_3d.py 直接加载（保证完全一致的场景）
    if hasattr(test_loader.dataset, 'scenarios'):
        test_scenarios_path = os.path.join(save_dir, 'test_scenarios.pkl')
        with open(test_scenarios_path, 'wb') as _f:
            pickle.dump(test_loader.dataset.scenarios, _f)
        print(f"测试集场景已保存 → {test_scenarios_path}（{len(test_loader.dataset.scenarios)} 条）")
    
    # 创建模型
    print("创建模型...")
    model = BAIT(
        d_model=config.get('model', 'd_model', default=256),
        nhead=config.get('model', 'nhead', default=8),
        num_encoder_layers=config.get('model', 'num_encoder_layers', default=6),
        num_associate_decoder_layers=config.get('model', 'num_associate_decoder_layers', default=3),
        num_filtering_decoder_layers=config.get('model', 'num_filtering_decoder_layers', default=6),
        dim_feedforward_encoder=config.get('model', 'dim_feedforward_encoder', default=2048),
        dim_feedforward_associate=config.get('model', 'dim_feedforward_associate', default=1024),
        dim_feedforward_filtering=config.get('model', 'dim_feedforward_filtering', default=2048),
        dropout=config.get('model', 'dropout', default=0.1),
        max_targets=config.get('model', 'max_targets', default=20)
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数数量: {num_params:,}")
    
    # 创建损失函数
    criterion = BAITLoss(
        gamma=config.get('training', 'gamma', default=1.0),
        association_weight=config.get('training', 'association_weight', default=1.0),
        filtering_weight=config.get('training', 'filtering_weight', default=1.0)
    )
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('training', 'lr', default=1e-4),
        weight_decay=config.get('training', 'weight_decay', default=1e-5)
    )
    
    # 学习率调度器
    lr_scheduler_config = config.get('training', 'lr_scheduler', default={'type': 'step', 'step_size': 200000, 'gamma': 0.5})
    if lr_scheduler_config.get('type') == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_scheduler_config.get('step_size', 200000),
            gamma=lr_scheduler_config.get('gamma', 0.5)
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200000, gamma=0.5)
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(log_dir, f'run_{timestamp}'))
    
    # 恢复训练
    start_step = 0
    if args.resume is not None and os.path.exists(args.resume):
        print(f"从检查点恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = checkpoint['step'] + 1
        print(f"从步骤 {start_step} 恢复")
    
    # 获取训练参数
    num_steps = config.get('training', 'num_steps', default=800000)
    grad_clip = config.get('training', 'grad_clip', default=1.0)
    log_interval = config.get('logging', 'log_interval', default=100)
    val_interval = config.get('logging', 'val_interval', default=5000)
    save_interval = config.get('logging', 'save_interval', default=10000)
    
    # 训练循环
    print("\n开始训练...")
    step = start_step
    epoch = 0
    best_val_ospa = float('inf')
    
    train_iter = iter(train_loader)
    
    while step < num_steps:
        epoch_start_time = time.time()
        
        # 获取批次
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # 训练一步
        loss_dict = train_one_step(model, batch, criterion, optimizer, device, grad_clip)
        
        # 更新学习率
        scheduler.step()
        
        # 记录日志
        if step % log_interval == 0:
            elapsed = time.time() - epoch_start_time
            print(f"步骤 {step}/{num_steps} | "
                  f"总损失: {loss_dict['total_loss']:.4f} | "
                  f"关联损失: {loss_dict['association_loss']:.4f} | "
                  f"关联准确率: {loss_dict['association_acc']*100:.1f}% | "
                  f"滤波损失: {loss_dict['filtering_loss']:.4f} | "
                  f"位置误差: {loss_dict['pos_error_m']:.1f}m | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            writer.add_scalar('train/total_loss', loss_dict['total_loss'], step)
            writer.add_scalar('train/association_loss', loss_dict['association_loss'], step)
            writer.add_scalar('train/association_acc', loss_dict['association_acc'], step)
            writer.add_scalar('train/filtering_loss', loss_dict['filtering_loss'], step)
            writer.add_scalar('train/pos_error_m', loss_dict['pos_error_m'], step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)
        
        # 验证
        if step % val_interval == 0:
            print("\n验证中...")
            val_stats = validate(model, val_loader, criterion, device)
            print(f"验证 - 总损失: {val_stats['loss']:.4f} | "
                  f"关联准确率: {val_stats['association_acc']*100:.1f}% | "
                  f"位置误差: {val_stats['pos_error_m']:.1f}m | "
                  f"OSPA: {val_stats['ospa_mean']:.4f} | "
                  f"位置OSPA: {val_stats['ospa_loc_mean']:.4f} | "
                  f"数量OSPA: {val_stats['ospa_card_mean']:.4f}")
            
            writer.add_scalar('val/total_loss', val_stats['loss'], step)
            writer.add_scalar('val/association_loss', val_stats['association_loss'], step)
            writer.add_scalar('val/association_acc', val_stats['association_acc'], step)
            writer.add_scalar('val/filtering_loss', val_stats['filtering_loss'], step)
            writer.add_scalar('val/pos_error_m', val_stats['pos_error_m'], step)
            writer.add_scalar('val/ospa_mean', val_stats['ospa_mean'], step)
            writer.add_scalar('val/ospa_loc_mean', val_stats['ospa_loc_mean'], step)
            writer.add_scalar('val/ospa_card_mean', val_stats['ospa_card_mean'], step)
            
            # 保存最佳模型
            if val_stats['ospa_mean'] < best_val_ospa:
                best_val_ospa = val_stats['ospa_mean']
                save_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_ospa': best_val_ospa,
                }, save_path)
                print(f"保存最佳模型，OSPA: {best_val_ospa:.4f}")
        
        # 定期保存
        if step % save_interval == 0 and step > 0:
            save_path = os.path.join(save_dir, f'checkpoint_step_{step}.pth')
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)
            print(f"保存检查点，步骤 {step}")
        
        step += 1
    
    # 最终保存
    save_path = os.path.join(save_dir, 'final_model.pth')
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, save_path)
    print(f"\n训练完成！最终模型已保存到 {save_path}")
    
    # 在测试集上评估
    print("\n在测试集上评估...")
    test_stats = validate(model, test_loader, criterion, device)
    print(f"测试 - 总损失: {test_stats['loss']:.4f} | "
          f"关联准确率: {test_stats['association_acc']*100:.1f}% | "
          f"位置误差: {test_stats['pos_error_m']:.1f}m | "
          f"OSPA: {test_stats['ospa_mean']:.4f} | "
          f"位置OSPA: {test_stats['ospa_loc_mean']:.4f} | "
          f"数量OSPA: {test_stats['ospa_card_mean']:.4f}")
    
    writer.close()


if __name__ == "__main__":
    main()
