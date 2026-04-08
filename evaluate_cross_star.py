"""
BAIT模型星形交叉场景评估脚本
在星形交叉测试集上评估模型性能，生成详细的指标和可视化
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch

from bait_model import BAIT
from data_generation_with_crossing import MTTDataGeneratorWithCrossing
from metrics import TrackingMetrics

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def generate_star_crossing_scenario_training_type(num_trajectories=6, num_frames=20, dt=0.1, 
                                                  P_d=1.0, lambda_c=8, seed=42):
    """
    生成训练类型的星形交叉轨迹测试场景（使用训练数据生成器，保证可复现）
    
    场景设计：
    - 多条轨迹（4-10条）从中心点辐射出去
    - 所有轨迹在场景中心区域交叉
    - 形成星形或放射状模式
    - 使用与训练相同的生成逻辑
    
    Args:
        num_trajectories: 星形轨迹数量（4-10条）
        num_frames: 帧数（默认20，与训练相同）
        dt: 时间间隔
        P_d: 检测概率
        lambda_c: 杂波泊松参数
        seed: 随机种子（用于可复现性）
    
    Returns:
        trajectories: 轨迹列表
        measurements: 测量值列表
        gt_associations: 真实关联列表
    """
    # 使用星形交叉生成器（与训练相同的配置）
    generator = MTTDataGeneratorWithCrossing(
        crossing_probability=1.0,  # 100%交叉场景
        star_crossing_probability=1.0,  # 100%星形交叉
        num_star_trajectories=(num_trajectories, num_trajectories),  # 固定数量
        task_type=1,
        seed=seed
    )
    
    # 生成场景
    trajectories, measurements, associations = generator.generate_single_scenario()
    
    # 截断到指定帧数
    if len(measurements) > num_frames:
        measurements = measurements[:num_frames]
        associations = associations[:num_frames]
        for traj in trajectories:
            if traj['death_frame'] >= num_frames:
                traj['death_frame'] = num_frames - 1
            if len(traj['states']) > num_frames:
                traj['states'] = traj['states'][:num_frames]
    
    return trajectories, measurements, associations


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate BAIT model with star crossing trajectories')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num-trajectories', type=int, default=6,
                        help='Number of trajectories in star crossing (4-10)')
    parser.add_argument('--scenario-type', type=str, default='training',
                        choices=['training'],
                        help='Scenario type: training (uses training data generator for reproducibility)')
    parser.add_argument('--num-frames', type=int, default=20,
                        help='Number of frames (default 20, same as training)')
    parser.add_argument('--tau', type=int, default=4,
                        help='Number of past frames')
    parser.add_argument('--max-targets', type=int, default=20,
                        help='Maximum number of targets')
    parser.add_argument('--max-measurements', type=int, default=30,
                        help='Maximum number of measurements')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output-dir', type=str, default='evaluation_results_star',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for scenario generation')
    
    return parser.parse_args()


def evaluate_star_scenario(model, num_trajectories, tau, max_targets, max_measurements, 
                           device, output_dir, num_frames=20, scenario_type='training', seed=42):
    """
    评估星形交叉轨迹场景并生成详细的可视化和指标
    
    Args:
        scenario_type: 'training' - 使用训练数据生成器（可复现）
        num_frames: 帧数（默认20，与训练相同）
        seed: 随机种子
    
    Returns:
        scenario_results: dict，包含该场景的所有评估结果
    """
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"评估星形交叉轨迹场景（{num_trajectories}条轨迹）")
    print(f"{'='*60}")
    
    # 生成星形交叉轨迹场景（使用训练类型场景）
    if scenario_type == 'training':
        trajectories, measurements, gt_associations = generate_star_crossing_scenario_training_type(
            num_trajectories=num_trajectories,
            num_frames=num_frames,  # 使用与训练相同的帧数
            dt=0.1,
            P_d=1.0,        # 100%检测率（与训练相同）
            lambda_c=8,     # 平均8个杂波（与训练相同）
            seed=seed       # 使用传入的随机种子
        )
        scenario_description = f"训练类型场景: 星形交叉（{num_trajectories}条轨迹，seed={seed}）"
    
    num_frames = len(measurements)
    print(f"场景类型: {scenario_description}")
    
    # 🔧 归一化坐标（与data_generation.py中的MTTDataset保持一致）
    COORD_SCALE = 50.0  # 场景范围 [-50, 50]m
    for traj in trajectories:
        traj['states'][:, :2] = traj['states'][:, :2] / COORD_SCALE  # 归一化 x, y
    for frame_meas in measurements:
        frame_meas[:] = frame_meas / COORD_SCALE  # 归一化测量
    
    print(f"目标数量: {len(trajectories)}")
    print(f"帧数: {num_frames}")
    
    # 打印轨迹信息
    print(f"\n轨迹配置:")
    for i, traj in enumerate(trajectories[:3]):  # 显示前3条
        start_pos = traj['states'][0, :2] * COORD_SCALE
        end_pos = traj['states'][-1, :2] * COORD_SCALE
        distance = np.linalg.norm(end_pos - start_pos)
        print(f"  轨迹{traj['label']}: ({start_pos[0]:.1f},{start_pos[1]:.1f}) → " +
              f"({end_pos[0]:.1f},{end_pos[1]:.1f}) [移动~{distance:.1f}m]")
    if len(trajectories) > 3:
        print(f"  ... 还有 {len(trajectories)-3} 条轨迹")
    
    # 🔧 初始化跟踪状态 - 使用测量值而非真实状态
    tracked_states = {}  # label -> states history
    tracked_measurements = {}  # label -> matched measurements history
    
    for traj in trajectories:
        label = traj['label']
        tracked_states[label] = []
        tracked_measurements[label] = []
        
        # 初始化前tau帧 - 使用对应的真实测量值
        for t in range(tau):
            if traj['birth_frame'] <= t <= traj['death_frame']:
                # 找到该目标在该帧的测量值
                frame_assoc = gt_associations[t]
                frame_meas = measurements[t]
                
                # 找到对应label的测量
                matched_meas = None
                for i, assoc_label in enumerate(frame_assoc):
                    if assoc_label == label and i < len(frame_meas):
                        matched_meas = frame_meas[i]
                        break
                
                if matched_meas is not None:
                    tracked_states[label].append(matched_meas)
                    tracked_measurements[label].append(matched_meas)
                else:
                    # 如果没有检测到，使用真实状态
                    state = traj['states'][t, :2]
                    tracked_states[label].append(state)
                    tracked_measurements[label].append(None)
            else:
                # 对于不活跃的目标，填充零向量（保持索引对齐）
                tracked_states[label].append(np.zeros(2))
                tracked_measurements[label].append(None)
    
    # 记录所有帧的结果
    all_pred_states = []
    all_true_states = []
    all_matched_measurements = []
    all_association_accuracy = []
    all_num_tracked = []  # 记录每帧跟踪的目标数
    
    with torch.no_grad():
        # 从第tau帧开始跟踪
        for frame_idx in range(tau, num_frames):
            if frame_idx % 5 == 0:
                print(f"\n  处理帧 {frame_idx}/{num_frames-1}...")
            
            # 准备输入
            past_states = prepare_past_states(
                tracked_states, trajectories, frame_idx, tau, max_targets, dt=0.1
            )
            current_meas = prepare_measurements(
                measurements[frame_idx], max_measurements
            )
            
            # 转换为tensor
            past_states_tensor = torch.FloatTensor(past_states).unsqueeze(0).to(device)
            current_meas_tensor = torch.FloatTensor(current_meas).unsqueeze(0).to(device)
            
            num_past_targets_list = []
            for t in range(frame_idx - tau, frame_idx):
                count = sum(1 for traj in trajectories 
                           if traj['birth_frame'] <= t <= traj['death_frame'])
                num_past_targets_list.append(count)
            num_past_targets_tensor = torch.LongTensor([num_past_targets_list]).to(device)
            
            num_current_meas = len(measurements[frame_idx])
            num_current_meas_tensor = torch.LongTensor([num_current_meas]).to(device)
            
            # 推理
            match_prob_matrix, filtered_states, existence_probs = model(
                past_states_tensor,
                current_meas_tensor,
                num_past_targets_tensor,
                num_current_meas_tensor
            )
            
            # 获取预测结果
            filtered_states_np = filtered_states[0].cpu().numpy()  # [max_targets, 2]
            match_prob_np = match_prob_matrix[0].cpu().numpy()  # [max_meas, max_targets+1]
            
            # 计算数据关联正确率
            pred_associations = np.argmax(match_prob_np[:num_current_meas, 1:], axis=1) + 1
            gt_assoc_frame = gt_associations[frame_idx][:num_current_meas]
            
            # 只计算非杂波的正确率
            valid_mask = gt_assoc_frame > 0
            if valid_mask.sum() > 0:
                correct = (pred_associations[valid_mask] == gt_assoc_frame[valid_mask]).sum()
                accuracy = correct / valid_mask.sum()
                all_association_accuracy.append(accuracy)
            
            # 根据匹配概率矩阵找到每个轨迹对应的测量
            frame_matched_meas = {}
            
            for traj_idx in range(min(len(trajectories), max_targets)):
                traj = trajectories[traj_idx]
                # 找到该轨迹概率最高的测量
                traj_probs = match_prob_np[:num_current_meas, traj_idx + 1]
                if len(traj_probs) > 0:
                    best_meas_idx = np.argmax(traj_probs)
                    matched_meas = measurements[frame_idx][best_meas_idx]
                    frame_matched_meas[traj_idx] = matched_meas
                else:
                    frame_matched_meas[traj_idx] = None
            
            # 更新跟踪状态
            for idx, traj in enumerate(trajectories):
                if idx < max_targets:
                    pred_state = filtered_states_np[idx]
                    tracked_states[traj['label']].append(pred_state)
                    if idx in frame_matched_meas:
                        tracked_measurements[traj['label']].append(frame_matched_meas[idx])
                    else:
                        tracked_measurements[traj['label']].append(None)
            
            # 记录该帧的状态
            frame_pred_states = []
            frame_true_states = []
            num_active_targets = 0
            for traj in trajectories:
                if traj['birth_frame'] <= frame_idx <= traj['death_frame']:
                    num_active_targets += 1
                    label = traj['label']
                    if label <= max_targets:
                        frame_pred_states.append(filtered_states_np[label - 1])
                    frame_true_states.append(traj['states'][frame_idx, :2])
            
            all_pred_states.append(frame_pred_states)
            all_true_states.append(frame_true_states)
            all_matched_measurements.append(frame_matched_meas)
            all_num_tracked.append(len(frame_pred_states))
    
    # 计算RMSE（反归一化到真实米数）
    COORD_SCALE = 50.0
    all_rmse = []
    for frame_idx in range(len(all_pred_states)):
        pred = np.array(all_pred_states[frame_idx])
        true = np.array(all_true_states[frame_idx])
        if len(pred) > 0 and len(true) > 0:
            # 反归一化到真实坐标
            pred_real = pred * COORD_SCALE
            true_real = true * COORD_SCALE
            # 计算每个点的误差（真实米数）
            errors = np.sqrt(np.sum((pred_real - true_real) ** 2, axis=1))
            all_rmse.append(np.mean(errors))
        else:
            all_rmse.append(0.0)
    
    # 生成可视化
    visualize_star_scenario(
        trajectories, measurements, tracked_states, tracked_measurements,
        tau, output_dir, num_frames
    )
    
    # 生成RMSE图
    plot_rmse(all_rmse, tau, output_dir, num_trajectories)
    
    # 计算统计结果
    avg_association_accuracy = np.mean(all_association_accuracy) if all_association_accuracy else 0.0
    avg_rmse = np.mean(all_rmse)
    
    print(f"\n{'='*60}")
    print(f"星形交叉轨迹场景结果:")
    print(f"{'='*60}")
    print(f"  轨迹数量: {len(trajectories)}")
    print(f"  平均每帧预测轨迹数: {np.mean(all_num_tracked):.1f}")
    print(f"  平均数据关联正确率: {avg_association_accuracy * 100:.2f}%")
    print(f"  平均RMSE: {avg_rmse:.4f} m")
    print(f"{'='*60}\n")
    
    return {
        'association_accuracy': all_association_accuracy,
        'rmse': all_rmse,
        'avg_association_accuracy': avg_association_accuracy,
        'avg_rmse': avg_rmse,
        'num_trajectories': len(trajectories)
    }


def prepare_past_states(tracked_states, trajectories, current_frame, tau, max_targets, dt):
    """准备过去tau帧的状态"""
    past_states = []
    
    for t in range(current_frame - tau, current_frame):
        for idx, traj in enumerate(trajectories):
            if idx >= max_targets:
                break
            
            label = traj['label']
            if traj['birth_frame'] <= t <= traj['death_frame']:
                if t < len(tracked_states[label]):
                    state_xy = tracked_states[label][t]
                    if state_xy is not None and not np.all(state_xy == 0):
                        state = np.array([label, state_xy[0], state_xy[1], t * dt])
                    else:
                        state = np.zeros(4)
                else:
                    state = np.zeros(4)
            else:
                state = np.zeros(4)
            
            past_states.append(state)
        
        # 填充到max_targets
        while len(past_states) < (t - (current_frame - tau) + 1) * max_targets:
            past_states.append(np.zeros(4))
    
    return np.array(past_states)


def prepare_measurements(measurements, max_measurements):
    """准备测量值"""
    meas = measurements.copy() if len(measurements) > 0 else np.empty((0, 2))
    
    if len(meas) < max_measurements:
        padding = np.zeros((max_measurements - len(meas), 2))
        meas = np.vstack([meas, padding]) if len(meas) > 0 else padding
    else:
        meas = meas[:max_measurements]
    
    return meas


def visualize_star_scenario(trajectories, measurements, tracked_states, tracked_measurements, 
                            tau, output_dir, num_frames):
    """可视化星形交叉轨迹场景：真实轨迹、预测轨迹、测量点"""
    plt.figure(figsize=(16, 14))
    
    # 🔧 反归一化因子
    COORD_SCALE = 50.0
    
    # 颜色映射 - 使用更多颜色
    colors = plt.cm.tab20(np.linspace(0, 1, len(trajectories)))
    
    print(f"  开始绘制轨迹...")
    
    # 绘制真实轨迹（虚线）- 反归一化到真实米数
    for idx, traj in enumerate(trajectories):
        states = traj['states'][:, :2] * COORD_SCALE  # 反归一化
        color = colors[idx]
        plt.plot(states[:, 0], states[:, 1], '--', color=color, 
                linewidth=2, alpha=0.6, label=f'真实轨迹 {traj["label"]}')
        # 标记起点
        plt.scatter(states[0, 0], states[0, 1], c=[color], s=150, 
                   marker='o', edgecolors='black', linewidths=2, zorder=5)
    
    # 绘制预测轨迹（实线）- 反归一化到真实米数
    num_predicted = 0
    for idx, traj in enumerate(trajectories):
        label = traj['label']
        if label in tracked_states:
            # 提取从tau帧开始的预测状态
            pred_states = []
            for i in range(tau, len(tracked_states[label])):
                state = tracked_states[label][i]
                if state is not None and not np.all(state == 0):
                    pred_states.append(state)
            
            if len(pred_states) > 0:
                pred_states = np.array(pred_states) * COORD_SCALE  # 反归一化
                color = colors[idx]
                plt.plot(pred_states[:, 0], pred_states[:, 1], '-', color=color, 
                        linewidth=3, alpha=1.0, label=f'预测轨迹 {label}')
                num_predicted += 1
    
    print(f"  总共绘制了 {num_predicted} 条预测轨迹")
    
    # 绘制所有测量点（灰色小点）- 反归一化到真实米数
    all_meas_x = []
    all_meas_y = []
    for frame_meas in measurements[tau:]:  # 从tau帧开始
        if len(frame_meas) > 0:
            frame_meas_denorm = frame_meas * COORD_SCALE  # 反归一化
            all_meas_x.extend(frame_meas_denorm[:, 0])
            all_meas_y.extend(frame_meas_denorm[:, 1])
    
    if len(all_meas_x) > 0:
        plt.scatter(all_meas_x, all_meas_y, c='gray', s=8, alpha=0.2, 
                   label='测量点', zorder=1)
    
    # 标记交叉中心区域
    circle = Circle((0, 0), 10, fill=False, color='red', linestyle=':', 
                   linewidth=2, label='交叉中心区域')
    plt.gca().add_patch(circle)
    
    plt.xlabel('X (m)', fontsize=16)
    plt.ylabel('Y (m)', fontsize=16)
    plt.title(f'星形交叉场景跟踪性能 ({len(trajectories)}条轨迹)', 
             fontsize=18, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(-55, 55)
    plt.ylim(-55, 55)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'star_crossing_{len(trajectories)}_trajectories.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存轨迹可视化: {output_path}")


def plot_rmse(rmse_values, tau, output_dir, num_trajectories):
    """绘制RMSE曲线"""
    frames = list(range(tau, tau + len(rmse_values)))
    
    plt.figure(figsize=(12, 7))
    plt.plot(frames, rmse_values, 'b-', linewidth=2.5, marker='o', markersize=6)
    plt.axhline(y=np.mean(rmse_values), color='r', linestyle='--', 
               linewidth=2, label=f'平均RMSE: {np.mean(rmse_values):.4f} m')
    
    # 标记交叉区域（约在中间帧）
    mid_frame = (tau + len(rmse_values)) // 2
    plt.axvspan(mid_frame - 2, mid_frame + 2, alpha=0.2, color='orange', 
               label='交叉区域')
    
    plt.xlabel('帧数', fontsize=16)
    plt.ylabel('RMSE (m)', fontsize=16)
    plt.title(f'星形交叉场景预测误差 ({num_trajectories}条轨迹)', 
             fontsize=18, fontweight='bold')
    plt.legend(fontsize=13)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'star_crossing_{num_trajectories}_rmse.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  保存RMSE图: {output_path}")


def main():
    args = parse_args()
    
    # 验证轨迹数量
    if args.num_trajectories < 4 or args.num_trajectories > 10:
        print("警告: 建议轨迹数量在4-10之间，将使用默认值6")
        args.num_trajectories = 6
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    print(f"\n加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    model = BAIT(
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_associate_decoder_layers=3,
        num_filtering_decoder_layers=6,
        dim_feedforward_encoder=2048,
        dim_feedforward_associate=1024,
        dim_feedforward_filtering=2048,
        max_targets=args.max_targets
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("模型加载成功！")
    
    print("\n" + "="*60)
    print("星形交叉轨迹场景测试")
    print("="*60)
    print("场景描述：")
    print(f"  - {args.num_trajectories}条轨迹，{args.num_frames}帧 ({args.num_frames * 0.1:.1f}秒)")
    print(f"  - 场景类型: {args.scenario_type} (使用训练数据生成器)")
    print(f"  - 轨迹从中心区域辐射出去，形成星形模式")
    print(f"  - 所有轨迹在中心区域交叉或接近")
    print(f"  - 100%检测率，平均8个杂波/帧")
    print(f"  - 随机种子: {args.seed} (可复现)")
    print("="*60 + "\n")
    
    # 评估星形场景
    results = evaluate_star_scenario(
        model, args.num_trajectories, args.tau, args.max_targets, 
        args.max_measurements, device, args.output_dir, 
        num_frames=args.num_frames, scenario_type=args.scenario_type, seed=args.seed
    )
    
    # 保存结果
    summary_results = {
        'scenario_type': 'star_crossing_trajectories_training',
        'generation_type': args.scenario_type,
        'num_trajectories': results['num_trajectories'],
        'num_frames': args.num_frames,
        'seed': args.seed,
        'avg_association_accuracy': float(results['avg_association_accuracy']),
        'avg_rmse': float(results['avg_rmse']),
        'description': f'Star crossing scenario with {results["num_trajectories"]} trajectories radiating from center (training-type, seed={args.seed})'
    }
    
    results_path = os.path.join(args.output_dir, f'star_crossing_{args.num_trajectories}_summary.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("评估完成！")
    print(f"{'='*60}")
    print(f"结果保存至: {results_path}")
    print(f"可视化保存至: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
