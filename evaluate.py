"""
BAIT模型评估脚本
在测试集上评估模型性能，生成详细的指标和可视化
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch

from bait_model import BAIT
from data_generation import MTTDataGenerator
from metrics import TrackingMetrics

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate BAIT model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--task-type', type=int, default=1, choices=[1, 2],
                        help='Task type')
    parser.add_argument('--num-scenarios', type=int, default=1,
                        help='Number of test scenarios')
    parser.add_argument('--tau', type=int, default=4,
                        help='Number of past frames')
    parser.add_argument('--max-targets', type=int, default=20,
                        help='Maximum number of targets')
    parser.add_argument('--max-measurements', type=int, default=30,
                        help='Maximum number of measurements')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Directory to save results')
    
    return parser.parse_args()


def evaluate_single_scenario(model, generator, tau, max_targets, max_measurements, device, scenario_idx, output_dir):
    """
    评估单个场景并生成详细的可视化和指标
    
    Returns:
        scenario_results: dict，包含该场景的所有评估结果
    """
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"评估场景 {scenario_idx + 1}")
    print(f"{'='*60}")
    
    # 生成场景
    trajectories, measurements, gt_associations = generator.generate_single_scenario()
    num_frames = len(measurements)
    
    # 🔧 归一化坐标（与data_generation.py中的MTTDataset保持一致）
    COORD_SCALE = 50.0  # 场景范围 [-50, 50]m
    for traj in trajectories:
        traj['states'][:, :2] = traj['states'][:, :2] / COORD_SCALE  # 归一化 x, y
    for frame_meas in measurements:
        frame_meas[:] = frame_meas / COORD_SCALE  # 归一化测量
    
    print(f"目标数量: {len(trajectories)}")
    print(f"帧数: {num_frames}")
    
    # 🔧 初始化跟踪状态 - 使用测量值而非真实状态
    # tracked_states[label] 的索引对应帧号，即 tracked_states[label][t] 对应第t帧的状态
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
            print(f"\n  --- 帧 {frame_idx} ---")
            # 准备输入
            past_states = prepare_past_states(
                tracked_states, trajectories, frame_idx, tau, max_targets, generator.dt
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
            # 说明：对每个测量，找到其在match_prob_matrix中概率最高的轨迹
            pred_associations = np.argmax(match_prob_np[:num_current_meas, 1:], axis=1) + 1  # 排除clutter列(索引0)
            gt_assoc_frame = gt_associations[frame_idx][:num_current_meas]
            
            # 只计算非杂波的正确率（gt_assoc_frame > 0表示真实测量）
            valid_mask = gt_assoc_frame > 0
            if valid_mask.sum() > 0:
                correct = (pred_associations[valid_mask] == gt_assoc_frame[valid_mask]).sum()
                accuracy = correct / valid_mask.sum()
                all_association_accuracy.append(accuracy)
                
                # 每一帧都打印数据关联信息
                print(f"  数据关联结果:")
                print(f"    总测量数: {num_current_meas}, 真实测量: {valid_mask.sum()}, 杂波: {(~valid_mask).sum()}")
                
                # 打印真实测量的关联情况
                COORD_SCALE = 50.0
                for i, (pred, gt) in enumerate(zip(pred_associations[valid_mask], gt_assoc_frame[valid_mask])):
                    match_status = "✓" if pred == gt else "✗"
                    meas_pos = measurements[frame_idx][np.where(valid_mask)[0][i]]
                    meas_pos_real = meas_pos * COORD_SCALE  # 反归一化
                    print(f"    测量{i+1} {match_status}: 位置({meas_pos_real[0]:.2f},{meas_pos_real[1]:.2f})m -> 预测:轨迹{pred}, 真实:轨迹{gt}")
                
                print(f"    正确率: {correct}/{valid_mask.sum()} = {accuracy*100:.1f}%")
            
            # 根据匹配概率矩阵找到每个轨迹对应的测量
            frame_matched_meas = {}
            print(f"  轨迹匹配的测量:")
            
            # 检查是否有轨迹在该帧没有真实测量
            gt_labels_in_frame = set(gt_associations[frame_idx][gt_associations[frame_idx] > 0])
            
            for traj_idx in range(min(len(trajectories), max_targets)):
                traj = trajectories[traj_idx]
                # 找到该轨迹概率最高的测量
                traj_probs = match_prob_np[:num_current_meas, traj_idx + 1]
                if len(traj_probs) > 0:
                    best_meas_idx = np.argmax(traj_probs)
                    best_prob = traj_probs[best_meas_idx]
                    matched_meas = measurements[frame_idx][best_meas_idx]
                    
                    # 检查该轨迹是否应该有测量（是否被检测到）
                    has_detection = traj['label'] in gt_labels_in_frame
                    
                    # 如果概率很低，可能不应该匹配
                    if best_prob < 0.1:
                        frame_matched_meas[traj_idx] = None  # 概率太低，不匹配
                        status = "⚠ 低概率"
                    else:
                        frame_matched_meas[traj_idx] = matched_meas
                        status = "✓" if gt_associations[frame_idx][best_meas_idx] == traj['label'] else "✗"
                    
                    # 找到这个测量的真实关联
                    true_label = gt_associations[frame_idx][best_meas_idx]
                    detection_status = f", 该帧{'有' if has_detection else '无'}真实测量"
                    
                    # 反归一化显示
                    COORD_SCALE = 50.0
                    matched_meas_real = matched_meas * COORD_SCALE
                    print(f"    轨迹{traj['label']} {status}: 选择测量{best_meas_idx} ({matched_meas_real[0]:.2f},{matched_meas_real[1]:.2f})m, " +
                          f"概率={best_prob:.3f}, 真实标签={true_label}{detection_status}")
                else:
                    frame_matched_meas[traj_idx] = None
                    print(f"    轨迹{traj['label']}: 无测量匹配")
            
            # 更新跟踪状态
            print(f"  滤波后的状态:")
            COORD_SCALE = 50.0
            for idx, traj in enumerate(trajectories):
                if idx < max_targets:
                    pred_state = filtered_states_np[idx]
                    true_state = traj['states'][frame_idx, :2]
                    # 反归一化到真实坐标显示
                    pred_state_real = pred_state * COORD_SCALE
                    true_state_real = true_state * COORD_SCALE
                    error = np.sqrt(np.sum((pred_state_real - true_state_real) ** 2))
                    
                    # 检查预测点是否在变化
                    if len(tracked_states[traj['label']]) > tau:
                        prev_state = tracked_states[traj['label']][-1]
                        prev_state_real = prev_state * COORD_SCALE
                        state_change = np.sqrt(np.sum((pred_state_real - prev_state_real) ** 2))
                        print(f"    轨迹{traj['label']}: 预测({pred_state_real[0]:.2f},{pred_state_real[1]:.2f}), " +
                              f"真实({true_state_real[0]:.2f},{true_state_real[1]:.2f}), " +
                              f"误差={error:.2f}m, 变化={state_change:.2f}m")
                    else:
                        print(f"    轨迹{traj['label']}: 预测({pred_state_real[0]:.2f},{pred_state_real[1]:.2f}), " +
                              f"真实({true_state_real[0]:.2f},{true_state_real[1]:.2f}), " +
                              f"误差={error:.2f}m")
                    
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
    
    # 打印每帧跟踪的轨迹数和预测点诊断信息
    print(f"\n  每帧预测轨迹数:")
    for i, (frame_idx, num_tracked) in enumerate(zip(range(tau, num_frames), all_num_tracked)):
        if i % 5 == 0:  # 每5帧打印一次
            print(f"    帧 {frame_idx}: {num_tracked} 条")
    
    # 诊断：检查预测点的分布
    print(f"\n  预测轨迹诊断:")
    COORD_SCALE = 50.0
    for idx, traj in enumerate(trajectories[:3]):  # 只检查前3条轨迹
        label = traj['label']
        if label in tracked_states and len(tracked_states[label]) > tau:
            pred_points = np.array([s for s in tracked_states[label][tau:] if s is not None and not np.all(s == 0)])
            true_points = traj['states'][tau:, :2]
            
            if len(pred_points) > 0:
                # 反归一化到真实米数
                pred_points_real = pred_points * COORD_SCALE
                true_points_real = true_points * COORD_SCALE
                
                pred_range_x = pred_points_real[:, 0].max() - pred_points_real[:, 0].min()
                pred_range_y = pred_points_real[:, 1].max() - pred_points_real[:, 1].min()
                true_range_x = true_points_real[:, 0].max() - true_points_real[:, 0].min()
                true_range_y = true_points_real[:, 1].max() - true_points_real[:, 1].min()
                
                print(f"    轨迹 {label}:")
                print(f"      预测范围: X={pred_range_x:.2f}m, Y={pred_range_y:.2f}m")
                print(f"      真实范围: X={true_range_x:.2f}m, Y={true_range_y:.2f}m")
                print(f"      预测首点: ({pred_points_real[0, 0]:.2f}, {pred_points_real[0, 1]:.2f})m")
                print(f"      预测末点: ({pred_points_real[-1, 0]:.2f}, {pred_points_real[-1, 1]:.2f})m")
                print(f"      真实首点: ({true_points_real[0, 0]:.2f}, {true_points_real[0, 1]:.2f})m")
                print(f"      真实末点: ({true_points_real[-1, 0]:.2f}, {true_points_real[-1, 1]:.2f})m")
    
    # 生成可视化
    visualize_scenario(
        trajectories, measurements, tracked_states, tracked_measurements,
        tau, scenario_idx, output_dir, num_frames
    )
    
    # 生成RMSE图
    plot_rmse(all_rmse, tau, scenario_idx, output_dir)
    
    # 计算统计结果
    avg_association_accuracy = np.mean(all_association_accuracy) if all_association_accuracy else 0.0
    avg_rmse = np.mean(all_rmse)
    
    print(f"\n场景 {scenario_idx + 1} 结果:")
    print(f"  总目标数: {len(trajectories)}")
    print(f"  平均每帧预测轨迹数: {np.mean(all_num_tracked):.1f}")
    print(f"  平均数据关联正确率: {avg_association_accuracy * 100:.2f}%")
    print(f"  平均RMSE: {avg_rmse:.4f} m")
    
    # 航迹管理说明
    if scenario_idx == 0:
        print(f"\n{'='*60}")
        print("航迹管理说明:")
        print(f"{'='*60}")
        print("当前实现:")
        print("  1. 轨迹初始化: 使用前4帧的真实测量值初始化")
        print("  2. 轨迹维持: 所有轨迹从开始到结束全程存在")
        print("  3. 轨迹终止: 无终止机制，所有轨迹持续到最后一帧")
        print("  4. 标签分配: 预先由数据生成器分配，不动态管理")
        print("\n限制:")
        print("  - 无轨迹确认/删除逻辑")
        print("  - 无新生目标检测")
        print("  - 依赖外部提供的初始标签")
        print(f"{'='*60}\n")
    
    return {
        'association_accuracy': all_association_accuracy,
        'rmse': all_rmse,
        'avg_association_accuracy': avg_association_accuracy,
        'avg_rmse': avg_rmse
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
                # 🔧 修复：使用正确的索引 - 从tracked_states列表末尾向前数
                # tracked_states[label] 存储了从第0帧到当前的所有状态
                # 我们需要第t帧的状态，即索引为t的元素
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


def visualize_scenario(trajectories, measurements, tracked_states, tracked_measurements, 
                       tau, scenario_idx, output_dir, num_frames):
    """可视化单个场景：真实轨迹、预测轨迹、测量点"""
    plt.figure(figsize=(14, 12))
    
    # 🔧 反归一化因子
    COORD_SCALE = 50.0
    
    # 颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    print(f"  开始绘制轨迹...")
    
    # 绘制真实轨迹（虚线）- 反归一化到真实米数
    for idx, traj in enumerate(trajectories):
        states = traj['states'][:, :2] * COORD_SCALE  # 反归一化
        color = colors[idx]
        plt.plot(states[:, 0], states[:, 1], '--', color=color, 
                linewidth=2, alpha=0.7, label=f'Ground Truth {traj["label"]}')
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
                        linewidth=3, alpha=1.0, label=f'Prediction {label}')
                num_predicted += 1
                print(f"    轨迹 {label}: 预测了 {len(pred_states)} 个点")
            else:
                print(f"    轨迹 {label}: 没有有效预测点")
    
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
                   label='Measurements', zorder=1)
    
    plt.xlabel('X (m)', fontsize=14)
    plt.ylabel('Y (m)', fontsize=14)
    plt.title(f'Scenario {scenario_idx + 1} - Tracking Comparison', fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'scenario_{scenario_idx + 1}_trajectory.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存轨迹可视化: {output_path}")


def plot_rmse(rmse_values, tau, scenario_idx, output_dir):
    """绘制RMSE曲线"""
    frames = list(range(tau, tau + len(rmse_values)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(frames, rmse_values, 'b-', linewidth=2, marker='o', markersize=4)
    plt.axhline(y=np.mean(rmse_values), color='r', linestyle='--', 
               linewidth=2, label=f'Mean RMSE: {np.mean(rmse_values):.4f} m')
    
    plt.xlabel('Frame', fontsize=14)
    plt.ylabel('RMSE (m)', fontsize=14)
    plt.title(f'Scenario {scenario_idx + 1} - Prediction Error (RMSE)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'scenario_{scenario_idx + 1}_rmse.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  保存RMSE图: {output_path}")


def main():
    args = parse_args()
    
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
    
    # 创建数据生成器（明确指定P_d=1.0确保所有目标都被检测）
    generator = MTTDataGenerator(
        task_type=args.task_type,
        field_of_view=100.0,  # [-50, 50]m 范围
        lambda_0=3,  # 与训练时一致
        P_d=1.0,     # 确保100%检测
        seed=998
    )
    
    # 评估每个场景
    all_results = []
    for scenario_idx in range(args.num_scenarios):
        scenario_results = evaluate_single_scenario(
            model, generator, args.tau, args.max_targets, 
            args.max_measurements, device, scenario_idx, args.output_dir
        )
        all_results.append(scenario_results)
    
    # 汇总统计
    print(f"\n{'='*60}")
    print("Overall Evaluation Summary")
    print(f"{'='*60}")
    print(f"Task Type: Task {args.task_type}")
    print(f"Test Scenarios: {args.num_scenarios}")
    print(f"\nData Association Accuracy:")
    for idx, res in enumerate(all_results):
        print(f"  Scenario {idx + 1}: {res['avg_association_accuracy'] * 100:.2f}%")
    
    overall_assoc_acc = np.mean([r['avg_association_accuracy'] for r in all_results])
    print(f"  Overall Average: {overall_assoc_acc * 100:.2f}%")
    
    print(f"\nPrediction RMSE:")
    for idx, res in enumerate(all_results):
        print(f"  Scenario {idx + 1}: {res['avg_rmse']:.4f} m")
    
    overall_rmse = np.mean([r['avg_rmse'] for r in all_results])
    print(f"  Overall Average: {overall_rmse:.4f} m")
    print(f"{'='*60}\n")
    
    # 保存结果
    summary_results = {
        'task_type': args.task_type,
        'num_scenarios': args.num_scenarios,
        'scenarios': [
            {
                'scenario_id': idx + 1,
                'avg_association_accuracy': float(res['avg_association_accuracy']),
                'avg_rmse': float(res['avg_rmse'])
            }
            for idx, res in enumerate(all_results)
        ],
        'overall_summary': {
            'avg_association_accuracy': float(overall_assoc_acc),
            'avg_rmse': float(overall_rmse)
        }
    }
    
    results_path = os.path.join(args.output_dir, 'evaluation_summary.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to: {results_path}")
    print(f"Visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
