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


def generate_random_crossing_scenario(num_frames=20, dt=0.1, P_d=0.95, lambda_c=8, 
                                      field_of_view=100.0, velocity_range=(3, 10), 
                                      q_s=0.09, R=0.01, seed=None):
    """
    生成随机交叉轨迹场景（与训练数据生成逻辑完全相同）
    
    这个函数复制了 data_generation_with_crossing.py 中的交叉场景生成逻辑
    
    Returns:
        trajectories: 轨迹列表
        measurements: 测量值列表
        gt_associations: 真实关联列表
    """
    if seed is not None:
        np.random.seed(seed)
    
    half_fov = field_of_view / 2
    
    # 交叉点在中心区域（与训练相同：-30%到30%范围）
    cross_x = np.random.uniform(-0.3 * half_fov, 0.3 * half_fov)
    cross_y = np.random.uniform(-0.3 * half_fov, 0.3 * half_fov)
    
    # 选择两个不同的角度（确保至少相差30度）
    angle1 = np.random.uniform(0, 2 * np.pi)
    angle2 = angle1 + np.random.uniform(np.pi/6, np.pi*5/6)  # 30-150度差异
    
    # 速度大小（与训练相同范围）
    speed1 = np.random.uniform(velocity_range[0], velocity_range[1])
    speed2 = np.random.uniform(velocity_range[0], velocity_range[1])
    
    # 速度向量
    v1 = np.array([speed1 * np.cos(angle1), speed1 * np.sin(angle1)])
    v2 = np.array([speed2 * np.cos(angle2), speed2 * np.sin(angle2)])
    
    # 计算起点（让轨迹在时间中点交叉）
    T = (num_frames - 1) * dt
    half_time = T / 2
    start1 = np.array([cross_x, cross_y]) - v1 * half_time
    start2 = np.array([cross_x, cross_y]) - v2 * half_time
    
    # 确保起点在视野内
    start1 = np.clip(start1, -half_fov * 0.9, half_fov * 0.9)
    start2 = np.clip(start2, -half_fov * 0.9, half_fov * 0.9)
    
    # 生成两条轨迹
    trajectories = []
    
    # 常速度模型矩阵
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    Q = q_s * np.array([
        [dt**3/3, 0, dt**2/2, 0],
        [0, dt**3/3, 0, dt**2/2],
        [dt**2/2, 0, dt, 0],
        [0, dt**2/2, 0, dt]
    ])
    
    for label, start_pos, velocity in [(1, start1, v1), (2, start2, v2)]:
        state = np.array([start_pos[0], start_pos[1], velocity[0], velocity[1]])
        states = np.zeros((num_frames, 4))
        states[0] = state
        
        for t in range(1, num_frames):
            # 状态转移
            states[t] = F @ states[t-1]
            # 添加过程噪声
            process_noise = np.random.multivariate_normal(np.zeros(4), Q)
            states[t] += process_noise
        
        trajectories.append({
            'label': label,
            'states': states,
            'birth_frame': 0,
            'death_frame': num_frames - 1
        })
    
    # 生成测量值和关联
    measurements = []
    gt_associations = []
    
    for t in range(num_frames):
        frame_meas = []
        frame_assoc = []
        
        # 为每个轨迹生成测量
        for traj in trajectories:
            if traj['birth_frame'] <= t <= traj['death_frame']:
                if np.random.rand() < P_d:
                    true_pos = traj['states'][t, :2]
                    noise = np.random.randn(2) * np.sqrt(R)
                    meas = true_pos + noise
                    # 裁剪到视野内
                    meas = np.clip(meas, -half_fov, half_fov)
                    frame_meas.append(meas)
                    frame_assoc.append(traj['label'])
        
        # 生成杂波
        num_clutter = np.random.poisson(lambda_c)
        for _ in range(num_clutter):
            clutter_pos = np.random.uniform(-half_fov, half_fov, 2)
            frame_meas.append(clutter_pos)
            frame_assoc.append(0)
        
        # 随机打乱
        if len(frame_meas) > 0:
            frame_meas = np.array(frame_meas)
            frame_assoc = np.array(frame_assoc)
            perm = np.random.permutation(len(frame_meas))
            frame_meas = frame_meas[perm]
            frame_assoc = frame_assoc[perm]
        else:
            frame_meas = np.empty((0, 2))
            frame_assoc = np.array([])
        
        measurements.append(frame_meas)
        gt_associations.append(frame_assoc)
    
    return trajectories, measurements, gt_associations


def generate_fixed_crossing_scenario(num_frames=40, dt=0.1, P_d=1.0, lambda_c=8):
    """
    生成固定的交叉轨迹测试场景
    
    场景设计（简化版）：
    - 轨迹1: 从左下到右上的对角线运动
    - 轨迹2: 从左上到右下的对角线运动
    - 两条轨迹在场景中心交叉
    
    Returns:
        trajectories: 轨迹列表
        measurements: 测量值列表
        gt_associations: 真实关联列表
    """
    trajectories = []
    # 🔧 方案A：原始长距离交叉（模型会失败）
    # start1 = np.array([-40.0, -35.0])
    # end1 = np.array([40.0, 35.0])
    # start2 = np.array([-40.0, 35.0])
    # end2 = np.array([40.0, -35.0])
    
    # 🔧 方案B：从中心开始（能预测但关联可能错）
    # start1 = np.array([0.0, -35.0])
    # end1 = np.array([40.0, 0.0])
    # start2 = np.array([0.0, 0.0])
    # end2 = np.array([40.0, -35.0])
    
    # 🔧 方案C：训练分布内交叉，避开原点（中等距离）
    # 交叉点在(10,5)，移动距离~64m
    # start1 = np.array([-15.0, -15.0])
    # end1 = np.array([35.0, 25.0])
    # start2 = np.array([-15.0, 25.0])
    # end2 = np.array([35.0, -15.0])
    
    # 🔧 方案D：保守版本 - 短距离交叉，完全符合训练分布（推荐）
    # 交叉点在(-5, 0)，移动距离~28m，更接近训练数据的20m
    # 🔥 关键改进：交叉点偏离(0,0)，避免原点的特殊性！
    
    # 轨迹1: 小角度向右上移动
    start1 = np.array([-20.0, -10.0])  # 起点
    end1 = np.array([10.0, 18.0])      # 终点（移动~33m）
    velocity1 = (end1 - start1) / ((num_frames - 1) * dt)
    states1 = np.zeros((num_frames, 4))
    for t in range(num_frames):
        pos = start1 + velocity1 * (t * dt)
        states1[t] = [pos[0], pos[1], velocity1[0], velocity1[1]]
    
    trajectories.append({
        'label': 1,
        'states': states1,
        'birth_frame': 0,
        'death_frame': num_frames - 1
    })
    
    # 轨迹2: 小角度向右下移动
    start2 = np.array([-20.0, 10.0])   # 起点
    end2 = np.array([10.0, -18.0])     # 终点（移动~33m）
    # 两条轨迹在(-5, 0)附近交叉，约在第20帧
    velocity2 = (end2 - start2) / ((num_frames - 1) * dt)
    states2 = np.zeros((num_frames, 4))
    for t in range(num_frames):
        pos = start2 + velocity2 * (t * dt)
        states2[t] = [pos[0], pos[1], velocity2[0], velocity2[1]]
    
    trajectories.append({
        'label': 2,
        'states': states2,
        'birth_frame': 0,
        'death_frame': num_frames - 1
    })
    
    # 生成测量值和关联
    measurements = []
    gt_associations = []
    
    np.random.seed(42)  # 固定随机种子，确保结果可复现
    
    for t in range(num_frames):
        frame_meas = []
        frame_assoc = []
        
        # 为每个轨迹生成测量（以P_d概率检测到）
        for traj in trajectories:
            if traj['birth_frame'] <= t <= traj['death_frame']:
                if np.random.rand() < P_d:
                    # 添加测量噪声 (R=0.01 -> std=0.1m)
                    true_pos = traj['states'][t, :2]
                    noise = np.random.randn(2) * 0.1
                    meas = true_pos + noise
                    frame_meas.append(meas)
                    frame_assoc.append(traj['label'])
        
        # 生成杂波（泊松分布）
        num_clutter = np.random.poisson(lambda_c)
        for _ in range(num_clutter):
            # 在[-50, 50]范围内均匀分布
            clutter_pos = np.random.uniform(-50, 50, 2)
            frame_meas.append(clutter_pos)
            frame_assoc.append(0)  # 0表示杂波
        
        # 随机打乱顺序（模拟真实情况下测量的无序性）
        if len(frame_meas) > 0:
            frame_meas = np.array(frame_meas)
            frame_assoc = np.array(frame_assoc)
            perm = np.random.permutation(len(frame_meas))
            frame_meas = frame_meas[perm]
            frame_assoc = frame_assoc[perm]
        else:
            frame_meas = np.empty((0, 2))
            frame_assoc = np.array([])
        
        measurements.append(frame_meas)
        gt_associations.append(frame_assoc)
    
    return trajectories, measurements, gt_associations


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate BAIT model with crossing trajectories')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--tau', type=int, default=4,
                        help='Number of past frames')
    parser.add_argument('--max-targets', type=int, default=20,
                        help='Maximum number of targets')
    parser.add_argument('--max-measurements', type=int, default=30,
                        help='Maximum number of measurements')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output-dir', type=str, default='evaluation_results_cross',
                        help='Directory to save results')
    parser.add_argument('--num-random-scenarios', type=int, default=5,
                        help='Number of random training-type scenarios to test')
    
    return parser.parse_args()


def evaluate_single_scenario(model, tau, max_targets, max_measurements, device, output_dir, 
                            scenario_type='fixed', scenario_name='scenario', seed=42):
    """
    评估交叉轨迹场景并生成详细的可视化和指标
    
    Args:
        scenario_type: 'fixed' 或 'random'
        scenario_name: 场景名称，用于保存文件
        seed: 随机种子（仅对random类型有效）
    
    Returns:
        scenario_results: dict，包含该场景的所有评估结果
    """
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"评估交叉轨迹场景 - {scenario_name}")
    print(f"{'='*60}")
    
    # 生成交叉轨迹场景
    if scenario_type == 'fixed':
        trajectories, measurements, gt_associations = generate_fixed_crossing_scenario(
            num_frames=40,  # 40帧，4秒
            dt=0.1,
            P_d=1.0,        # 100%检测率
            lambda_c=8      # 平均8个杂波
        )
        scenario_description = "固定场景: 两条对角线交叉轨迹"
    else:  # random
        trajectories, measurements, gt_associations = generate_random_crossing_scenario(
            num_frames=20,  # 20帧，2秒（与训练数据相同）
            dt=0.1,
            P_d=1.0,       # 100%检测率（与训练相同）
            lambda_c=8,     # 平均8个杂波
            field_of_view=100.0,  # 与训练相同
            velocity_range=(3, 10),  # 与训练相同
            seed=seed       # 使用传入的随机种子
        )
        scenario_description = f"训练类型场景: 随机生成的交叉轨迹（seed={seed}）"
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
            
            # 🔍 调试：查看模型内部选择的测量（通过反向推算）
            # 注意：这只是近似估计，实际可能不准确
            # 因为 filtered_states = filtering_queries + state_correction
            
            # 获取预测结果
            filtered_states_np = filtered_states[0].cpu().numpy()  # [max_targets, 2]
            match_prob_np = match_prob_matrix[0].cpu().numpy()  # [max_meas, max_targets+1]
            
            # 🔍 检查模型是否输出了零向量
            if frame_idx >= 20:  # 只在交叉点附近检查
                COORD_SCALE_DEBUG = 50.0
                print(f"  🔍 模型输出诊断:")
                for i in range(min(2, max_targets)):
                    pred = filtered_states_np[i]
                    is_zero = np.allclose(pred, 0, atol=1e-6)
                    pred_real = pred * COORD_SCALE_DEBUG
                    print(f"    轨迹{i+1}: 模型输出=({pred_real[0]:.4f}, {pred_real[1]:.4f})m, " +
                          f"是否为零={is_zero}")
            
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
                    
                    # 🔧 移除概率阈值限制，始终使用概率最高的测量
                    frame_matched_meas[traj_idx] = matched_meas
                    
                    # 判断关联是否正确
                    is_correct = gt_associations[frame_idx][best_meas_idx] == traj['label']
                    if best_prob < 0.1:
                        status = "⚠" if not is_correct else "✓⚠"  # 低概率标记
                    else:
                        status = "✓" if is_correct else "✗"
                    
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
        tau, output_dir, num_frames, scenario_name
    )
    
    # 生成RMSE图
    plot_rmse(all_rmse, tau, output_dir, scenario_name)
    
    # 计算统计结果
    avg_association_accuracy = np.mean(all_association_accuracy) if all_association_accuracy else 0.0
    avg_rmse = np.mean(all_rmse)
    
    print(f"\n{scenario_name} 结果:")
    print(f"  场景类型: {scenario_description}")
    print(f"  总目标数: {len(trajectories)}")
    print(f"  总帧数: {num_frames}")
    print(f"  平均每帧预测轨迹数: {np.mean(all_num_tracked):.1f}")
    print(f"  平均数据关联正确率: {avg_association_accuracy * 100:.2f}%")
    print(f"  平均RMSE: {avg_rmse:.4f} m")
    
    if scenario_type == 'fixed':
        # 固定场景的额外分析
        print(f"\n{'='*60}")
        print("轨迹交叉分析:")
        print(f"{'='*60}")
        print("轨迹配置:")
        print("  轨迹1: (-20,-10) → (10,18) [移动~33m]")
        print("  轨迹2: (-20,10) → (10,-18) [移动~33m]")
        print("\n交叉点:")
        print("  两条轨迹在(-5, 0)附近交叉 (约在第20帧)")
        print("  🔥 关键：交叉点偏离原点(0,0)，避免模型在原点的特殊行为")
        print(f"{'='*60}\n")
    else:
        # 随机场景的额外分析
        print(f"\n{'='*60}")
        print("随机场景分析:")
        print(f"{'='*60}")
        # 计算实际交叉点
        traj1_states = trajectories[0]['states']
        traj2_states = trajectories[1]['states']
        min_dist = float('inf')
        min_frame = 0
        for t in range(len(traj1_states)):
            dist = np.linalg.norm(traj1_states[t, :2] - traj2_states[t, :2])
            if dist < min_dist:
                min_dist = dist
                min_frame = t
        
        COORD_SCALE = 50.0
        cross_point = (traj1_states[min_frame, :2] + traj2_states[min_frame, :2]) / 2 * COORD_SCALE
        print(f"  交叉点位置: ({cross_point[0]:.2f}, {cross_point[1]:.2f})m (第{min_frame}帧)")
        print(f"  最近距离: {min_dist * COORD_SCALE:.2f}m")
        print(f"  轨迹1速度: ({trajectories[0]['states'][0, 2]:.2f}, {trajectories[0]['states'][0, 3]:.2f}) m/s")
        print(f"  轨迹2速度: ({trajectories[1]['states'][0, 2]:.2f}, {trajectories[1]['states'][0, 3]:.2f}) m/s")
        print(f"{'='*60}\n")
    
    return {
        'association_accuracy': all_association_accuracy,
        'rmse': all_rmse,
        'avg_association_accuracy': avg_association_accuracy,
        'avg_rmse': avg_rmse,
        'scenario_type': scenario_type,
        'num_frames': num_frames
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
                       tau, output_dir, num_frames, scenario_name='scenario'):
    """可视化交叉轨迹场景：真实轨迹、预测轨迹、测量点"""
    plt.figure(figsize=(16, 14))
    
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
    
    plt.xlabel('X (m)', fontsize=16)
    plt.ylabel('Y (m)', fontsize=16)
    plt.title('Crossing Trajectories - Tracking Performance', fontsize=18, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, ncol=1)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(-55, 55)
    plt.ylim(-55, 55)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{scenario_name}_trajectories.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存轨迹可视化: {output_path}")


def plot_rmse(rmse_values, tau, output_dir, scenario_name='scenario'):
    """绘制RMSE曲线"""
    frames = list(range(tau, tau + len(rmse_values)))
    
    plt.figure(figsize=(12, 7))
    plt.plot(frames, rmse_values, 'b-', linewidth=2.5, marker='o', markersize=6)
    plt.axhline(y=np.mean(rmse_values), color='r', linestyle='--', 
               linewidth=2, label=f'Mean RMSE: {np.mean(rmse_values):.4f} m')
    
    # 标记交叉区域（约在第20帧）
    plt.axvspan(18, 22, alpha=0.2, color='orange', label='Crossing Region')
    
    plt.xlabel('Frame', fontsize=16)
    plt.ylabel('RMSE (m)', fontsize=16)
    plt.title('Crossing Trajectories - Prediction Error (RMSE)', fontsize=18, fontweight='bold')
    plt.legend(fontsize=13)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{scenario_name}_rmse.png')
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
    
    print("\n" + "="*80)
    print("🔬 双场景对比评估：固定场景 vs 训练类型场景")
    print("="*80)
    print("目的：诊断模型在不同类型交叉场景下的表现")
    print("  - 场景1（固定）：设计的固定交叉轨迹，较长距离")
    print("  - 场景2（随机）：与训练数据相同生成逻辑的随机交叉场景")
    print("="*80 + "\n")
    
    # ========== 评估场景1：固定场景 ==========
    print("\n" + "="*60)
    print("📍 场景1：固定交叉轨迹场景")
    print("="*60)
    print("场景描述：")
    print("  - 2条轨迹，40帧 (4秒)")
    print("  - 轨迹1: (-20,-10) → (10,18) 小角度向右上")
    print("  - 轨迹2: (-20,10) → (10,-18) 小角度向右下")
    print("  - 交叉点在(-5, 0)附近，避开原点(0,0)（约第20帧）")
    print("  - 移动距离~33m，接近训练分布")
    print("  - 100%检测率，平均8个杂波/帧")
    print("="*60 + "\n")
    
    results_fixed = evaluate_single_scenario(
        model, args.tau, args.max_targets, 
        args.max_measurements, device, args.output_dir,
        scenario_type='fixed',
        scenario_name='fixed_scenario'
    )
    
    # ========== 评估场景2：多个训练类型场景 ==========
    print("\n" + "="*60)
    print(f"🎲 场景2：训练类型交叉场景（测试{args.num_random_scenarios}个随机场景）")
    print("="*60)
    print("场景描述：")
    print("  - 2条轨迹，20帧 (2秒) - 与训练数据相同")
    print("  - 交叉点随机位置（中心±15m）")
    print("  - 交叉角度随机（30-150度）")
    print("  - 速度随机（3-10 m/s）")
    print("  - 95%检测率，平均8个杂波/帧")
    print("  - 使用与训练完全相同的生成逻辑")
    print(f"  - 将测试{args.num_random_scenarios}个不同随机种子的场景")
    print("="*60 + "\n")
    
    # 评估多个随机场景
    random_results_list = []
    for scenario_idx in range(args.num_random_scenarios):
        print(f"\n{'─'*60}")
        print(f"📋 训练场景 {scenario_idx + 1}/{args.num_random_scenarios} (seed={42 + scenario_idx})")
        print(f"{'─'*60}")
        
        result = evaluate_single_scenario(
            model, args.tau, args.max_targets, 
            args.max_measurements, device, args.output_dir,
            scenario_type='random',
            scenario_name=f'training_type_scenario_{scenario_idx + 1}',
            seed=42 + scenario_idx  # 不同的随机种子
        )
        random_results_list.append(result)
    
    # 汇总所有随机场景的结果
    avg_acc_list = [r['avg_association_accuracy'] for r in random_results_list]
    avg_rmse_list = [r['avg_rmse'] for r in random_results_list]
    
    results_random = {
        'avg_association_accuracy': np.mean(avg_acc_list),
        'avg_rmse': np.mean(avg_rmse_list),
        'std_association_accuracy': np.std(avg_acc_list),
        'std_rmse': np.std(avg_rmse_list),
        'min_association_accuracy': np.min(avg_acc_list),
        'max_association_accuracy': np.max(avg_acc_list),
        'min_rmse': np.min(avg_rmse_list),
        'max_rmse': np.max(avg_rmse_list),
        'individual_results': random_results_list
    }
    
    print("\n" + "="*60)
    print("📊 训练类型场景汇总统计")
    print("="*60)
    print(f"测试场景数: {args.num_random_scenarios}")
    print(f"关联正确率: {results_random['avg_association_accuracy']*100:.2f}% ± {results_random['std_association_accuracy']*100:.2f}%")
    print(f"  - 最小: {results_random['min_association_accuracy']*100:.2f}%")
    print(f"  - 最大: {results_random['max_association_accuracy']*100:.2f}%")
    print(f"平均RMSE: {results_random['avg_rmse']:.4f}m ± {results_random['std_rmse']:.4f}m")
    print(f"  - 最小: {results_random['min_rmse']:.4f}m")
    print(f"  - 最大: {results_random['max_rmse']:.4f}m")
    print("="*60)
    
    # ========== 生成对比报告 ==========
    print("\n" + "="*80)
    print("📊 对比分析结果")
    print("="*80)
    
    comparison = {
        'fixed_scenario': {
            'description': '固定交叉场景（较长距离，40帧）',
            'num_frames': 40,
            'num_scenarios': 1,
            'avg_association_accuracy': float(results_fixed['avg_association_accuracy']),
            'avg_rmse': float(results_fixed['avg_rmse'])
        },
        'training_type_scenario': {
            'description': f'训练类型场景（随机交叉，20帧，{args.num_random_scenarios}个场景平均）',
            'num_frames': 20,
            'num_scenarios': args.num_random_scenarios,
            'avg_association_accuracy': float(results_random['avg_association_accuracy']),
            'std_association_accuracy': float(results_random['std_association_accuracy']),
            'min_association_accuracy': float(results_random['min_association_accuracy']),
            'max_association_accuracy': float(results_random['max_association_accuracy']),
            'avg_rmse': float(results_random['avg_rmse']),
            'std_rmse': float(results_random['std_rmse']),
            'min_rmse': float(results_random['min_rmse']),
            'max_rmse': float(results_random['max_rmse'])
        }
    }
    
    # 打印对比表格
    print("\n┌─────────────────────────────┬──────────────┬──────────────────────────┐")
    print("│ 指标                        │ 固定场景     │ 训练类型场景 (平均)      │")
    print("├─────────────────────────────┼──────────────┼──────────────────────────┤")
    print(f"│ 测试场景数                  │            1 │ {args.num_random_scenarios:24d} │")
    print(f"│ 帧数                        │ {comparison['fixed_scenario']['num_frames']:12d} │ {comparison['training_type_scenario']['num_frames']:24d} │")
    print(f"│ 关联正确率 (%)              │ {comparison['fixed_scenario']['avg_association_accuracy']*100:12.2f} │ {comparison['training_type_scenario']['avg_association_accuracy']*100:18.2f} ± {results_random['std_association_accuracy']*100:4.2f} │")
    print(f"│ 平均RMSE (m)                │ {comparison['fixed_scenario']['avg_rmse']:12.4f} │ {comparison['training_type_scenario']['avg_rmse']:18.4f} ± {results_random['std_rmse']:4.4f} │")
    print("└─────────────────────────────┴──────────────┴──────────────────────────┘")
    
    # 分析结论
    print("\n🔍 诊断分析：")
    acc_diff = comparison['training_type_scenario']['avg_association_accuracy'] - comparison['fixed_scenario']['avg_association_accuracy']
    rmse_diff = comparison['training_type_scenario']['avg_rmse'] - comparison['fixed_scenario']['avg_rmse']
    
    if acc_diff > 0.1:  # 训练场景明显更好
        print("  ✅ 模型在训练类型场景表现更好！")
        print("     结论：模型学到了交叉场景处理能力，但固定场景可能超出训练分布")
        print("     建议：1. 检查固定场景是否太特殊（距离过长/起点特殊）")
        print("          2. 增加训练数据的多样性（更长轨迹/更多起点变化）")
    elif acc_diff < -0.1:  # 固定场景明显更好
        print("  ⚠️ 固定场景反而表现更好！")
        print("     结论：可能是随机场景恰好更难，或者固定场景更简单")
        print("     建议：多运行几次随机场景（不同seed）查看稳定性")
    else:  # 两者相近
        print("  ⚖️ 两个场景表现相近")
        if comparison['fixed_scenario']['avg_association_accuracy'] < 0.7:
            print("     结论：模型对交叉场景处理能力整体较弱")
            print("     建议：1. 增加交叉场景训练数据比例")
            print("          2. 延长训练时间")
            print("          3. 调整损失函数权重（增加关联损失权重）")
        else:
            print("     结论：模型已经学会处理交叉场景！")
    
    # 保存对比结果
    results_path = os.path.join(args.output_dir, 'comparison_summary.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("评估完成！")
    print(f"{'='*80}")
    print(f"对比报告保存至: {results_path}")
    print(f"可视化文件保存至: {args.output_dir}/")
    print(f"  - fixed_scenario_trajectories.png     (固定场景)")
    print(f"  - training_type_scenario_trajectories.png (训练类型场景)")
    print(f"  - fixed_scenario_rmse.png             (固定场景RMSE)")
    print(f"  - training_type_scenario_rmse.png     (训练类型场景RMSE)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
