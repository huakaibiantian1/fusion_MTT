"""
测试视野外(FOV)问题：
1. 训练数据中有多少目标飞出视野？
2. 飞出视野后的影响
"""
import numpy as np
from data_generation import MTTDataGenerator

print("=" * 80)
print("测试视野范围(FOV)问题")
print("=" * 80)

generator = MTTDataGenerator(task_type=1, seed=42)

print(f"\n生成器参数:")
print(f"  视野范围: [{-generator.field_of_view/2}, {generator.field_of_view/2}]m")
print(f"  速度范围: {generator.velocity_range} m/s")
print(f"  时长: {generator.T}s")
print(f"  帧数: {generator.num_frames}")
print(f"  dt: {generator.dt}s")

# 生成100个场景，统计视野外情况
num_scenarios = 100
total_trajectories = 0
trajectories_leaving_fov = 0
total_frames_outside_fov = 0
total_frames = 0

fov_statistics = {
    'total_targets': 0,
    'targets_always_inside': 0,
    'targets_sometimes_outside': 0,
    'targets_always_outside': 0,
    'missed_detections_due_to_fov': 0,
    'total_possible_detections': 0
}

print(f"\n分析 {num_scenarios} 个场景...")

for scenario_idx in range(num_scenarios):
    trajectories, measurements, associations = generator.generate_single_scenario()
    
    for traj in trajectories:
        total_trajectories += 1
        fov_statistics['total_targets'] += 1
        
        frames_inside = 0
        frames_outside = 0
        
        for t in range(generator.num_frames):
            if traj['birth_frame'] <= t <= traj['death_frame']:
                total_frames += 1
                pos = traj['states'][t, :2]
                
                # 检查是否在视野内
                in_fov = generator._in_field_of_view(pos)
                
                if in_fov:
                    frames_inside += 1
                    fov_statistics['total_possible_detections'] += 1
                    
                    # 检查是否有对应的测量
                    has_measurement = traj['label'] in associations[t]
                    if not has_measurement:
                        # 理论上P_d=1应该有测量，但没有
                        # （不应该发生，因为在视野内且P_d=1）
                        pass
                else:
                    frames_outside += 1
                    total_frames_outside_fov += 1
                    fov_statistics['missed_detections_due_to_fov'] += 1
        
        # 统计轨迹类型
        if frames_outside == 0:
            fov_statistics['targets_always_inside'] += 1
        elif frames_inside == 0:
            fov_statistics['targets_always_outside'] += 1
        else:
            fov_statistics['targets_sometimes_outside'] += 1
            trajectories_leaving_fov += 1

print("\n" + "=" * 80)
print("统计结果")
print("=" * 80)

print(f"\n总轨迹数: {total_trajectories}")
print(f"总帧数（目标存活）: {total_frames}")
print(f"视野外帧数: {total_frames_outside_fov}")
print(f"视野外比例: {total_frames_outside_fov/total_frames*100:.2f}%")

print(f"\n轨迹分类:")
print(f"  始终在视野内: {fov_statistics['targets_always_inside']} ({fov_statistics['targets_always_inside']/total_trajectories*100:.1f}%)")
print(f"  部分飞出视野: {fov_statistics['targets_sometimes_outside']} ({fov_statistics['targets_sometimes_outside']/total_trajectories*100:.1f}%)")
print(f"  始终在视野外: {fov_statistics['targets_always_outside']} ({fov_statistics['targets_always_outside']/total_trajectories*100:.1f}%)")

print(f"\n检测情况:")
print(f"  理论可检测数（视野内+P_d=1）: {fov_statistics['total_possible_detections']}")
print(f"  因视野限制漏检: {fov_statistics['missed_detections_due_to_fov']}")
print(f"  漏检率: {fov_statistics['missed_detections_due_to_fov']/(fov_statistics['total_possible_detections']+fov_statistics['missed_detections_due_to_fov'])*100:.2f}%")

print("\n" + "=" * 80)
print("详细分析：前5个场景")
print("=" * 80)

for scenario_idx in range(5):
    trajectories, measurements, associations = generator.generate_single_scenario()
    
    print(f"\n场景 {scenario_idx + 1}:")
    print(f"  目标数: {len(trajectories)}")
    
    for traj in trajectories:
        # 找到轨迹的位置范围
        states = traj['states'][:, :2]  # [T, 2]
        x_min, x_max = states[:, 0].min(), states[:, 0].max()
        y_min, y_max = states[:, 1].min(), states[:, 1].max()
        
        # 检查是否超出视野
        fov_half = generator.field_of_view / 2
        outside_x = x_min < -fov_half or x_max > fov_half
        outside_y = y_min < -fov_half or y_max > fov_half
        
        if outside_x or outside_y:
            print(f"    ⚠️ 轨迹{traj['label']}: 超出视野")
            print(f"       X范围: [{x_min:.1f}, {x_max:.1f}]m (视野: [-30, 30])")
            print(f"       Y范围: [{y_min:.1f}, {y_max:.1f}]m (视野: [-30, 30])")
            
            # 统计有多少帧在视野外
            frames_outside = 0
            for t in range(generator.num_frames):
                if traj['birth_frame'] <= t <= traj['death_frame']:
                    pos = states[t]
                    if not generator._in_field_of_view(pos):
                        frames_outside += 1
            print(f"       视野外帧数: {frames_outside}/{generator.num_frames}")
        else:
            print(f"    ✅ 轨迹{traj['label']}: 始终在视野内")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)

if total_frames_outside_fov > 0:
    print(f"\n⚠️  训练数据中确实存在目标飞出视野的情况！")
    print(f"    - {total_frames_outside_fov}帧（{total_frames_outside_fov/total_frames*100:.1f}%）的目标在视野外")
    print(f"    - {trajectories_leaving_fov}条轨迹部分时间在视野外")
    print(f"\n这意味着：")
    print(f"    1. 模型需要学习处理\"无测量更新\"的情况")
    print(f"    2. 这是合理的设计，符合真实传感器的限制")
    print(f"    3. 但增加了跟踪难度：轨迹需要\"预测\"而不是\"更新\"")
else:
    print(f"\n✅ 所有目标始终在视野内")

print("\n" + "=" * 80)
