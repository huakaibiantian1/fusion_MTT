"""
验证修复后的数据生成：
1. 所有真实目标都有测量
2. 轨迹不会飞太远
"""
import numpy as np
from data_generation import MTTDataGenerator

print("=" * 80)
print("验证修复后的数据生成")
print("=" * 80)

generator = MTTDataGenerator(task_type=1, seed=42)

print(f"\n生成器参数（修复后）:")
print(f"  视野范围: [{-generator.field_of_view/2}, {generator.field_of_view/2}]m （仅用于初始化）")
print(f"  速度范围: {generator.velocity_range} m/s （已降低）")
print(f"  时长: {generator.T}s")
print(f"  帧数: {generator.num_frames}")
print(f"  最大移动距离: {generator.velocity_range[1] * generator.T:.1f}m")

# 测试100个场景
num_scenarios = 100
total_targets = 0
total_frames_with_target = 0
total_frames_with_measurement = 0
total_missed = 0

max_distance_from_origin = 0
all_distances = []

print(f"\n分析 {num_scenarios} 个场景...")

for scenario_idx in range(num_scenarios):
    trajectories, measurements, associations = generator.generate_single_scenario()
    
    for traj in trajectories:
        total_targets += 1
        
        for t in range(generator.num_frames):
            if traj['birth_frame'] <= t <= traj['death_frame']:
                total_frames_with_target += 1
                
                # 计算轨迹离原点的距离
                pos = traj['states'][t, :2]
                distance = np.linalg.norm(pos)
                all_distances.append(distance)
                max_distance_from_origin = max(max_distance_from_origin, distance)
                
                # 检查是否有对应的测量
                has_measurement = traj['label'] in associations[t]
                
                if has_measurement:
                    total_frames_with_measurement += 1
                else:
                    total_missed += 1
                    if scenario_idx < 5:  # 只打印前5个场景的详细信息
                        print(f"  ⚠️ 场景{scenario_idx+1}, 帧{t}: 轨迹{traj['label']}无测量！")
                        print(f"     位置: ({pos[0]:.2f}, {pos[1]:.2f})m")

print("\n" + "=" * 80)
print("统计结果")
print("=" * 80)

print(f"\n测量覆盖率:")
print(f"  总目标帧数: {total_frames_with_target}")
print(f"  有测量的帧数: {total_frames_with_measurement}")
print(f"  缺失测量的帧数: {total_missed}")
print(f"  测量覆盖率: {total_frames_with_measurement/total_frames_with_target*100:.2f}%")

print(f"\n轨迹范围:")
print(f"  最大离原点距离: {max_distance_from_origin:.2f}m")
print(f"  平均离原点距离: {np.mean(all_distances):.2f}m")
print(f"  95%分位数: {np.percentile(all_distances, 95):.2f}m")
print(f"  99%分位数: {np.percentile(all_distances, 99):.2f}m")

print("\n" + "=" * 80)
print("详细分析：前5个场景")
print("=" * 80)

for scenario_idx in range(5):
    trajectories, measurements, associations = generator.generate_single_scenario()
    
    print(f"\n场景 {scenario_idx + 1}:")
    print(f"  目标数: {len(trajectories)}")
    
    for traj in trajectories:
        states = traj['states'][:, :2]
        x_min, x_max = states[:, 0].min(), states[:, 0].max()
        y_min, y_max = states[:, 1].min(), states[:, 1].max()
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        total_distance = np.sqrt(x_range**2 + y_range**2)
        
        # 统计测量覆盖
        frames_with_measurement = 0
        frames_alive = 0
        for t in range(generator.num_frames):
            if traj['birth_frame'] <= t <= traj['death_frame']:
                frames_alive += 1
                if traj['label'] in associations[t]:
                    frames_with_measurement += 1
        
        coverage = frames_with_measurement / frames_alive * 100 if frames_alive > 0 else 0
        
        status = "✅" if coverage == 100 else "⚠️"
        print(f"    {status} 轨迹{traj['label']}: 测量覆盖率 {coverage:.0f}%")
        print(f"       位置范围: X=[{x_min:.1f}, {x_max:.1f}]m, Y=[{y_min:.1f}, {y_max:.1f}]m")
        print(f"       移动距离: {total_distance:.1f}m")
        
        # 检查是否超出原始视野
        if abs(x_min) > 30 or abs(x_max) > 30 or abs(y_min) > 30 or abs(y_max) > 30:
            print(f"       ⚠️ 超出原始视野[-30, 30]m，但有测量")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)

if total_missed == 0:
    print(f"\n✅ 完美！所有目标的所有帧都有测量！")
    print(f"   - 测量覆盖率: 100%")
    print(f"   - 共测试了 {total_frames_with_target} 个目标帧")
else:
    print(f"\n⚠️ 仍有 {total_missed}/{total_frames_with_target} 帧缺失测量")
    print(f"   - 测量覆盖率: {total_frames_with_measurement/total_frames_with_target*100:.2f}%")

if max_distance_from_origin <= 60:
    print(f"\n✅ 轨迹范围合理！最大距离 {max_distance_from_origin:.1f}m <= 60m")
else:
    print(f"\n⚠️ 轨迹范围较大：最大距离 {max_distance_from_origin:.1f}m > 60m")
    print(f"   建议进一步降低速度或缩短时长")

print(f"\n速度降低效果:")
old_max_distance = 20 * 2.0  # 旧参数：20 m/s × 2s
new_max_distance = generator.velocity_range[1] * generator.T
print(f"  旧参数最大移动: {old_max_distance:.1f}m")
print(f"  新参数最大移动: {new_max_distance:.1f}m")
print(f"  减少比例: {(1 - new_max_distance/old_max_distance)*100:.1f}%")

print("\n" + "=" * 80)
