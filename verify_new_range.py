"""
验证新的[-50, 50]m范围设置
"""
import numpy as np
from data_generation import MTTDataGenerator

print("=" * 80)
print("验证新的数据范围设置")
print("=" * 80)

generator = MTTDataGenerator(task_type=1, seed=42)

print(f"\n✅ 生成器参数:")
print(f"  视野范围: [{-generator.field_of_view/2:.1f}, {generator.field_of_view/2:.1f}]m")
print(f"  速度范围: {generator.velocity_range} m/s")
print(f"  检测概率: {generator.P_d}")
print(f"  最大理论移动: {generator.velocity_range[1] * generator.T:.1f}m")

# 测试100个场景
num_scenarios = 100
stats = {
    'total_measurements': 0,
    'measurements_at_boundary': 0,
    'targets_with_full_coverage': 0,
    'total_targets': 0,
    'measurement_ranges': {'x_min': float('inf'), 'x_max': float('-inf'),
                          'y_min': float('inf'), 'y_max': float('-inf')},
    'trajectory_ranges': {'x_min': float('inf'), 'x_max': float('-inf'),
                         'y_min': float('inf'), 'y_max': float('-inf')}
}

print(f"\n分析 {num_scenarios} 个场景...")

for scenario_idx in range(num_scenarios):
    trajectories, measurements, associations = generator.generate_single_scenario()
    
    for traj in trajectories:
        stats['total_targets'] += 1
        frames_with_meas = 0
        frames_alive = 0
        
        # 统计轨迹范围
        traj_states = traj['states'][:, :2]
        stats['trajectory_ranges']['x_min'] = min(stats['trajectory_ranges']['x_min'], traj_states[:, 0].min())
        stats['trajectory_ranges']['x_max'] = max(stats['trajectory_ranges']['x_max'], traj_states[:, 0].max())
        stats['trajectory_ranges']['y_min'] = min(stats['trajectory_ranges']['y_min'], traj_states[:, 1].min())
        stats['trajectory_ranges']['y_max'] = max(stats['trajectory_ranges']['y_max'], traj_states[:, 1].max())
        
        for t in range(generator.num_frames):
            if traj['birth_frame'] <= t <= traj['death_frame']:
                frames_alive += 1
                if traj['label'] in associations[t]:
                    frames_with_meas += 1
        
        if frames_with_meas == frames_alive:
            stats['targets_with_full_coverage'] += 1
    
    # 统计测量范围
    for frame_meas in measurements:
        stats['total_measurements'] += len(frame_meas)
        if len(frame_meas) > 0:
            stats['measurement_ranges']['x_min'] = min(stats['measurement_ranges']['x_min'], frame_meas[:, 0].min())
            stats['measurement_ranges']['x_max'] = max(stats['measurement_ranges']['x_max'], frame_meas[:, 0].max())
            stats['measurement_ranges']['y_min'] = min(stats['measurement_ranges']['y_min'], frame_meas[:, 1].min())
            stats['measurement_ranges']['y_max'] = max(stats['measurement_ranges']['y_max'], frame_meas[:, 1].max())
            
            # 检查是否有测量在边界
            boundary_threshold = generator.field_of_view / 2 - 0.01
            at_boundary = np.any(np.abs(frame_meas) >= boundary_threshold)
            if at_boundary:
                stats['measurements_at_boundary'] += len(frame_meas[np.any(np.abs(frame_meas) >= boundary_threshold, axis=1)])

print("\n" + "=" * 80)
print("统计结果")
print("=" * 80)

print(f"\n📊 目标覆盖率:")
print(f"  总目标数: {stats['total_targets']}")
print(f"  100%测量覆盖: {stats['targets_with_full_coverage']} ({stats['targets_with_full_coverage']/stats['total_targets']*100:.1f}%)")

print(f"\n📏 测量范围:")
print(f"  X: [{stats['measurement_ranges']['x_min']:.2f}, {stats['measurement_ranges']['x_max']:.2f}]m")
print(f"  Y: [{stats['measurement_ranges']['y_min']:.2f}, {stats['measurement_ranges']['y_max']:.2f}]m")
print(f"  期望范围: [-50, 50]m")

x_in_range = -50 <= stats['measurement_ranges']['x_min'] and stats['measurement_ranges']['x_max'] <= 50
y_in_range = -50 <= stats['measurement_ranges']['y_min'] and stats['measurement_ranges']['y_max'] <= 50

if x_in_range and y_in_range:
    print(f"  ✅ 所有测量都在 [-50, 50]m 范围内！")
else:
    print(f"  ⚠️ 有测量超出范围！")

print(f"\n🎯 轨迹范围（未裁剪）:")
print(f"  X: [{stats['trajectory_ranges']['x_min']:.2f}, {stats['trajectory_ranges']['x_max']:.2f}]m")
print(f"  Y: [{stats['trajectory_ranges']['y_min']:.2f}, {stats['trajectory_ranges']['y_max']:.2f}]m")

print(f"\n🔧 边界裁剪情况:")
print(f"  在边界附近的测量: {stats['measurements_at_boundary']}/{stats['total_measurements']}")
print(f"  裁剪比例: {stats['measurements_at_boundary']/stats['total_measurements']*100:.3f}%")

print("\n" + "=" * 80)
print("归一化测试")
print("=" * 80)

COORD_SCALE = 50.0
print(f"\n归一化因子: {COORD_SCALE}")

# 测试边界值
test_values = [
    ([-50, -50], "左下角"),
    ([50, 50], "右上角"),
    ([0, 0], "中心"),
    ([25, 25], "1/4处"),
    ([-25, 25], "对角")
]

print(f"\n测试边界值归一化:")
for pos, desc in test_values:
    normalized = np.array(pos) / COORD_SCALE
    recovered = normalized * COORD_SCALE
    print(f"  {desc:8s}: {pos} -> {normalized} -> {recovered}")

print("\n" + "=" * 80)
print("详细场景分析（前3个）")
print("=" * 80)

for scenario_idx in range(3):
    trajectories, measurements, associations = generator.generate_single_scenario()
    
    print(f"\n场景 {scenario_idx + 1}:")
    print(f"  目标数: {len(trajectories)}")
    
    all_have_meas = True
    for traj in trajectories:
        frames_with_meas = sum(1 for t in range(generator.num_frames) 
                              if traj['birth_frame'] <= t <= traj['death_frame'] 
                              and traj['label'] in associations[t])
        frames_alive = sum(1 for t in range(generator.num_frames) 
                          if traj['birth_frame'] <= t <= traj['death_frame'])
        
        if frames_with_meas < frames_alive:
            all_have_meas = False
            print(f"    ⚠️ 轨迹{traj['label']}: {frames_with_meas}/{frames_alive}帧有测量")
        
        # 显示轨迹范围
        states = traj['states'][:, :2]
        x_range = [states[:, 0].min(), states[:, 0].max()]
        y_range = [states[:, 1].min(), states[:, 1].max()]
        print(f"    轨迹{traj['label']}: X=[{x_range[0]:6.1f}, {x_range[1]:6.1f}]m, Y=[{y_range[0]:6.1f}, {y_range[1]:6.1f}]m")
    
    if all_have_meas:
        print(f"  ✅ 所有轨迹100%测量覆盖")
    
    # 检查测量范围
    all_meas = np.vstack(measurements) if len(measurements) > 0 else np.empty((0, 2))
    if len(all_meas) > 0:
        meas_x_range = [all_meas[:, 0].min(), all_meas[:, 0].max()]
        meas_y_range = [all_meas[:, 1].min(), all_meas[:, 1].max()]
        print(f"  测量范围: X=[{meas_x_range[0]:6.1f}, {meas_x_range[1]:6.1f}]m, Y=[{meas_y_range[0]:6.1f}, {meas_y_range[1]:6.1f}]m")
        
        if np.any(np.abs(all_meas) > 50.01):
            print(f"  ⚠️ 有测量超出[-50, 50]范围！")

print("\n" + "=" * 80)
print("最终验证")
print("=" * 80)

checks = [
    (stats['targets_with_full_coverage'] == stats['total_targets'], 
     "100%测量覆盖率"),
    (x_in_range and y_in_range, 
     "所有测量在[-50, 50]m范围内"),
    (generator.field_of_view == 100.0, 
     "视野范围正确设置为100.0m"),
    (generator.velocity_range == (3, 10), 
     "速度范围正确设置为(3, 10) m/s"),
    (generator.P_d == 1.0, 
     "检测概率为1.0"),
]

all_pass = True
for passed, desc in checks:
    status = "✅" if passed else "❌"
    print(f"{status} {desc}")
    if not passed:
        all_pass = False

if all_pass:
    print(f"\n" + "=" * 80)
    print("🎉 所有检查通过！可以开始重新训练了！")
    print("=" * 80)
    print(f"\n建议命令:")
    print(f"  1. 清理旧检查点: rm -rf checkpoints/* logs/*")
    print(f"  2. 开始训练: python train.py --config config.json --device cuda")
    print(f"  3. 监控训练: tensorboard --logdir logs")
else:
    print(f"\n⚠️ 有检查失败，请检查代码修改！")

print("=" * 80)
