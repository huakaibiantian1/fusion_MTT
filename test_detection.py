"""
测试检测概率是否真的是1.0
"""
import numpy as np
from data_generation import MTTDataGenerator

print("=" * 80)
print("测试检测概率")
print("=" * 80)

# 生成10个场景，统计检测情况
generator = MTTDataGenerator(task_type=1, seed=42)

print(f"\n生成器参数:")
print(f"  P_d = {generator.P_d}")
print(f"  lambda_0 = {generator.lambda_0}")
print(f"  场景范围 = [-{generator.field_of_view/2}, {generator.field_of_view/2}]m")

total_frames = 0
total_possible_detections = 0
total_actual_detections = 0

for i in range(10):
    trajectories, measurements, associations = generator.generate_single_scenario()
    num_frames = len(measurements)
    num_targets = len(trajectories)
    
    # 统计每帧的检测情况
    for t in range(num_frames):
        # 这一帧应该有多少个目标存在
        alive_targets = 0
        for traj in trajectories:
            if traj['birth_frame'] <= t <= traj['death_frame']:
                alive_targets += 1
        
        # 这一帧实际有多少个真实测量（非杂波）
        actual_detections = np.sum(associations[t] > 0)
        
        total_possible_detections += alive_targets
        total_actual_detections += actual_detections
        
        if actual_detections < alive_targets:
            print(f"\n场景{i+1}, 帧{t}: 应有{alive_targets}个检测，实际{actual_detections}个")
            print(f"  轨迹信息:")
            for traj in trajectories:
                if traj['birth_frame'] <= t <= traj['death_frame']:
                    # 检查这个轨迹是否被检测到
                    detected = traj['label'] in associations[t]
                    print(f"    轨迹{traj['label']}: {'✓检测到' if detected else '✗漏检'}")
    
    total_frames += num_frames

print(f"\n" + "=" * 80)
print("统计结果:")
print("=" * 80)
print(f"总帧数: {total_frames}")
print(f"应有检测数: {total_possible_detections}")
print(f"实际检测数: {total_actual_detections}")
print(f"检测率: {total_actual_detections / total_possible_detections * 100:.2f}%")

if total_actual_detections == total_possible_detections:
    print("\n✅ P_d=1.0 工作正常！所有目标都被检测到")
else:
    print(f"\n❌ 有 {total_possible_detections - total_actual_detections} 次漏检！")
    print(f"   理论上P_d=1.0应该没有漏检")
    print(f"   实际检测率: {total_actual_detections / total_possible_detections:.3f}")

print("=" * 80)
