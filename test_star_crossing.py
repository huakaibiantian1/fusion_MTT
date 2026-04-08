"""
测试和可视化星形交叉场景生成
"""
import numpy as np
import matplotlib.pyplot as plt
from data_generation_with_crossing import MTTDataGeneratorWithCrossing

def visualize_scenario(trajectories, measurements, title="Star Crossing Scenario"):
    """可视化一个场景"""
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 绘制轨迹
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    for i, traj in enumerate(trajectories):
        states = traj['states']
        positions = states[:, :2]
        
        # 绘制轨迹线
        ax.plot(positions[:, 0], positions[:, 1], 
               color=colors[i], linewidth=2, alpha=0.7,
               label=f"Target {traj['label']}")
        
        # 标记起点和终点
        ax.scatter(positions[0, 0], positions[0, 1], 
                  color=colors[i], s=100, marker='o', 
                  edgecolors='black', linewidths=2, zorder=5)
        ax.scatter(positions[-1, 0], positions[-1, 1], 
                  color=colors[i], s=100, marker='x', 
                  linewidths=3, zorder=5)
    
    # 绘制测量点（只显示第一帧的测量）
    if len(measurements) > 0 and len(measurements[0]) > 0:
        meas = measurements[0]
        ax.scatter(meas[:, 0], meas[:, 1], 
                  color='red', s=30, marker='+', alpha=0.5,
                  label='Measurements (Frame 0)')
    
    # 设置坐标轴
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(loc='upper right', fontsize=9)
    
    # 添加视野边界
    fov = 50  # 假设视野为100m
    ax.set_xlim(-fov, fov)
    ax.set_ylim(-fov, fov)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    return fig

def test_star_crossing_generation():
    """测试星形交叉场景生成"""
    print("="*60)
    print("测试星形交叉场景生成")
    print("="*60)
    
    # 创建生成器（100%生成星形交叉用于测试）
    generator = MTTDataGeneratorWithCrossing(
        crossing_probability=1.0,  # 100%交叉场景
        star_crossing_probability=1.0,  # 100%星形交叉
        num_star_trajectories=(6, 8),  # 6-8条轨迹
        task_type=1,
        seed=42
    )
    
    # 生成多个场景并可视化
    num_scenarios = 3
    
    for i in range(num_scenarios):
        print(f"\n生成场景 {i+1}/{num_scenarios}...")
        
        trajectories, measurements, associations = generator.generate_single_scenario()
        
        print(f"  轨迹数量: {len(trajectories)}")
        print(f"  帧数: {len(measurements)}")
        
        # 分析轨迹交叉情况
        if len(trajectories) >= 2:
            # 计算所有轨迹两两之间的最小距离
            min_distances = []
            for j in range(len(trajectories)):
                for k in range(j+1, len(trajectories)):
                    traj1_pos = trajectories[j]['states'][:, :2]
                    traj2_pos = trajectories[k]['states'][:, :2]
                    
                    min_len = min(len(traj1_pos), len(traj2_pos))
                    if min_len > 0:
                        distances = np.linalg.norm(
                            traj1_pos[:min_len] - traj2_pos[:min_len], 
                            axis=1
                        )
                        min_dist = np.min(distances)
                        min_distances.append(min_dist)
            
            if min_distances:
                print(f"  轨迹间最小距离: {np.min(min_distances):.2f}m (平均: {np.mean(min_distances):.2f}m)")
        
        # 可视化
        fig = visualize_scenario(
            trajectories, 
            measurements, 
            title=f"星形交叉场景 {i+1} ({len(trajectories)}条轨迹)"
        )
        
        # 保存图片
        filename = f"star_crossing_scenario_{i+1}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  已保存: {filename}")
        plt.close()
    
    print("\n" + "="*60)
    print("✅ 测试完成！已生成可视化图片。")
    print("="*60)

def compare_crossing_types():
    """比较简单交叉和星形交叉"""
    print("\n" + "="*60)
    print("比较不同类型的交叉场景")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # 1. 简单交叉（2条轨迹）
    print("\n生成简单交叉场景...")
    generator_simple = MTTDataGeneratorWithCrossing(
        crossing_probability=1.0,
        star_crossing_probability=0.0,  # 0%星形交叉
        task_type=1,
        seed=100
    )
    trajectories1, measurements1, _ = generator_simple.generate_single_scenario()
    
    print(f"  简单交叉 - 轨迹数: {len(trajectories1)}")
    
    # 2. 星形交叉（多条轨迹）
    print("\n生成星形交叉场景...")
    generator_star = MTTDataGeneratorWithCrossing(
        crossing_probability=1.0,
        star_crossing_probability=1.0,  # 100%星形交叉
        num_star_trajectories=(6, 8),
        task_type=1,
        seed=100
    )
    trajectories2, measurements2, _ = generator_star.generate_single_scenario()
    
    print(f"  星形交叉 - 轨迹数: {len(trajectories2)}")
    
    # 绘制简单交叉
    ax1 = axes[0]
    colors1 = plt.cm.tab10(np.linspace(0, 1, len(trajectories1)))
    for i, traj in enumerate(trajectories1):
        positions = traj['states'][:, :2]
        ax1.plot(positions[:, 0], positions[:, 1], 
                color=colors1[i], linewidth=3, alpha=0.8,
                label=f"Target {traj['label']}")
        ax1.scatter(positions[0, 0], positions[0, 1], 
                   color=colors1[i], s=150, marker='o', 
                   edgecolors='black', linewidths=2, zorder=5)
        ax1.scatter(positions[-1, 0], positions[-1, 1], 
                   color=colors1[i], s=150, marker='x', 
                   linewidths=3, zorder=5)
    
    ax1.set_xlabel('X (m)', fontsize=14)
    ax1.set_ylabel('Y (m)', fontsize=14)
    ax1.set_title(f'简单交叉场景 ({len(trajectories1)}条轨迹)', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend(fontsize=11)
    ax1.set_xlim(-50, 50)
    ax1.set_ylim(-50, 50)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # 绘制星形交叉
    ax2 = axes[1]
    colors2 = plt.cm.tab10(np.linspace(0, 1, len(trajectories2)))
    for i, traj in enumerate(trajectories2):
        positions = traj['states'][:, :2]
        ax2.plot(positions[:, 0], positions[:, 1], 
                color=colors2[i], linewidth=3, alpha=0.8,
                label=f"Target {traj['label']}")
        ax2.scatter(positions[0, 0], positions[0, 1], 
                   color=colors2[i], s=150, marker='o', 
                   edgecolors='black', linewidths=2, zorder=5)
        ax2.scatter(positions[-1, 0], positions[-1, 1], 
                   color=colors2[i], s=150, marker='x', 
                   linewidths=3, zorder=5)
    
    ax2.set_xlabel('X (m)', fontsize=14)
    ax2.set_ylabel('Y (m)', fontsize=14)
    ax2.set_title(f'星形交叉场景 ({len(trajectories2)}条轨迹)', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    ax2.legend(fontsize=11)
    ax2.set_xlim(-50, 50)
    ax2.set_ylim(-50, 50)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('crossing_types_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ 已保存对比图: crossing_types_comparison.png")
    plt.close()

if __name__ == "__main__":
    # 测试星形交叉生成
    test_star_crossing_generation()
    
    # 比较不同类型的交叉
    compare_crossing_types()
    
    print("\n" + "="*60)
    print("🎉 所有测试完成！")
    print("="*60)
