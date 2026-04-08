"""
星形交叉场景对比评估脚本
评估不同轨迹数量的星形交叉场景，生成对比分析
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from bait_model import BAIT
from evaluate_cross_star import evaluate_star_scenario

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    parser = argparse.ArgumentParser(description='Compare BAIT model performance on different star crossing scenarios')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--trajectory-counts', type=int, nargs='+', default=[4, 6, 8],
                        help='List of trajectory counts to test (e.g., 4 6 8)')
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
    parser.add_argument('--output-dir', type=str, default='evaluation_results_star_comparison',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for scenario generation')
    
    return parser.parse_args()


def plot_comparison(all_results, output_dir):
    """绘制对比图表"""
    
    # 提取数据
    traj_counts = [r['num_trajectories'] for r in all_results]
    accuracies = [r['avg_association_accuracy'] * 100 for r in all_results]
    rmses = [r['avg_rmse'] for r in all_results]
    
    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 图1: 数据关联正确率
    ax1 = axes[0]
    bars1 = ax1.bar(range(len(traj_counts)), accuracies, 
                    color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'][:len(traj_counts)],
                    alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('轨迹数量', fontsize=14, fontweight='bold')
    ax1.set_ylabel('数据关联正确率 (%)', fontsize=14, fontweight='bold')
    ax1.set_title('不同轨迹数量的数据关联性能', fontsize=16, fontweight='bold')
    ax1.set_xticks(range(len(traj_counts)))
    ax1.set_xticklabels([f'{c}条' for c in traj_counts], fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 105])
    
    # 在柱状图上标注数值
    for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 图2: RMSE
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(traj_counts)), rmses,
                    color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'][:len(traj_counts)],
                    alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('轨迹数量', fontsize=14, fontweight='bold')
    ax2.set_ylabel('平均RMSE (m)', fontsize=14, fontweight='bold')
    ax2.set_title('不同轨迹数量的位置预测误差', fontsize=16, fontweight='bold')
    ax2.set_xticks(range(len(traj_counts)))
    ax2.set_xticklabels([f'{c}条' for c in traj_counts], fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上标注数值
    for i, (bar, rmse) in enumerate(zip(bars2, rmses)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{rmse:.3f}m',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'star_crossing_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存对比图: {output_path}")
    
    # 创建详细的对比表格图
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据
    table_data = [['轨迹数量', '数据关联正确率', '平均RMSE', '复杂度评级']]
    complexity_levels = ['简单', '中等', '困难', '很困难', '极困难']
    
    for i, r in enumerate(all_results):
        complexity_idx = min(i, len(complexity_levels) - 1)
        table_data.append([
            f'{r["num_trajectories"]}条',
            f'{r["avg_association_accuracy"]*100:.2f}%',
            f'{r["avg_rmse"]:.4f}m',
            complexity_levels[complexity_idx]
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.2, 0.3, 0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置数据行样式
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    plt.title('星形交叉场景性能对比表', fontsize=16, fontweight='bold', pad=20)
    
    output_path = os.path.join(output_dir, 'star_crossing_comparison_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存对比表格: {output_path}")


def generate_comparison_report(all_results, output_dir):
    """生成对比报告"""
    report_path = os.path.join(output_dir, 'comparison_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 星形交叉场景性能对比报告\n\n")
        f.write("## 测试概述\n\n")
        f.write(f"- 测试场景数: {len(all_results)}\n")
        f.write(f"- 轨迹数量范围: {min(r['num_trajectories'] for r in all_results)}-{max(r['num_trajectories'] for r in all_results)}条\n")
        f.write(f"- 场景类型: 星形交叉（多条轨迹从中心辐射）\n\n")
        
        f.write("## 详细结果\n\n")
        f.write("| 轨迹数量 | 数据关联正确率 | 平均RMSE | 性能评价 |\n")
        f.write("|---------|--------------|---------|----------|\n")
        
        for r in all_results:
            acc = r['avg_association_accuracy'] * 100
            rmse = r['avg_rmse']
            
            # 性能评价
            if acc >= 90 and rmse < 1.0:
                rating = "优秀 ⭐⭐⭐⭐⭐"
            elif acc >= 80 and rmse < 2.0:
                rating = "良好 ⭐⭐⭐⭐"
            elif acc >= 70 and rmse < 3.0:
                rating = "中等 ⭐⭐⭐"
            elif acc >= 60:
                rating = "及格 ⭐⭐"
            else:
                rating = "需改进 ⭐"
            
            f.write(f"| {r['num_trajectories']}条 | {acc:.2f}% | {rmse:.4f}m | {rating} |\n")
        
        f.write("\n## 性能趋势分析\n\n")
        
        # 计算趋势
        accuracies = [r['avg_association_accuracy'] for r in all_results]
        rmses = [r['avg_rmse'] for r in all_results]
        
        acc_trend = accuracies[-1] - accuracies[0]
        rmse_trend = rmses[-1] - rmses[0]
        
        f.write("### 数据关联正确率\n")
        if acc_trend > 0:
            f.write(f"- ✅ 随轨迹数量增加，正确率**提升**了 {acc_trend*100:.2f}%\n")
        elif acc_trend < 0:
            f.write(f"- ⚠️ 随轨迹数量增加，正确率**下降**了 {abs(acc_trend)*100:.2f}%\n")
        else:
            f.write(f"- ➡️ 正确率保持**稳定**\n")
        
        f.write(f"- 最高正确率: {max(accuracies)*100:.2f}% ({all_results[accuracies.index(max(accuracies))]['num_trajectories']}条轨迹)\n")
        f.write(f"- 最低正确率: {min(accuracies)*100:.2f}% ({all_results[accuracies.index(min(accuracies))]['num_trajectories']}条轨迹)\n\n")
        
        f.write("### 位置预测误差(RMSE)\n")
        if rmse_trend > 0:
            f.write(f"- ⚠️ 随轨迹数量增加，RMSE**增加**了 {rmse_trend:.4f}m\n")
        elif rmse_trend < 0:
            f.write(f"- ✅ 随轨迹数量增加，RMSE**减少**了 {abs(rmse_trend):.4f}m\n")
        else:
            f.write(f"- ➡️ RMSE保持**稳定**\n")
        
        f.write(f"- 最低RMSE: {min(rmses):.4f}m ({all_results[rmses.index(min(rmses))]['num_trajectories']}条轨迹)\n")
        f.write(f"- 最高RMSE: {max(rmses):.4f}m ({all_results[rmses.index(max(rmses))]['num_trajectories']}条轨迹)\n\n")
        
        f.write("## 结论与建议\n\n")
        
        avg_acc = np.mean(accuracies) * 100
        avg_rmse = np.mean(rmses)
        
        f.write(f"1. **整体性能**: 平均数据关联正确率 {avg_acc:.2f}%，平均RMSE {avg_rmse:.4f}m\n\n")
        
        if avg_acc >= 80:
            f.write("2. **模型表现**: 模型在星形交叉场景中表现**良好**，能够有效处理多目标交叉情况\n\n")
        elif avg_acc >= 60:
            f.write("2. **模型表现**: 模型在星形交叉场景中表现**中等**，建议继续训练以提升性能\n\n")
        else:
            f.write("2. **模型表现**: 模型在星形交叉场景中表现**欠佳**，建议增加星形交叉训练数据\n\n")
        
        # 找出表现最差的场景
        worst_idx = accuracies.index(min(accuracies))
        f.write(f"3. **改进方向**: {all_results[worst_idx]['num_trajectories']}条轨迹场景性能最低，")
        f.write(f"建议增加该复杂度的训练样本\n\n")
        
        f.write("4. **应用建议**: \n")
        for r in all_results:
            if r['avg_association_accuracy'] >= 0.8:
                f.write(f"   - ✅ 可用于{r['num_trajectories']}个目标同时交叉的场景\n")
            else:
                f.write(f"   - ⚠️ {r['num_trajectories']}个目标同时交叉场景建议谨慎使用\n")
    
    print(f"  保存对比报告: {report_path}")


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
    
    print("\n" + "="*60)
    print("星形交叉场景对比评估")
    print("="*60)
    print(f"将测试以下轨迹数量: {args.trajectory_counts}")
    print(f"场景类型: {args.scenario_type} (使用训练数据生成器)")
    print(f"帧数: {args.num_frames}")
    print(f"随机种子: {args.seed} (可复现)")
    print("="*60 + "\n")
    
    # 评估所有场景
    all_results = []
    
    for num_traj in args.trajectory_counts:
        print(f"\n{'='*60}")
        print(f"评估 {num_traj} 条轨迹的星形交叉场景")
        print(f"{'='*60}")
        
        results = evaluate_star_scenario(
            model, num_traj, args.tau, args.max_targets,
            args.max_measurements, device, args.output_dir,
            num_frames=args.num_frames, scenario_type=args.scenario_type, seed=args.seed
        )
        
        all_results.append(results)
    
    # 生成对比图表
    print(f"\n{'='*60}")
    print("生成对比分析...")
    print(f"{'='*60}")
    plot_comparison(all_results, args.output_dir)
    
    # 生成对比报告
    generate_comparison_report(all_results, args.output_dir)
    
    # 保存汇总结果
    summary = {
        'trajectory_counts': args.trajectory_counts,
        'results': [
            {
                'num_trajectories': r['num_trajectories'],
                'avg_association_accuracy': float(r['avg_association_accuracy']),
                'avg_rmse': float(r['avg_rmse'])
            }
            for r in all_results
        ]
    }
    
    summary_path = os.path.join(args.output_dir, 'comparison_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("对比评估完成！")
    print(f"{'='*60}")
    print(f"汇总结果: {summary_path}")
    print(f"对比报告: {os.path.join(args.output_dir, 'comparison_report.md')}")
    print(f"可视化: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
