"""
生成验证集 pkl 文件并保存到 checkpoint 目录。

验证集使用固定 seed=142，与训练时 create_dataloaders_multi_scenario 完全一致。
生成后可通过以下方式评估：
  python evaluate_3d.py --checkpoint checkpoints_multi/best_model.pth --scene-source val_split --num-scenarios 10
  # 或按场景类型分别评估：
  python evaluate_3d.py --checkpoint checkpoints_multi/best_model.pth --scenarios-pkl checkpoints_multi/val_scenarios_crossing.pkl --num-scenarios 10
"""

import os
import pickle
import argparse

from data_generation_multi_scenario import MTTDatasetMultiScenario, SCENARIO_TYPES


def parse_args():
    p = argparse.ArgumentParser(description='生成验证集 pkl 文件')
    p.add_argument('--save-dir',    type=str, default='checkpoints_multi',
                   help='保存目录（默认: checkpoints_multi）')
    p.add_argument('--num-val',     type=int, default=200,
                   help='验证集总场景数，将平均分配给每种场景类型（默认: 200）')
    p.add_argument('--seed',        type=int, default=142,
                   help='随机种子，必须与训练时一致（默认: 142）')
    p.add_argument('--tau',         type=int, default=4)
    p.add_argument('--max-targets', type=int, default=20)
    p.add_argument('--max-meas',    type=int, default=30)
    p.add_argument('--task-type',   type=int, default=1)
    p.add_argument('--cross-prob',  type=float, default=0.7,
                   help='交叉概率，与训练 config 保持一致（默认: 0.7）')
    return p.parse_args()


def main():
    args = parse_args()

    n_per_type = max(1, args.num_val // len(SCENARIO_TYPES))
    total      = n_per_type * len(SCENARIO_TYPES)

    print(f"{'='*60}")
    print(f"生成多场景验证集（seed={args.seed}）")
    print(f"  场景类型: {SCENARIO_TYPES}")
    print(f"  每类数量: {n_per_type}  共计: {total} 条")
    print(f"  保存目录: {args.save_dir}")
    print(f"{'='*60}\n")

    os.makedirs(args.save_dir, exist_ok=True)

    val_ds = MTTDatasetMultiScenario(
        num_scenarios_per_type=n_per_type,
        seed=args.seed,
        tau=args.tau,
        max_targets=args.max_targets,
        max_measurements=args.max_meas,
        task_type=args.task_type,
        crossing_probability=args.cross_prob,
    )

    print()
    for stype, scenarios in val_ds.scenarios_by_type.items():
        pkl_path = os.path.join(args.save_dir, f'val_scenarios_{stype}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(scenarios, f)
        print(f"  [已保存] val_scenarios_{stype}.pkl  ({len(scenarios)} 条)  →  {pkl_path}")

    print(f"\n完成！共保存 {len(SCENARIO_TYPES)} 个验证集文件。")
    print("\n使用方法示例:")
    print(f"  # 自动加载所有类型（val_split 模式）")
    print(f"  python evaluate_3d.py --checkpoint {args.save_dir}/best_model.pth --scene-source val_split --num-scenarios 10")
    print(f"  # 按类型分别评估")
    for stype in SCENARIO_TYPES:
        print(f"  python evaluate_3d.py --checkpoint {args.save_dir}/best_model.pth "
              f"--scenarios-pkl {args.save_dir}/val_scenarios_{stype}.pkl "
              f"--output-dir eval_val_{stype} --num-scenarios 10")


if __name__ == '__main__':
    main()
