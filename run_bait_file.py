import argparse
import copy
import os

import numpy as np
import torch

from bait_data_io import load_scenario_file
from bait_model import BAIT
from evaluate_3d import (
    COORD_SCALE,
    evaluate_scenario_kf_managed,
    plot_3d_trajectories,
    plot_association_accuracy,
    plot_error_curves,
    plot_ospa_curve,
)


MODEL_KWARGS = dict(
    d_model=256,
    nhead=8,
    num_encoder_layers=6,
    num_associate_decoder_layers=3,
    num_filtering_decoder_layers=6,
    dim_feedforward_encoder=2048,
    dim_feedforward_associate=1024,
    dim_feedforward_filtering=2048,
    max_targets=20,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run BAIT(KF-managed) from scenario file")
    parser.add_argument("--checkpoint", default=os.path.join("checkpoints_multi", "best_model.pth"))
    parser.add_argument("--input", required=True, help="pkl/npz/json scenario file")
    parser.add_argument("--output-dir", default="eval_gui")
    parser.add_argument("--gate-m", type=float, default=600.0)
    parser.add_argument("--sigma-r-m", type=float, default=80.0)
    parser.add_argument("--sigma-q-m", type=float, default=50.0)
    parser.add_argument("--confirm-hits", type=int, default=4)
    parser.add_argument("--max-missed", type=int, default=3)
    parser.add_argument("--association-gate-m", type=float, default=300.0)
    return parser.parse_args()


def main():
    args = parse_args()
    scenario = load_scenario_file(args.input, association_gate_m=args.association_gate_m)
    print(f"输入文件: {args.input}")
    print(f"目标数: {len(scenario[0])}, 帧数: {len(scenario[1])}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = BAIT(**MODEL_KWARGS).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"模型加载成功（训练步骤 {ckpt.get('step', '?')}）")

    max_measurements = max(30, max((len(m) for m in scenario[1]), default=0))
    result = evaluate_scenario_kf_managed(
        model,
        copy.deepcopy(scenario),
        tau=4,
        max_targets=20,
        max_measurements=max_measurements,
        device=device,
        scenario_idx=0,
        kf_params=dict(
            gate_m=args.gate_m,
            sigma_r_m=args.sigma_r_m,
            sigma_q_m=args.sigma_q_m,
            confirm_hits=args.confirm_hits,
            max_missed=args.max_missed,
        ),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    plot_3d_trajectories(result, 0, args.output_dir)
    plot_error_curves(result, 0, args.output_dir)
    avg_ospa = plot_ospa_curve(result, 0, args.output_dir)
    plot_association_accuracy(result, 0, args.output_dir)

    avg_acc = float(np.mean(result["frame_assoc_acc"])) if result["frame_assoc_acc"] else 0.0
    all_dist = []
    for pred, true in zip(result["frame_pred_xyz"], result["frame_true_xyz"]):
        n = min(len(pred), len(true))
        if n <= 0:
            continue
        valid = ~np.any(np.isnan(pred[:n]), axis=1)
        if np.any(valid):
            all_dist.extend(np.linalg.norm((pred[:n][valid] - true[:n][valid]) * COORD_SCALE, axis=1).tolist())
    avg_dist = float(np.mean(all_dist)) if all_dist else 0.0
    print(f"\n[BAIT] 关联正确率={avg_acc*100:.1f}% OSPA={avg_ospa:.2f}m 命中3D误差={avg_dist:.2f}m")


if __name__ == "__main__":
    main()

