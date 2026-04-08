"""
3D BAIT 模型评估脚本
输出：
  1. 三维轨迹对比图（真实 vs 预测 + 测量点）
  2. 每轴误差曲线图（X / Y / Z）+ 3D 欧氏距离误差图
  3. OSPA 误差曲线
  4. 数据关联正确率：逐帧打印 + 汇总图
  5. 评估汇总 JSON

场景来源（--scene-source）：
  new        : 每次随机生成新场景（可用 --seed / --crossing-prob 控制）
  test_split : 使用与训练相同的测试集（固定 seed=242，与 create_dataloaders_with_crossing 一致）
"""

import os
import argparse
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
import torch

from bait_model import BAIT
from data_generation_with_crossing import (
    MTTDataGeneratorWithCrossing,
    MTTDatasetWithCrossing,
)
from metrics import TrackingMetrics

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

COORD_SCALE = 50000.0   # 归一化因子：R_max=50000m，与训练保持一致


# ============================================================
# 命令行参数
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate 3D BAIT model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
场景来源说明（--scene-source）:
  new         每次重新随机生成，可配合 --seed / --crossing-prob 使用
  test_split  使用与训练完全一致的测试集（seed=242，与 create_dataloaders_with_crossing 相同）

示例:
  # 新场景（默认）
  python evaluate_3d.py --checkpoint checkpoints_star_crossing/best_model.pth --scene-source new --num-scenarios 5

  # 训练时的测试集
  python evaluate_3d.py --checkpoint checkpoints_star_crossing/best_model.pth --scene-source test_split --num-scenarios 10
"""
    )
    p.add_argument('--checkpoint',        type=str,   required=True,
                   help='模型检查点路径 (.pth)')
    p.add_argument('--num-scenarios',     type=int,   default=5,
                   help='评估场景数量')
    p.add_argument('--tau',               type=int,   default=4)
    p.add_argument('--max-targets',       type=int,   default=20)
    p.add_argument('--max-measurements',  type=int,   default=30)
    p.add_argument('--task-type',         type=int,   default=1, choices=[1, 2])
    p.add_argument('--device',            type=str,   default='cuda')
    p.add_argument('--output-dir',        type=str,   default='evaluation_results_3d')

    # ── 场景来源控制 ──────────────────────────────────────────────
    p.add_argument(
        '--scene-source',
        type=str,
        default='new',
        choices=['new', 'test_split'],
        help=(
            'new        = 随机生成新场景（可用 --seed / --crossing-prob 控制）\n'
            'test_split = 使用训练时的测试集（固定 seed=242）'
        )
    )
    # 仅在 --scene-source new 时生效
    p.add_argument('--seed',              type=int,   default=9999,
                   help='随机种子（仅 --scene-source new 时有效）')
    p.add_argument('--crossing-prob',     type=float, default=0.8,
                   help='交叉场景概率（仅 --scene-source new 时有效）')
    p.add_argument('--star-crossing-prob', type=float, default=0.7,
                   help='星形交叉概率（仅 --scene-source new 时有效）')
    # 仅在 --scene-source test_split 时生效
    p.add_argument('--test-seed',         type=int,   default=242,
                   help='测试集 seed（仅 --scene-source test_split 时有效，默认与训练一致=242）')
    p.add_argument('--test-crossing-prob', type=float, default=0.5,
                   help='测试集交叉概率（仅 --scene-source test_split 时有效，与训练 config 保持一致）')
    p.add_argument('--test-star-prob',     type=float, default=0.7,
                   help='测试集星形交叉概率（仅 --scene-source test_split 时有效）')

    return p.parse_args()


# ============================================================
# 辅助：准备过去 tau 帧的状态（3D，5维）
# ============================================================

def prepare_past_states(tracked_states, trajectories, frame_idx, tau, max_targets, dt, T_total=30.0):
    """
    Returns: np.ndarray [tau * max_targets, 5]  = [label, x_norm, y_norm, z_norm, t_norm]
    时间戳归一化到 [0, 1]，与训练时保持一致
    """
    past = []
    for t in range(frame_idx - tau, frame_idx):
        frame_past = []
        for idx, traj in enumerate(trajectories):
            if idx >= max_targets:
                break
            label = traj['label']
            if traj['birth_frame'] <= t <= traj['death_frame'] and t < len(tracked_states[label]):
                xyz = tracked_states[label][t]
                if xyz is not None and not np.all(xyz == 0):
                    t_norm = (t * dt) / T_total   # 归一化到[0,1]
                    state  = np.array([label, xyz[0], xyz[1], xyz[2], t_norm])
                else:
                    state = np.zeros(5)
            else:
                state = np.zeros(5)
            frame_past.append(state)

        while len(frame_past) < max_targets:
            frame_past.append(np.zeros(5))
        past.extend(frame_past[:max_targets])

    return np.array(past)   # [tau * max_targets, 5]


def prepare_measurements(meas_frame, max_measurements):
    """
    Returns: np.ndarray [max_measurements, 3]（已归一化）
    meas_frame 已经是归一化后的 xyz 坐标
    """
    meas = meas_frame.copy() if len(meas_frame) > 0 else np.empty((0, 3))
    n = len(meas)
    if n < max_measurements:
        pad = np.zeros((max_measurements - n, 3))
        meas = np.vstack([meas, pad]) if n > 0 else pad
    else:
        meas = meas[:max_measurements]
    return meas


# ============================================================
# 核心评估逻辑（单个场景）
# ============================================================

def evaluate_scenario(model, scenario, tau, max_targets, max_measurements, device, scenario_idx):
    """
    评估单个3D场景，返回所有逐帧指标。

    Args:
        scenario: (trajectories, measurements, gt_associations) 元组
                  由调用方负责提供（new 模式：实时生成；test_split 模式：来自数据集）
    """
    model.eval()
    print(f"\n{'='*60}")
    print(f"场景 {scenario_idx + 1}")
    print(f"{'='*60}")

    trajectories, measurements, gt_associations = scenario
    num_frames = len(measurements)

    # —— 归一化 ——
    for traj in trajectories:
        traj['states'][:, :3] /= COORD_SCALE   # xyz 归一化
    for fm in measurements:
        if len(fm) > 0:
            fm[:] = fm / COORD_SCALE

    print(f"目标数: {len(trajectories)}  帧数: {num_frames}")

    # —— 初始化 tracked_states：前 tau 帧用真实测量初始化 ——
    tracked_states = {traj['label']: [] for traj in trajectories}

    for traj in trajectories:
        lbl = traj['label']
        for t in range(tau):
            if traj['birth_frame'] <= t <= traj['death_frame']:
                # 在该帧找到此目标的测量
                found = None
                for i, al in enumerate(gt_associations[t]):
                    if al == lbl and i < len(measurements[t]):
                        found = measurements[t][i].copy()
                        break
                tracked_states[lbl].append(found if found is not None else traj['states'][t, :3].copy())
            else:
                tracked_states[lbl].append(np.zeros(3))

    # —— 逐帧推理 ——
    frame_pred_xyz   = []   # [num_eval_frames][n_active, 3]
    frame_true_xyz   = []
    frame_assoc_acc  = []   # [num_eval_frames] float
    frame_assoc_detail = [] # list of dicts

    DT      = 1.0    # 采样周期（与训练生成器一致）
    T_TOTAL = 30.0   # 场景持续时间（与训练生成器一致）

    with torch.no_grad():
        for frame_idx in range(tau, num_frames):
            past_np  = prepare_past_states(tracked_states, trajectories, frame_idx, tau, max_targets, DT, T_TOTAL)
            meas_np  = prepare_measurements(measurements[frame_idx], max_measurements)
            n_meas   = len(measurements[frame_idx])

            # 每帧实际目标数
            n_past_per_frame = []
            for t in range(frame_idx - tau, frame_idx):
                n_past_per_frame.append(
                    sum(1 for tr in trajectories if tr['birth_frame'] <= t <= tr['death_frame'])
                )

            past_t   = torch.FloatTensor(past_np).unsqueeze(0).to(device)
            meas_t   = torch.FloatTensor(meas_np).unsqueeze(0).to(device)
            npast_t  = torch.LongTensor([n_past_per_frame]).to(device)
            nmeas_t  = torch.LongTensor([n_meas]).to(device)

            match_prob, filtered_states, _ = model(past_t, meas_t, npast_t, nmeas_t)

            filtered_np = filtered_states[0].cpu().numpy()   # [max_targets, 3]
            match_np    = match_prob[0].cpu().numpy()         # [max_meas, max_targets+1]

            # ── 关联正确率 ──
            gt_assoc_frame = gt_associations[frame_idx][:n_meas]
            pred_assoc = np.argmax(match_np[:n_meas, 1:], axis=1) + 1  # 排除 clutter 列
            valid = gt_assoc_frame > 0
            detail_rows = []
            acc = 0.0
            if valid.sum() > 0:
                correct = (pred_assoc[valid] == gt_assoc_frame[valid]).sum()
                acc = correct / valid.sum()
                for i, vi in enumerate(np.where(valid)[0]):
                    mpos = measurements[frame_idx][vi] * COORD_SCALE
                    detail_rows.append({
                        'meas_idx': int(vi),
                        'pos_real': mpos.tolist(),
                        'pred_label': int(pred_assoc[vi]),
                        'gt_label':   int(gt_assoc_frame[vi]),
                        'correct':    bool(pred_assoc[vi] == gt_assoc_frame[vi])
                    })
            frame_assoc_acc.append(acc)
            frame_assoc_detail.append(detail_rows)

            # 控制台打印
            print(f"  帧{frame_idx:3d} | 测量总数={n_meas:3d} 真实={int(valid.sum()):2d} 杂波={int((~valid).sum()):2d} | 关联正确率={acc*100:.1f}%")
            for row in detail_rows:
                mark = "✓" if row['correct'] else "✗"
                p = row['pos_real']
                print(f"          {mark} 测量{row['meas_idx']} ({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})m  预测→轨迹{row['pred_label']}  真实→轨迹{row['gt_label']}")

            # ── 收集预测 / 真实 xyz ──
            pred_xyz_frame = []
            true_xyz_frame = []
            for idx_t, traj in enumerate(trajectories):
                if traj['birth_frame'] <= frame_idx <= traj['death_frame'] and idx_t < max_targets:
                    pred_xyz_frame.append(filtered_np[idx_t])
                    true_xyz_frame.append(traj['states'][frame_idx, :3])

            frame_pred_xyz.append(np.array(pred_xyz_frame) if pred_xyz_frame else np.empty((0, 3)))
            frame_true_xyz.append(np.array(true_xyz_frame) if true_xyz_frame else np.empty((0, 3)))

            # ── 更新 tracked_states ──
            for idx_t, traj in enumerate(trajectories):
                if idx_t < max_targets:
                    tracked_states[traj['label']].append(filtered_np[idx_t].copy())

    return {
        'trajectories':       trajectories,
        'measurements':       measurements,
        'gt_associations':    gt_associations,
        'frame_pred_xyz':     frame_pred_xyz,
        'frame_true_xyz':     frame_true_xyz,
        'frame_assoc_acc':    frame_assoc_acc,
        'frame_assoc_detail': frame_assoc_detail,
        'tau':                tau,
        'num_frames':         num_frames,
        'tracked_states':     tracked_states,
    }


# ============================================================
# 绘图函数
# ============================================================

def plot_3d_trajectories(result, scenario_idx, output_dir):
    """三维轨迹图：真实轨迹（虚线）+ 预测轨迹（实线）+ 测量点（灰色散点）"""
    trajectories  = result['trajectories']
    measurements  = result['measurements']
    tracked_states = result['tracked_states']
    tau           = result['tau']

    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection='3d')

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(trajectories), 1)))

    # ---- 先收集所有真实轨迹点，用于自适应坐标轴 ----
    traj_pts_all = []

    # 真实轨迹（虚线）
    for idx, traj in enumerate(trajectories):
        xyz = traj['states'][:, :3] * COORD_SCALE
        traj_pts_all.append(xyz)
        c   = colors[idx % len(colors)]
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                '--', color=c, linewidth=1.8, alpha=0.7,
                label=f'GT traj{traj["label"]}')

    # 预测轨迹（实线）
    for idx, traj in enumerate(trajectories):
        lbl  = traj['label']
        hist = tracked_states.get(lbl, [])
        pts  = []
        for s in hist[tau:]:
            if s is not None and not np.all(s == 0):
                pts.append(s * COORD_SCALE)
        if len(pts) > 1:
            pts = np.array(pts)
            traj_pts_all.append(pts)
            c   = colors[idx % len(colors)]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    '-', color=c, linewidth=2.5, alpha=1.0,
                    label=f'Pred traj{lbl}')

    # ---- 自适应坐标轴（基于轨迹点，不受杂波范围影响）----
    if traj_pts_all:
        all_traj = np.vstack(traj_pts_all)
        xmin, xmax = all_traj[:, 0].min(), all_traj[:, 0].max()
        ymin, ymax = all_traj[:, 1].min(), all_traj[:, 1].max()
        zmin, zmax = all_traj[:, 2].min(), all_traj[:, 2].max()
        # 让三轴范围相等，以免图形变形
        half = max(xmax - xmin, ymax - ymin, zmax - zmin) * 0.55 + 1e-3
        xc = (xmin + xmax) / 2; yc = (ymin + ymax) / 2; zc = (zmin + zmax) / 2
        ax.set_xlim(xc - half, xc + half)
        ax.set_ylim(yc - half, yc + half)
        ax.set_zlim(zc - half, zc + half)

    # 测量点：只绘制落在坐标轴范围内的点，避免杂波撑开视野
    xlim = ax.get_xlim(); ylim = ax.get_ylim(); zlim = ax.get_zlim()
    all_meas = []
    for fm in measurements[tau:]:
        if len(fm) > 0:
            m = fm * COORD_SCALE
            mask = ((m[:, 0] >= xlim[0]) & (m[:, 0] <= xlim[1]) &
                    (m[:, 1] >= ylim[0]) & (m[:, 1] <= ylim[1]) &
                    (m[:, 2] >= zlim[0]) & (m[:, 2] <= zlim[1]))
            if mask.any():
                all_meas.append(m[mask])
    if all_meas:
        all_meas = np.vstack(all_meas)
        ax.scatter(all_meas[:, 0], all_meas[:, 1], all_meas[:, 2],
                   c='gray', s=4, alpha=0.25, label='Measurements')

    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)
    ax.set_title(f'3D Trajectory — Scenario {scenario_idx+1}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=7, ncol=2, framealpha=0.6)

    plt.tight_layout()
    path = os.path.join(output_dir, f'scenario_{scenario_idx+1}_3d_trajectory.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [图] 3D轨迹图 → {path}")


def plot_error_curves(result, scenario_idx, output_dir):
    """
    误差曲线图（4 子图）：
      1. X 轴误差  2. Y 轴误差  3. Z 轴误差  4. 3D 欧氏距离误差
    每条轨迹单独一条曲线，再加平均线。
    """
    trajectories    = result['trajectories']
    frame_pred_xyz  = result['frame_pred_xyz']
    frame_true_xyz  = result['frame_true_xyz']
    tau             = result['tau']
    num_frames      = result['num_frames']
    eval_frames     = list(range(tau, num_frames))

    n_traj = len(trajectories)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_traj, 1)))

    # 逐轨迹逐帧误差
    # traj_errors[i] = {x:[], y:[], z:[], dist:[]} 按帧
    traj_errors = [{} for _ in range(n_traj)]
    for i in range(n_traj):
        traj_errors[i] = {'x': [], 'y': [], 'z': [], 'dist': [], 'frames': []}

    for fi, frame_idx in enumerate(eval_frames):
        pred = frame_pred_xyz[fi]   # [n_active, 3] 归一化
        true = frame_true_xyz[fi]   # [n_active, 3] 归一化
        n_active = min(len(pred), len(true), n_traj)
        for i in range(n_traj):
            if i < n_active:
                diff = (pred[i] - true[i]) * COORD_SCALE   # 反归一化到米
                traj_errors[i]['x'].append(abs(diff[0]))
                traj_errors[i]['y'].append(abs(diff[1]))
                traj_errors[i]['z'].append(abs(diff[2]))
                traj_errors[i]['dist'].append(np.linalg.norm(diff))
                traj_errors[i]['frames'].append(frame_idx)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_map = {
        'x':    (axes[0, 0], 'X轴误差 (m)',      'X Error (m)'),
        'y':    (axes[0, 1], 'Y轴误差 (m)',      'Y Error (m)'),
        'z':    (axes[1, 0], 'Z轴误差 (m)',      'Z Error (m)'),
        'dist': (axes[1, 1], '3D欧氏距离误差 (m)', '3D Dist Error (m)'),
    }

    for key, (ax, title_zh, ylabel) in ax_map.items():
        all_vals_by_frame = {}
        for i, traj in enumerate(trajectories):
            errs = traj_errors[i]
            if len(errs[key]) == 0:
                continue
            fr   = errs['frames']
            vals = errs[key]
            c = colors[i % len(colors)]
            ax.plot(fr, vals, '-', color=c, linewidth=1.2, alpha=0.7,
                    label=f'Traj {traj["label"]}')
            for f, v in zip(fr, vals):
                all_vals_by_frame.setdefault(f, []).append(v)

        # 平均线
        if all_vals_by_frame:
            avg_fr  = sorted(all_vals_by_frame.keys())
            avg_val = [np.mean(all_vals_by_frame[f]) for f in avg_fr]
            ax.plot(avg_fr, avg_val, 'k--', linewidth=2.2, label=f'Mean={np.mean(avg_val):.2f}m')

        ax.set_xlabel('Frame', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title_zh, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, ncol=2, framealpha=0.6)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Prediction Error — Scenario {scenario_idx+1}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, f'scenario_{scenario_idx+1}_error_curves.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [图] 误差曲线 → {path}")


def plot_ospa_curve(result, scenario_idx, output_dir):
    """OSPA 误差曲线"""
    frame_pred_xyz = result['frame_pred_xyz']
    frame_true_xyz = result['frame_true_xyz']
    tau            = result['tau']
    num_frames     = result['num_frames']
    eval_frames    = list(range(tau, num_frames))

    from metrics import OSPAMetric
    ospa_metric = OSPAMetric(c=50.0, p=1)   # cutoff=50m

    ospa_vals, ospa_loc_vals, ospa_card_vals = [], [], []
    for fi in range(len(eval_frames)):
        pred = frame_pred_xyz[fi] * COORD_SCALE   # 反归一化
        true = frame_true_xyz[fi] * COORD_SCALE
        if len(pred) == 0 and len(true) == 0:
            ospa_vals.append(0.0); ospa_loc_vals.append(0.0); ospa_card_vals.append(0.0)
            continue
        pred3 = pred if len(pred) > 0 else np.empty((0, 3))
        true3 = true if len(true) > 0 else np.empty((0, 3))
        d, loc, card = ospa_metric(pred3, true3)
        ospa_vals.append(d); ospa_loc_vals.append(loc); ospa_card_vals.append(card)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eval_frames, ospa_vals,      'b-',  linewidth=2,   label=f'OSPA  mean={np.mean(ospa_vals):.2f}m')
    ax.plot(eval_frames, ospa_loc_vals,  'g--', linewidth=1.5, label=f'OSPA-Loc  mean={np.mean(ospa_loc_vals):.2f}m')
    ax.plot(eval_frames, ospa_card_vals, 'r:',  linewidth=1.5, label=f'OSPA-Card mean={np.mean(ospa_card_vals):.2f}m')
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('OSPA (m)', fontsize=12)
    ax.set_title(f'OSPA Error — Scenario {scenario_idx+1}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f'scenario_{scenario_idx+1}_ospa.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [图] OSPA曲线 → {path}")

    return np.mean(ospa_vals)


def plot_association_accuracy(result, scenario_idx, output_dir):
    """逐帧关联正确率柱状图"""
    accs = result['frame_assoc_acc']
    tau  = result['tau']
    frames = list(range(tau, tau + len(accs)))

    fig, ax = plt.subplots(figsize=(12, 5))
    bar_colors = ['#2ecc71' if a >= 0.9 else '#e67e22' if a >= 0.7 else '#e74c3c' for a in accs]
    ax.bar(frames, [a * 100 for a in accs], color=bar_colors, alpha=0.8, width=0.7)
    mean_acc = np.mean(accs) * 100
    ax.axhline(mean_acc, color='navy', linestyle='--', linewidth=2, label=f'平均 {mean_acc:.1f}%')
    ax.set_ylim(0, 105)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('关联正确率 (%)', fontsize=12)
    ax.set_title(f'数据关联正确率 — Scenario {scenario_idx+1}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(output_dir, f'scenario_{scenario_idx+1}_association_acc.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [图] 关联正确率 → {path}")


def plot_overall_summary(all_results, output_dir):
    """多场景汇总对比图（关联正确率 + OSPA + 平均3D误差）"""
    n = len(all_results)
    scenario_ids = list(range(1, n + 1))

    avg_acc  = [r['avg_assoc_acc']  * 100 for r in all_results]
    avg_ospa = [r['avg_ospa']             for r in all_results]
    avg_err  = [r['avg_dist_error']       for r in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].bar(scenario_ids, avg_acc, color='steelblue', alpha=0.85)
    axes[0].axhline(np.mean(avg_acc), color='r', linestyle='--', linewidth=2,
                    label=f'Overall={np.mean(avg_acc):.1f}%')
    axes[0].set_title('数据关联正确率 (%)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Scenario'); axes[0].set_ylim(0, 105)
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(scenario_ids, avg_ospa, color='tomato', alpha=0.85)
    axes[1].axhline(np.mean(avg_ospa), color='k', linestyle='--', linewidth=2,
                    label=f'Overall={np.mean(avg_ospa):.2f}m')
    axes[1].set_title('OSPA 误差 (m)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Scenario')
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')

    axes[2].bar(scenario_ids, avg_err, color='mediumseagreen', alpha=0.85)
    axes[2].axhline(np.mean(avg_err), color='k', linestyle='--', linewidth=2,
                    label=f'Overall={np.mean(avg_err):.2f}m')
    axes[2].set_title('平均3D距离误差 (m)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Scenario')
    axes[2].legend(); axes[2].grid(True, alpha=0.3, axis='y')

    fig.suptitle('Overall Evaluation Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'overall_summary.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  [图] 汇总对比图 → {path}")


# ============================================================
# main
# ============================================================

def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # —— 加载模型 ——
    print(f"\n加载检查点: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = BAIT(
        d_model=256, nhead=8,
        num_encoder_layers=6,
        num_associate_decoder_layers=3,
        num_filtering_decoder_layers=6,
        dim_feedforward_encoder=2048,
        dim_feedforward_associate=1024,
        dim_feedforward_filtering=2048,
        max_targets=args.max_targets
        # state_dim=5, measurement_dim=3 — 使用默认值
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"模型加载成功！（来自训练步骤 {ckpt.get('step', '?')}）")

    # ── 场景来源 ──────────────────────────────────────────────────
    print(f"\n场景来源: {args.scene_source.upper()}")

    if args.scene_source == 'new':
        print(f"  随机种子:     {args.seed}")
        print(f"  交叉场景概率: {args.crossing_prob * 100:.0f}%")
        print(f"  星形交叉概率: {args.star_crossing_prob * 100:.0f}%")
        generator = MTTDataGeneratorWithCrossing(
            task_type=args.task_type,
            crossing_probability=args.crossing_prob,
            star_crossing_probability=args.star_crossing_prob,
            num_star_trajectories=(4, 8),
            P_d=1.0,
            seed=args.seed
        )

        def get_scenario(_idx):
            return generator.generate_single_scenario()

    else:  # test_split
        # 优先从 checkpoint 同目录加载训练时保存的测试场景文件
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        saved_path = os.path.join(ckpt_dir, 'test_scenarios.pkl')

        if os.path.exists(saved_path):
            print(f"  [test_split] 加载训练时保存的测试场景: {saved_path}")
            with open(saved_path, 'rb') as _f:
                all_saved_scenarios = pickle.load(_f)
            print(f"  共 {len(all_saved_scenarios)} 条场景，本次使用前 {args.num_scenarios} 条")
            def get_scenario(idx):
                return all_saved_scenarios[idx % len(all_saved_scenarios)]
        else:
            # 回退：用相同 seed 重新生成（要求参数与训练完全一致）
            print(f"  [test_split] 未找到 {saved_path}，回退到 seed={args.test_seed} 重新生成")
            print(f"  ⚠ 请确保 --test-crossing-prob 与训练 config 中的 crossing_probability 一致！")
            print(f"  交叉场景概率: {args.test_crossing_prob * 100:.0f}%")
            test_dataset = MTTDatasetWithCrossing(
                num_scenarios=args.num_scenarios,
                tau=args.tau,
                max_targets=args.max_targets,
                max_measurements=args.max_measurements,
                task_type=args.task_type,
                seed=args.test_seed,
                crossing_probability=args.test_crossing_prob,
            )
            def get_scenario(idx):
                return test_dataset.scenarios[idx % len(test_dataset.scenarios)]

    # —— 逐场景评估 ——
    all_results = []

    for s_idx in range(args.num_scenarios):
        scenario = get_scenario(s_idx)
        result = evaluate_scenario(
            model, scenario,
            args.tau, args.max_targets, args.max_measurements,
            device, s_idx
        )

        # 绘图
        plot_3d_trajectories(result, s_idx, args.output_dir)
        plot_error_curves(result, s_idx, args.output_dir)
        avg_ospa = plot_ospa_curve(result, s_idx, args.output_dir)
        plot_association_accuracy(result, s_idx, args.output_dir)

        # 场景统计
        avg_acc = np.mean(result['frame_assoc_acc'])

        all_dist_errors = []
        for fi in range(len(result['frame_pred_xyz'])):
            pred = result['frame_pred_xyz'][fi]
            true = result['frame_true_xyz'][fi]
            n = min(len(pred), len(true))
            if n > 0:
                errs = np.linalg.norm((pred[:n] - true[:n]) * COORD_SCALE, axis=1)
                all_dist_errors.extend(errs.tolist())
        avg_dist = np.mean(all_dist_errors) if all_dist_errors else 0.0

        print(f"\n  场景{s_idx+1} 汇总: 关联正确率={avg_acc*100:.1f}%  OSPA={avg_ospa:.2f}m  平均3D误差={avg_dist:.2f}m")

        all_results.append({
            'scenario_id':   s_idx + 1,
            'avg_assoc_acc': float(avg_acc),
            'avg_ospa':      float(avg_ospa),
            'avg_dist_error': float(avg_dist),
        })

    # —— 汇总图 ——
    if len(all_results) > 1:
        plot_overall_summary(all_results, args.output_dir)

    # —— 控制台汇总 ——
    print(f"\n{'='*60}")
    print("Overall Evaluation Summary")
    print(f"{'='*60}")
    print(f"{'场景':<8} {'关联正确率':>12} {'OSPA(m)':>10} {'平均3D误差(m)':>14}")
    print("-" * 50)
    for r in all_results:
        print(f"  {r['scenario_id']:<6} {r['avg_assoc_acc']*100:>11.1f}% {r['avg_ospa']:>10.2f} {r['avg_dist_error']:>14.2f}")
    print("-" * 50)
    print(f"  {'总均值':<6} "
          f"{np.mean([r['avg_assoc_acc'] for r in all_results])*100:>11.1f}% "
          f"{np.mean([r['avg_ospa']      for r in all_results]):>10.2f} "
          f"{np.mean([r['avg_dist_error'] for r in all_results]):>14.2f}")
    print(f"{'='*60}")

    # —— 保存 JSON ——
    summary = {
        'checkpoint': args.checkpoint,
        'num_scenarios': args.num_scenarios,
        'scenarios': all_results,
        'overall': {
            'avg_assoc_acc':  float(np.mean([r['avg_assoc_acc']  for r in all_results])),
            'avg_ospa':       float(np.mean([r['avg_ospa']       for r in all_results])),
            'avg_dist_error': float(np.mean([r['avg_dist_error'] for r in all_results])),
        }
    }
    json_path = os.path.join(args.output_dir, 'evaluation_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"\n结果已保存: {args.output_dir}/")


if __name__ == '__main__':
    main()
