"""
逐步追踪 PHD 真值热启动时每帧分量权重变化
"""
import numpy as np, copy
from data_generation_multi_scenario import MTTDataGeneratorMultiScenario
from data_generation import COORD_SCALE
from phd_filter import GMPHDFilter

gen = MTTDataGeneratorMultiScenario(task_type=1, seed=42, n_cross_traj=3)
sc  = gen.generate_by_type('crossing')
trajectories, measurements, gt_associations = copy.deepcopy(sc)

# 归一化
for traj in trajectories:
    traj['states'][:, :3] /= COORD_SCALE
for fm in measurements:
    if len(fm) > 0:
        fm[:] = fm / COORD_SCALE

tau = 4
phd = GMPHDFilter(dt=1.0, sigma_q=0.5, sigma_r=50.0, P_d=0.95, P_s=0.99,
                  lambda_c=10.0, birth_weight=0.05, prune_thresh=1e-4, merge_dist=4.0)

for t in range(6):  # 只跑前6帧
    phd.predict()
    print(f"\n=== 帧 {t} — predict 后 ===")
    for i, (w, m, P) in enumerate(phd.components):
        print(f"  comp[{i}] w={w:.4f}  pos={m[:3]*COORD_SCALE}")

    # GT 过滤量测
    if t < tau:
        gt_assoc = gt_associations[t]
        n = len(measurements[t])
        z_list = [measurements[t][i] for i in range(n)
                  if i < len(gt_assoc) and gt_assoc[i] > 0]
    else:
        z_list = list(measurements[t])

    effective_bw = 0.0 if t >= tau else None

    phd.update(z_list, birth_weight=effective_bw)
    print(f"  — update 后（{len(z_list)} 条量测，bw={effective_bw}），分量数={len(phd.components)}")
    for i, (w, m, P) in enumerate(phd.components[:6]):
        print(f"  comp[{i}] w={w:.4f}  pos={m[:3]*COORD_SCALE}")

    phd.prune_and_merge()
    print(f"  — prune+merge 后，分量数={len(phd.components)}")
    for i, (w, m, P) in enumerate(phd.components):
        print(f"  comp[{i}] w={w:.4f}  pos={m[:3]*COORD_SCALE}")

    # 真值注入
    if t == tau - 1:
        print("  *** 真值注入 ***")
        phd.components = []
        v_init_n = 500.0 / COORD_SCALE
        for traj in trajectories:
            if traj['birth_frame'] <= t <= traj['death_frame']:
                pos_now = traj['states'][t, :3]
                if t > 0 and traj['birth_frame'] < t:
                    vel_est = (traj['states'][t, :3] - traj['states'][t-1, :3]) / 1.0
                else:
                    vel_est = np.zeros(3)
                m_gt = np.concatenate([pos_now, vel_est])
                P_gt = np.diag([phd.sigma_r_n**2]*3 + [v_init_n**2]*3)
                phd.components.append((1.0, m_gt, P_gt))
                print(f"    注入 traj '{traj['label']}': pos={pos_now*COORD_SCALE}  vel={vel_est*COORD_SCALE}")

    ests = phd.extract_states()
    print(f"  提取目标数={len(ests)}")
