"""
深度调试：检查帧 4 的量测是否真的落在预测位置附近
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

# 模拟前 3 帧 + 热启动
for t in range(4):
    phd.predict()
    gt_assoc = gt_associations[t]
    n = len(measurements[t])
    z_list = [measurements[t][i] for i in range(n)
              if i < len(gt_assoc) and gt_assoc[i] > 0]
    phd.update(z_list)
    phd.prune_and_merge()
    if t == tau - 1:
        phd.components = []
        v_init_n = 500.0 / COORD_SCALE
        for traj in trajectories:
            if traj['birth_frame'] <= t <= traj['death_frame']:
                pos_now = traj['states'][t, :3]
                vel_est = (traj['states'][t, :3] - traj['states'][t-1, :3]) if t > 0 else np.zeros(3)
                m_gt = np.concatenate([pos_now, vel_est])
                P_gt = np.diag([phd.sigma_r_n**2]*3 + [v_init_n**2]*3)
                phd.components.append((1.0, m_gt, P_gt))

# 现在做帧 4 的 predict
t = 4
phd.predict()

print("=== 帧 4 predict 后，目标分量 (归一化) ===")
for i, (w, m, P) in enumerate(phd.components):
    print(f"  comp[{i}] w={w:.4f}  pos={m[:3]}  sigma_pos={np.sqrt(np.diag(P)[:3])}")

# 帧 4 的全部量测
z_list = list(measurements[t])
print(f"\n=== 帧 4 共 {len(z_list)} 条量测 ===")
print("量测与预测位置的 L2 距离（归一化）：")
for iz, z in enumerate(z_list):
    dists = [np.linalg.norm(z - m[:3]) for _, m, _ in phd.components]
    gt_flag = "★" if (iz < len(gt_associations[t]) and gt_associations[t][iz] > 0) else " "
    print(f"  {gt_flag} z[{iz:2d}]={z}  dists={[f'{d:.5f}' for d in dists]}")

# 计算 kappa 和 PDF 峰值
print(f"\n=== 参数 ===")
print(f"  kappa = {phd.kappa:.4f}")
print(f"  sigma_r_n = {phd.sigma_r_n:.6f}")
print(f"  3*sigma_r_n = {3*phd.sigma_r_n:.6f}")

# 手动算第一个目标分量对最近量测的 lhood
comp0_w, comp0_m, comp0_P = phd.components[0]
S0 = phd.H @ comp0_P @ phd.H.T + phd.R
print(f"\n  comp[0] S 对角 = {np.diag(S0)}")
print(f"  comp[0] gate_radius(3-sigma) = {3 * np.sqrt(np.diag(S0)[0]):.6f}")

# 找最近的量测
dists0 = [np.linalg.norm(z - comp0_m[:3]) for z in z_list]
closest_idx = int(np.argmin(dists0))
z_closest = z_list[closest_idx]
print(f"\n  comp[0] 最近量测 z[{closest_idx}]={z_closest}  dist={dists0[closest_idx]:.6f}")

from phd_filter import GMPHDFilter as G
pdf_val = G._gauss_pdf(z_closest, phd.H @ comp0_m, S0)
lhood_k0 = phd.P_d * comp0_w * pdf_val
print(f"  gauss_pdf = {pdf_val:.4e}")
print(f"  lhood_k[0] = P_d * w * pdf = {lhood_k0:.4e}")
print(f"  kappa = {phd.kappa:.4f}")
print(f"  denom = kappa + lhood_k[0] = {phd.kappa + lhood_k0:.4e}")
print(f"  w_upd = lhood_k / denom = {lhood_k0 / (phd.kappa + lhood_k0):.6f}")
