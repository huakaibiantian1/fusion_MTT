"""
最小化测试：单组件 + 单量测，看 update 后权重是否 ≈ 1.0
"""
import numpy as np
from phd_filter import GMPHDFilter
from data_generation import COORD_SCALE

phd = GMPHDFilter(dt=1.0, sigma_q=0.5, sigma_r=50.0, P_d=0.95, P_s=0.99,
                  lambda_c=10.0, birth_weight=0.0, prune_thresh=1e-4, merge_dist=4.0)

# 注入一个 w=1.0 的分量，位置在归一化空间内
pos_n = np.array([-4212.15 / COORD_SCALE, -23139.70 / COORD_SCALE, 4326.37 / COORD_SCALE])
vel_n = np.array([115.44 / COORD_SCALE, 81.65 / COORD_SCALE, -7.58 / COORD_SCALE])
m = np.concatenate([pos_n, vel_n])
v_init_n = 500.0 / COORD_SCALE
P = np.diag([phd.sigma_r_n**2]*3 + [v_init_n**2]*3)
phd.components = [(1.0, m, P)]

print("=== 注入分量 ===")
print(f"  pos (m): {pos_n * COORD_SCALE}")
print(f"  w = 1.0")
print(f"  sigma_pos_n = {phd.sigma_r_n:.6f}  sigma_vel_n = {v_init_n:.6f}")

# predict
phd.predict()
w_pred, m_pred, P_pred = phd.components[0]
print(f"\n=== predict 后 ===")
print(f"  w = {w_pred:.4f}")
print(f"  pos_pred (m): {m_pred[:3] * COORD_SCALE}")
S = phd.H @ P_pred @ phd.H.T + phd.R
print(f"  S diag (normalized): {np.diag(S)}")
print(f"  gate 3-sigma (m): {3 * np.sqrt(np.diag(S)[0]) * COORD_SCALE:.1f}")
print(f"  kappa: {phd.kappa:.4f}")

# 构造一个"真实量测"：目标位置 + 50m 噪声
z_true = m_pred[:3] + np.random.randn(3) * phd.sigma_r_n
print(f"\n=== 喂入 1 条真实量测 ===")
print(f"  z_true (m): {z_true * COORD_SCALE}")
print(f"  dist_to_pred (m): {np.linalg.norm(z_true - m_pred[:3]) * COORD_SCALE:.1f}")

# 手动计算 gauss_pdf 和 lhood
from phd_filter import GMPHDFilter as G
nu = phd.H @ m_pred
pdf_val = G._gauss_pdf(z_true, nu, S)
lhood = phd.P_d * w_pred * pdf_val
denom = phd.kappa + lhood
w_upd = lhood / denom
print(f"  gauss_pdf = {pdf_val:.4e}")
print(f"  lhood = P_d * w * pdf = {lhood:.4e}")
print(f"  kappa = {phd.kappa:.4f}")
print(f"  denom = kappa + lhood = {denom:.4e}")
print(f"  w_upd = {w_upd:.6f}")

# 实际调用 update
phd.update([z_true], birth_weight=0.0)
print(f"\n=== update 后 (birth_weight=0) ===")
print(f"  分量数 = {len(phd.components)}")
for i, (w, m_, _) in enumerate(phd.components):
    print(f"  comp[{i}] w={w:.4f}  pos(m)={m_[:3]*COORD_SCALE}")

phd.prune_and_merge()
ests = phd.extract_states()
print(f"\n=== prune+merge 后 ===")
for i, (w, m_, _) in enumerate(phd.components):
    print(f"  comp[{i}] w={w:.4f}  pos(m)={m_[:3]*COORD_SCALE}")
print(f"  提取目标数 = {len(ests)}")
