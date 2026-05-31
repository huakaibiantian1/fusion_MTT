"""
jpda_tracker.py — Joint Probabilistic Data Association Filter (JPDAF)

参考: Bar-Shalom, Y. et al. (2011). Tracking and Data Fusion: A Handbook of Algorithms.
      Chapter 6: Joint Probabilistic Data Association.

算法流程（每帧）:
  1. Kalman 预测各轨迹
  2. 门限过滤：计算每条量测与每条轨迹的 Mahalanobis 距离，超出门限的直接排除
  3. JPDA 近似关联概率:
       β_ij = P_d * L_ij / (κ + Σ_i' P_d * L_i'j)   -- 归一化似然比
       β_i0 = 1 - Σ_j β_ij                             -- 漏检概率
  4. 加权量测更新 (JPDAF 协方差更新含扩散项):
       ν̄_i = Σ_j β_ij * (z_j - Hx̂_i)
       x̂_i = x̂_i + K_i * ν̄_i
       P_i = β_i0 * P̄ + (1-β_i0) * P̃ + K * (Σ β_ij νij νij^T - ν̄ν̄^T) * K^T
  5. 航迹管理：连续 N_confirm 帧命中 → 确认；连续 N_delete 帧漏检 → 删除
  6. 未关联量测启动候选航迹

坐标约定：与 phd_filter.py / mht_tracker.py 完全一致
  - 状态  [x, y, z, vx, vy, vz]，所有分量均以 COORD_SCALE 归一化
  - 量测  [x, y, z]，同样归一化
"""

import copy as _copy
import numpy as np
from scipy.optimize import linear_sum_assignment

from data_generation import COORD_SCALE   # 50000.0 m


# ====================================================================
# 单条 Kalman 轨迹
# ====================================================================

class _Track:
    _id_ctr = 0

    def __init__(self, z, P_birth, F, H, Q, R, label=None):
        _Track._id_ctr += 1
        self.id    = _Track._id_ctr
        self.label = label          # GT init 时记录关联的真实目标标签

        self.x = np.zeros(6)
        self.x[:3] = z.copy()
        self.P = P_birth.copy()

        self.F = F; self.H = H; self.Q = Q; self.R = R

        self.hits      = 1
        self.misses    = 0
        self.confirmed = False
        self.age       = 0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def innovation_cov(self):
        """返回 (z_pred, S)"""
        return self.H @ self.x, self.H @ self.P @ self.H.T + self.R

    def kalman_gain(self):
        """返回 (K, P_upd) — 使用 Joseph 形式"""
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.solve(S + np.eye(3)*1e-12, np.eye(3))
        except np.linalg.LinAlgError:
            K = np.zeros((6, 3))
        I_KH  = np.eye(6) - K @ self.H
        P_upd = I_KH @ self.P       # 简化 Kalman 协方差 P̃ = (I-KH)P̄
        return K, P_upd

    def mahal_dist2(self, z):
        z_pred, S = self.innovation_cov()
        dz = z - z_pred
        try:
            return float(dz @ np.linalg.solve(S + np.eye(3)*1e-12, dz))
        except Exception:
            return float('inf')


# ====================================================================
# JPDA 跟踪器
# ====================================================================

class JPDATracker:
    """
    Joint Probabilistic Data Association Filter 多目标跟踪器。

    参数
    ----
    dt          : 采样间隔（s）
    sigma_q     : 过程噪声（m/s^{3/2}），内部自动归一化
    sigma_r     : 测量噪声标准差（m），内部自动归一化
    P_d         : 检测概率
    lambda_c    : 每帧杂波期望数量（泊松）
    gate_gamma  : 门限（Mahalanobis 距离平方），chi²(3,p)
                  p=0.997 → 14.2，默认 16 稍留余量
    N_confirm   : 连续命中多少帧后确认轨迹
    N_delete    : 连续漏检多少帧后删除轨迹
    """

    def __init__(
        self,
        dt:         float = 1.0,
        sigma_q:    float = 0.5,
        sigma_r:    float = 50.0,
        P_d:        float = 0.95,
        lambda_c:   float = 10.0,
        gate_gamma: float = 16.0,
        N_confirm:  int   = 2,
        N_delete:   int   = 3,
    ):
        C = COORD_SCALE
        self.P_d        = P_d
        self.gate_gamma = gate_gamma
        self.N_confirm  = N_confirm
        self.N_delete   = N_delete

        self.sigma_r_n = sigma_r / C

        # 运动模型（CV）
        self.F = np.eye(6)
        self.F[0, 3] = dt; self.F[1, 4] = dt; self.F[2, 5] = dt

        # 过程噪声 Q
        q = (sigma_q / C) ** 2
        self.Q = q * np.array([
            [dt**4/4, 0,       0,       dt**3/2, 0,       0      ],
            [0,       dt**4/4, 0,       0,       dt**3/2, 0      ],
            [0,       0,       dt**4/4, 0,       0,       dt**3/2],
            [dt**3/2, 0,       0,       dt**2,   0,       0      ],
            [0,       dt**3/2, 0,       0,       dt**2,   0      ],
            [0,       0,       dt**3/2, 0,       0,       dt**2  ],
        ])

        # 量测模型
        self.H = np.zeros((3, 6)); self.H[:3, :3] = np.eye(3)
        self.R = (self.sigma_r_n ** 2) * np.eye(3)

        # 新生轨迹初始协方差
        v_init_n     = 500.0 / C
        self.P_birth = np.diag([self.sigma_r_n**2]*3 + [v_init_n**2]*3)

        # 杂波空间密度
        self.kappa = lambda_c / ((4.0 / 3.0) * np.pi)

        self.tracks: list = []

    # ------------------------------------------------------------------
    @staticmethod
    def _gauss_pdf(z, nu, S):
        d, dz = len(z), z - nu
        try:
            S_reg = S + np.eye(d) * 1e-12
            S_inv = np.linalg.solve(S_reg, np.eye(d))
            _, log_det = np.linalg.slogdet(S_reg)
            log_pdf = -0.5 * (d * np.log(2*np.pi) + log_det + dz @ S_inv @ dz)
            return float(np.exp(np.clip(log_pdf, -700, 700)))
        except Exception:
            return 1e-300

    # ------------------------------------------------------------------
    def step(self, measurements: list) -> dict:
        """处理一帧量测，返回已确认轨迹 {track_id: state[6]}。"""
        n_ms = len(measurements)

        # 1. 预测
        for tr in self.tracks:
            tr.predict()
            tr.age += 1

        if not self.tracks:
            for z in measurements:
                self.tracks.append(
                    _Track(z, self.P_birth, self.F, self.H, self.Q, self.R))
            self._manage_tracks()
            return {}

        n_tr = len(self.tracks)

        # 2. 似然矩阵 L[i][j] （门限过滤）
        L = np.zeros((n_tr, max(n_ms, 1)))
        zpreds = []
        for i, tr in enumerate(self.tracks):
            z_pred, S = tr.innovation_cov()
            zpreds.append(z_pred)
            for j, z in enumerate(measurements):
                if tr.mahal_dist2(z) <= self.gate_gamma:
                    L[i, j] = self._gauss_pdf(z, z_pred, S)

        # 3. JPDA 近似关联概率 β[i][j]
        beta = np.zeros((n_tr, max(n_ms, 1)))
        for j in range(n_ms):
            denom = self.kappa + self.P_d * L[:, j].sum()
            if denom > 1e-300:
                beta[:, j] = self.P_d * L[:, j] / denom
        beta_i0 = np.clip(1.0 - beta[:, :n_ms].sum(axis=1), 0.0, 1.0)

        # 4. JPDAF 状态更新
        for i, tr in enumerate(self.tracks):
            x_pred = tr.x.copy()           # 保存预测状态
            P_pred = tr.P.copy()           # 保存预测协方差
            K, P_tilde = tr.kalman_gain()  # P̃ = (I-KH)P̄

            z_pred = zpreds[i]

            # 加权新息 ν̄ = Σ_j β_ij * (z_j - ẑ)
            nu_bar = np.zeros(3)
            for j in range(n_ms):
                if beta[i, j] > 1e-10:
                    nu_bar += beta[i, j] * (measurements[j] - z_pred)

            # 扩散协方差项: K*(Σβ_ij νij νij^T - ν̄ν̄^T)*K^T
            spread = np.zeros((6, 6))
            for j in range(n_ms):
                if beta[i, j] > 1e-10:
                    nu_j  = measurements[j] - z_pred    # 3D
                    Knu_j = K @ nu_j                     # 6D
                    spread += beta[i, j] * np.outer(Knu_j, Knu_j)
            Knu_bar = K @ nu_bar
            spread -= np.outer(Knu_bar, Knu_bar)

            # JPDAF 协方差更新
            P_new = beta_i0[i] * P_pred + (1 - beta_i0[i]) * P_tilde + spread

            # 状态更新
            tr.x = x_pred + K @ nu_bar
            tr.P = P_new

            # 航迹管理：命中判断
            if beta[i, :n_ms].sum() > 0.1:
                tr.hits  += 1
                tr.misses = 0
            else:
                tr.misses += 1

        # 5. 从未关联量测新建候选轨迹
        for j in range(n_ms):
            if beta[:, j].sum() < 0.2:
                self.tracks.append(
                    _Track(measurements[j], self.P_birth,
                           self.F, self.H, self.Q, self.R))

        # 6. 确认 & 删除
        self._manage_tracks()

        return {tr.id: tr.x.copy() for tr in self.tracks if tr.confirmed}

    def _manage_tracks(self):
        survived = []
        for tr in self.tracks:
            if tr.misses >= self.N_delete:
                continue
            if tr.hits >= self.N_confirm:
                tr.confirmed = True
            survived.append(tr)
        self.tracks = survived


# ====================================================================
# 场景评估入口
# ====================================================================

def evaluate_scenario_jpda(
    scenario,
    sigma_q:     float = 0.5,
    sigma_r:     float = 50.0,
    P_d:         float = 0.95,
    lambda_c:    float = 10.0,
    gate_gamma:  float = 16.0,
    N_confirm:   int   = 2,
    N_delete:    int   = 3,
    tau:         int   = 4,
    use_gt_init: bool  = False,
) -> dict:
    """
    使用 JPDAF 评估单个场景。
    返回格式与 evaluate_3d.evaluate_scenario() 完全一致。

    use_gt_init=True：前 tau 帧只向跟踪器喂真实目标量测（过滤杂波），
    与 BAIT 的真值初始化等价。
    """
    scenario = _copy.deepcopy(scenario)
    trajectories, measurements, gt_associations = scenario
    num_frames = len(measurements)

    # 坐标归一化
    for traj in trajectories:
        traj['states'][:, :3] /= COORD_SCALE
    for fm in measurements:
        if len(fm) > 0:
            fm[:] = fm / COORD_SCALE

    init_mode = "真值热启动" if use_gt_init else "纯量测自主"
    print(f"[JPDA] 目标数: {len(trajectories)}  帧数: {num_frames}  起始模式: {init_mode}")
    print(f"[JPDA] sigma_r={sigma_r}m  sigma_q={sigma_q}  P_d={P_d}  lambda_c={lambda_c}")
    print(f"[JPDA] gate_gamma={gate_gamma}  N_confirm={N_confirm}  N_delete={N_delete}")

    _Track._id_ctr = 0

    tracker = JPDATracker(
        sigma_q=sigma_q, sigma_r=sigma_r,
        P_d=P_d, lambda_c=lambda_c,
        gate_gamma=gate_gamma,
        N_confirm=N_confirm, N_delete=N_delete,
    )

    frame_jpda_states: list = []  # list[dict{id: state[6]}]
    tid_to_label:      dict = {}  # 持久化 track_id → 真实标签

    for t in range(num_frames):
        # 真值热启动：前 tau 帧只喂真实目标量测
        if use_gt_init and t < tau:
            gt_assoc = gt_associations[t]
            n = len(measurements[t])
            z_list = [
                measurements[t][i] for i in range(n)
                if i < len(gt_assoc) and gt_assoc[i] > 0
            ]
        else:
            z_list = list(measurements[t])

        confirmed = tracker.step(z_list)
        frame_jpda_states.append(confirmed)

        # 实时更新 tid_to_label（匈牙利匹配新出现的轨迹 → 真实目标）
        active_trajs = [tr for tr in trajectories
                        if tr['birth_frame'] <= t <= tr['death_frame']]
        if confirmed and active_trajs:
            new_tids  = [tid for tid in confirmed if tid not in tid_to_label]
            used_lbls = set(tid_to_label.values())
            free_lbls = [tr['label'] for tr in active_trajs
                         if tr['label'] not in used_lbls]

            if new_tids and free_lbls:
                est_arr  = np.array([confirmed[tid][:3] for tid in new_tids])
                true_arr = np.array([tr['states'][t, :3]
                                     for tr in active_trajs
                                     if tr['label'] in free_lbls])
                if len(est_arr) > 0 and len(true_arr) > 0:
                    cost = np.linalg.norm(
                        est_arr[:, None, :] - true_arr[None, :, :], axis=2)
                    row_ind, col_ind = linear_sum_assignment(cost)
                    for r, c in zip(row_ind, col_ind):
                        tid_to_label[new_tids[r]] = free_lbls[c]

        print(f"  帧{t:3d} | 量测={len(z_list):2d}  确认轨迹={len(confirmed):2d}"
              f"  总轨迹={len(tracker.tracks):2d}")

    # 前 tau 帧 tracked_states 用真实轨迹填充
    tracked_states = {traj['label']: [] for traj in trajectories}
    for traj in trajectories:
        lbl = traj['label']
        for t in range(tau):
            if traj['birth_frame'] <= t <= traj['death_frame']:
                tracked_states[lbl].append(traj['states'][t, :3].copy())
            else:
                tracked_states[lbl].append(np.zeros(3))

    sigma_r_n = sigma_r / COORD_SCALE

    frame_pred_xyz     = []
    frame_true_xyz     = []
    frame_assoc_acc    = []
    frame_assoc_detail = []

    for frame_idx in range(tau, num_frames):
        active_trajs = [tr for tr in trajectories
                        if tr['birth_frame'] <= frame_idx <= tr['death_frame']]
        true_xyz = (np.array([tr['states'][frame_idx, :3] for tr in active_trajs])
                    if active_trajs else np.empty((0, 3)))

        states = frame_jpda_states[frame_idx]

        # 用 tid_to_label 映射填充 aligned_pred，无匹配轨迹 → NaN
        aligned_pred = np.full_like(true_xyz, np.nan)
        for j, tr in enumerate(active_trajs):
            lbl = tr['label']
            for tid, mapped_lbl in tid_to_label.items():
                if mapped_lbl == lbl and tid in states:
                    aligned_pred[j] = states[tid][:3]
                    break

        frame_pred_xyz.append(aligned_pred)
        frame_true_xyz.append(true_xyz)

        for j, tr in enumerate(active_trajs):
            tracked_states[tr['label']].append(aligned_pred[j].copy())
        for tr in trajectories:
            if tr not in active_trajs:
                tracked_states[tr['label']].append(np.zeros(3))

        # 关联正确率
        gt_assoc_frame = gt_associations[frame_idx]
        meas_frame     = measurements[frame_idx]
        n_meas         = len(meas_frame)
        acc            = 0.0
        detail_rows    = []

        ests = [states[tid][:3] for tid in states]
        valid_idx = (np.where(gt_assoc_frame[:n_meas] > 0)[0]
                     if n_meas > 0 else np.array([]))

        if len(valid_idx) > 0 and len(ests) > 0:
            est_arr   = np.array(ests)
            threshold = 3.0 * sigma_r_n
            correct   = 0
            for i in valid_idx:
                z        = meas_frame[i]
                gt_lbl   = int(gt_assoc_frame[i])
                dists    = np.linalg.norm(est_arr - z, axis=1)
                near_est = int(np.argmin(dists))
                matched  = bool(dists[near_est] < threshold)
                if matched:
                    correct += 1
                detail_rows.append({
                    'meas_idx':   int(i),
                    'pos_real':   (z * COORD_SCALE).tolist(),
                    'pred_label': near_est + 1,
                    'gt_label':   gt_lbl,
                    'correct':    matched,
                })
            acc = correct / len(valid_idx)

        frame_assoc_acc.append(acc)
        frame_assoc_detail.append(detail_rows)
        print(f"  帧{frame_idx:3d} | 关联正确率={acc*100:.1f}%  活跃目标={len(active_trajs)}")

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
