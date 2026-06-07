"""
phd_filter.py — GM-PHD 滤波器（Gaussian Mixture Probability Hypothesis Density）

参考：Vo, B.-N. and Ma, W.-K. (2006). The Gaussian Mixture Probability Hypothesis
     Density Filter. IEEE Trans. Signal Process., 54(11), 4091-4104.

坐标约定：
  - 状态  [x, y, z, vx, vy, vz]，所有分量均以 COORD_SCALE 归一化（同 evaluate_3d.py）
  - 量测  [x, y, z]，同样归一化
  - sigma_r / sigma_q 可由用户以 **原始米制单位** 指定，内部自动转换
"""

import copy as _copy
import numpy as np
from scipy.optimize import linear_sum_assignment

from data_generation import COORD_SCALE          # 50000.0 m


# ====================================================================
# 核心 GM-PHD 滤波器
# ====================================================================

class GMPHDFilter:
    """
    高斯混合 PHD 滤波器。

    参数
    ----
    dt          : 采样间隔（s）
    sigma_q     : 过程噪声加速度谱密度（m/s^{3/2}），内部转换为归一化单位
    sigma_r     : 测量噪声标准差（m），内部转换为归一化单位
    P_d         : 检测概率
    P_s         : 目标存活概率
    lambda_c    : 每帧杂波期望数量
    birth_weight: 每条测量处新生分量的初始权重（每帧约产生 n_meas * birth_weight 个潜在新目标）
    prune_thresh: 剪枝阈值（低于此权重的分量被删除）
    merge_dist  : 合并阈值（Mahalanobis 距离，低于此值的分量被合并）
    J_max       : 分量最大数量上限
    """

    def __init__(
        self,
        dt:            float = 1.0,
        sigma_q:       float = 0.5,
        sigma_r:       float = 50.0,
        P_d:           float = 0.95,
        P_s:           float = 0.99,
        lambda_c:      float = 10.0,
        birth_weight:  float = 0.5,
        prune_thresh:  float = 1e-3 ,
        merge_dist:    float = 4.0,
        J_max:         int   = 100,
    ):
        C = COORD_SCALE
        self.dt     = dt
        self.P_d    = P_d
        self.P_s    = P_s
        self.lambda_c   = lambda_c
        self.birth_weight = birth_weight
        self.prune_thresh = prune_thresh
        self.merge_dist   = merge_dist
        self.J_max        = J_max

        # 归一化噪声参数
        self.sigma_q_n = sigma_q / C   # 过程噪声（归一化）
        self.sigma_r_n = sigma_r / C   # 测量噪声（归一化）

        # ── 运动模型（CV，离散时间）──
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # ── 过程噪声矩阵 Q ──
        q  = self.sigma_q_n ** 2
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        self.Q = q * np.array([
            [dt4 / 4, 0,       0,       dt3 / 2, 0,       0      ],
            [0,       dt4 / 4, 0,       0,       dt3 / 2, 0      ],
            [0,       0,       dt4 / 4, 0,       0,       dt3 / 2],
            [dt3 / 2, 0,       0,       dt2,     0,       0      ],
            [0,       dt3 / 2, 0,       0,       dt2,     0      ],
            [0,       0,       dt3 / 2, 0,       0,       dt2    ],
        ])

        # ── 量测模型 H ──
        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)
        self.R = (self.sigma_r_n ** 2) * np.eye(3)

        # ── 新生分量初始协方差 ──
        v_init_n = 500.0 / C          # 速度不确定性（归一化）
        self.P_birth = np.diag(
            [self.sigma_r_n**2] * 3 + [v_init_n**2] * 3
        )

        # ── 杂波空间密度（均匀杂波，归一化空间体积 ≈ 4π/3 ≈ 4.19）──
        V_surv = (4.0 / 3.0) * np.pi   # normalized 单位中的近似体积
        self.kappa = lambda_c / V_surv

        # ── 分量集合：list of (weight, mean[6], cov[6×6]) ──
        self.components: list = []

    # ------------------------------------------------------------------
    # 基本工具
    # ------------------------------------------------------------------

    @staticmethod
    def _gauss_pdf(z: np.ndarray, nu: np.ndarray, S: np.ndarray) -> float:
        """多元高斯密度 N(z; nu, S)"""
        d = len(z)
        dz = z - nu
        try:
            L = np.linalg.cholesky(S + np.eye(d) * 1e-12)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            S_inv   = np.linalg.solve(S + np.eye(d) * 1e-12, np.eye(d))
        except np.linalg.LinAlgError:
            return 1e-300
        exponent = -0.5 * (dz @ S_inv @ dz)
        log_pdf  = -0.5 * (d * np.log(2 * np.pi) + log_det) + exponent
        return float(np.exp(np.clip(log_pdf, -700, 700)))

    # ------------------------------------------------------------------
    # 预测步
    # ------------------------------------------------------------------

    def predict(self):
        """PHD 预测：各分量权重乘以存活概率，均值/协方差按运动模型传播。"""
        new_components = []
        for w, m, P in self.components:
            w_pred = self.P_s * w
            m_pred = self.F @ m
            P_pred = self.F @ P @ self.F.T + self.Q
            new_components.append((w_pred, m_pred, P_pred))
        self.components = new_components

    # ------------------------------------------------------------------
    # 更新步（含新生分量）
    # ------------------------------------------------------------------

    def update(self, measurements: list, birth_weight: float = None):
        """
        PHD 更新：对每条测量分别更新预测分量的权重与状态，
        同时在每条测量处加入新生分量。

        Parameters
        ----------
        birth_weight : 覆盖实例的默认 birth_weight（None=使用默认值）
        """
        bw = self.birth_weight if birth_weight is None else birth_weight
        # ── 新生分量（基于量测的自适应新生）──
        birth = []
        for z in measurements:
            m_b = np.zeros(6)
            m_b[:3] = z
            birth.append((bw, m_b, self.P_birth.copy()))

        # 所有候选分量（预测 + 新生）
        all_comps = self.components + birth

        if not all_comps:
            self.components = []
            return

        # ── 漏检分量 ──
        missed = [(((1.0 - self.P_d) * w), m, P) for w, m, P in all_comps]

        if not measurements:
            self.components = missed
            return

        # ── 预计算各分量的 Kalman 增益等 ──
        innovations = []          # (nu, S, K, P_upd) for each component
        for _, m, P in all_comps:
            nu = self.H @ m
            S  = self.H @ P @ self.H.T + self.R
            try:
                S_inv = np.linalg.solve(S + np.eye(3) * 1e-12, np.eye(3))
                K     = P @ self.H.T @ S_inv
                P_upd = (np.eye(6) - K @ self.H) @ P
            except np.linalg.LinAlgError:
                K     = np.zeros((6, 3))
                P_upd = P.copy()
                S     = np.eye(3) * 1e10
            innovations.append((nu, S, K, P_upd))

        # ── 量测更新：对每条量测产生 n_comps 个更新分量 ──
        updated = list(missed)

        for z in measurements:
            # 各分量对该量测的似然项
            lhood_k = np.array([
                self.P_d * all_comps[k][0] * self._gauss_pdf(z, innovations[k][0], innovations[k][1])
                for k in range(len(all_comps))
            ])
            denom = self.kappa + lhood_k.sum()
            if denom < 1e-300:
                continue

            for k, (w, m, P) in enumerate(all_comps):
                w_upd = lhood_k[k] / denom
                if w_upd < self.prune_thresh * 0.01:
                    continue
                nu_k, S_k, K_k, P_upd_k = innovations[k]
                m_upd = m + K_k @ (z - nu_k)
                updated.append((w_upd, m_upd, P_upd_k))

        self.components = updated

    # ------------------------------------------------------------------
    # 剪枝 + 合并
    # ------------------------------------------------------------------

    def prune_and_merge(self):
        """删除低权重分量，合并相近分量，截断到 J_max。"""
        # 剪枝
        surviving = [(w, m, P) for w, m, P in self.components if w > self.prune_thresh]
        if not surviving:
            self.components = []
            return

        # 按权重降序排列
        surviving.sort(key=lambda x: -x[0])

        merged = []
        taken  = [False] * len(surviving)

        for j in range(len(surviving)):
            if taken[j]:
                continue
            if len(merged) >= self.J_max:
                break

            w_j, m_j, P_j = surviving[j]

            # Mahalanobis 距离 gate
            try:
                P_inv = np.linalg.solve(P_j + np.eye(6) * 1e-10, np.eye(6))
            except np.linalg.LinAlgError:
                P_inv = np.eye(6) * 1e-10

            merge_set = [j]
            for i in range(j + 1, len(surviving)):
                if taken[i]:
                    continue
                diff = surviving[i][1] - m_j
                mah2 = float(diff @ P_inv @ diff)
                if mah2 <= self.merge_dist ** 2:
                    merge_set.append(i)

            # 加权合并
            w_sum = sum(surviving[i][0] for i in merge_set)
            m_sum = sum(surviving[i][0] * surviving[i][1] for i in merge_set) / w_sum
            P_sum = sum(
                surviving[i][0] * (surviving[i][2] + np.outer(m_sum - surviving[i][1],
                                                               m_sum - surviving[i][1]))
                for i in merge_set
            ) / w_sum

            merged.append((w_sum, m_sum, P_sum))
            for i in merge_set:
                taken[i] = True

        self.components = merged

    # ------------------------------------------------------------------
    # 状态提取
    # ------------------------------------------------------------------

    def extract_states(self) -> list:
        """
        提取目标状态估计。
        对每个权重 >= 0.5 的分量，提取 round(w) 个目标。
        返回：list of np.ndarray [3]（归一化位置）
        """
        estimates = []
        for w, m, P in self.components:
            n = int(round(float(w)))
            for _ in range(n):
                estimates.append(m[:3].copy())
        return estimates

    def reset(self):
        self.components = []


# ====================================================================
# 场景评估入口（与 evaluate_3d.evaluate_scenario 格式一致）
# ====================================================================

def evaluate_scenario_phd(
    scenario,
    sigma_q:      float = 0.5,
    sigma_r:      float = 50.0,
    P_d:          float = 0.95,
    P_s:          float = 0.99,
    lambda_c:     float = 10.0,
    birth_weight: float = 0.01,
    prune_thresh: float = 1e-4,
    merge_dist:   float = 1.0,
    tau:          int   = 4,
    use_gt_init:  bool  = False,
) -> dict:
    """
    使用 GM-PHD 滤波器评估单个场景。

    返回 dict 格式与 evaluate_3d.evaluate_scenario() 完全一致，
    可直接传入 plot_3d_trajectories / plot_error_curves / plot_ospa_curve。

    说明
    ----
    - PHD 不维护轨迹标识，通过帧级匈牙利匹配将估计位置对齐到真实轨迹。
    - frame_assoc_acc：对每条真实目标量测，检查是否有 PHD 估计落在 3σ 门限内。
    - tau 前各帧以真实轨迹状态填充 tracked_states（与 BAIT 一致）。
    - use_gt_init=True：前 tau 帧只向滤波器喂真实目标量测（过滤杂波），
      与 BAIT 的真值热启动等价，可显著提升起始帧的跟踪质量。
    """
    scenario = _copy.deepcopy(scenario)
    trajectories, measurements, gt_associations = scenario
    num_frames = len(measurements)

    # ── 坐标归一化（与 evaluate_3d.evaluate_scenario 一致）──
    for traj in trajectories:
        traj['states'][:, :3] /= COORD_SCALE
    for fm in measurements:
        if len(fm) > 0:
            fm[:] = fm / COORD_SCALE

    init_mode = "真值热启动" if use_gt_init else "纯量测自主"
    print(f"[PHD] 目标数: {len(trajectories)}  帧数: {num_frames}  起始模式: {init_mode}")
    print(f"[PHD] sigma_r={sigma_r}m  sigma_q={sigma_q}  P_d={P_d}  lambda_c={lambda_c}")

    # ── 创建滤波器 ──
    phd = GMPHDFilter(
        dt=1.0, sigma_q=sigma_q, sigma_r=sigma_r,
        P_d=P_d, P_s=P_s, lambda_c=lambda_c,
        birth_weight=birth_weight, prune_thresh=prune_thresh,
        merge_dist=merge_dist,
    )

    # ── 逐帧运行 PHD ──
    frame_estimates = []     # list[list[ndarray[3]]]，每帧 PHD 提取的位置列表
    for t in range(num_frames):
        phd.predict()

        # 真值热启动：前 tau 帧只喂真实目标量测，过滤掉杂波
        if use_gt_init and t < tau:
            gt_assoc = gt_associations[t]
            n = len(measurements[t])
            z_list = [
                measurements[t][i] for i in range(n)
                if i < len(gt_assoc) and gt_assoc[i] > 0
            ]
        else:
            z_list = [measurements[t][i] for i in range(len(measurements[t]))]

        # 真值热启动：tau 帧后目标已由真值注入，禁用新生分量避免杂波干扰
        # 自主模式：始终使用默认 birth_weight
        effective_bw = 0.0 if (use_gt_init and t >= tau) else None
        phd.update(z_list, birth_weight=effective_bw)
        phd.prune_and_merge()

        # 真值热启动：在第 tau-1 帧结束后，用真实状态重置滤波器
        # （消除低权重累积问题，直接以 w=1.0 的高置信分量热启动）
        if use_gt_init and t == tau - 1:
            phd.components = []          # 丢弃之前所有低权重分量
            v_init_n = 500.0 / COORD_SCALE
            for traj in trajectories:
                if traj['birth_frame'] <= t <= traj['death_frame']:
                    pos_now = traj['states'][t, :3]
                    if t > 0 and traj['birth_frame'] < t:
                        vel_est = (traj['states'][t, :3] - traj['states'][t - 1, :3]) / 1.0
                    else:
                        vel_est = np.zeros(3)
                    m_gt = np.concatenate([pos_now, vel_est])
                    P_gt = np.diag([phd.sigma_r_n**2]*3 + [v_init_n**2]*3)
                    phd.components.append((1.0, m_gt, P_gt))

        ests = phd.extract_states()
        frame_estimates.append(ests)
        n_comp = len(phd.components)
        print(f"  帧{t:3d} | 量测={len(z_list):2d}  估计目标数={len(ests):2d}  分量数={n_comp:3d}")

    # ── 初始化 tracked_states（前 tau 帧用真实轨迹状态填充）──
    tracked_states = {traj['label']: [] for traj in trajectories}
    for traj in trajectories:
        lbl = traj['label']
        for t in range(tau):
            if traj['birth_frame'] <= t <= traj['death_frame']:
                tracked_states[lbl].append(traj['states'][t, :3].copy())
            else:
                tracked_states[lbl].append(np.zeros(3))

    sigma_r_n = sigma_r / COORD_SCALE   # 归一化测量噪声，用于门限

    frame_pred_xyz   = []
    frame_true_xyz   = []
    frame_assoc_acc  = []
    frame_assoc_detail = []

    for frame_idx in range(tau, num_frames):
        # 该帧活跃真实轨迹（按 trajectories 顺序）
        active_trajs = [tr for tr in trajectories
                        if tr['birth_frame'] <= frame_idx <= tr['death_frame']]
        true_xyz = np.array([tr['states'][frame_idx, :3] for tr in active_trajs]) \
                   if active_trajs else np.empty((0, 3))

        ests = frame_estimates[frame_idx]

        # ── 将 PHD 估计匹配到真实轨迹（匈牙利）──
        if len(ests) > 0 and len(active_trajs) > 0:
            est_arr = np.array(ests)          # [n_est, 3]

            cost = np.linalg.norm(
                est_arr[:, None, :] - true_xyz[None, :, :], axis=2
            )                                 # [n_est, n_true]
            row_ind, col_ind = linear_sum_assignment(cost)

            aligned_pred = np.full_like(true_xyz, np.nan)   # 未匹配目标显示 NaN（断点）
            for r, c in zip(row_ind, col_ind):
                aligned_pred[c] = est_arr[r]
        elif len(active_trajs) > 0:
            aligned_pred = np.full_like(true_xyz, np.nan)   # 无估计：NaN 让曲线断开
        else:
            aligned_pred = np.empty((0, 3))

        frame_pred_xyz.append(aligned_pred)
        frame_true_xyz.append(true_xyz)

        # 更新 tracked_states
        for j, tr in enumerate(active_trajs):
            tracked_states[tr['label']].append(aligned_pred[j].copy())
        for tr in trajectories:
            if tr not in active_trajs:
                tracked_states[tr['label']].append(np.zeros(3))

        # ── 关联正确率（近似：量测落在 3σ 门限内视为正确检测）──
        gt_assoc_frame = gt_associations[frame_idx]
        meas_frame     = measurements[frame_idx]
        n_meas         = len(meas_frame)
        acc            = 0.0
        detail_rows    = []

        valid_idx = np.where(gt_assoc_frame[:n_meas] > 0)[0] if n_meas > 0 else np.array([])

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
