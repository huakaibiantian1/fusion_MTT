"""
mht_tracker.py — N-scan 多假设跟踪器（Multiple Hypothesis Tracker）

算法简介
--------
- 维护若干"全局假设"，每个假设是一组互斥的「轨迹-量测」关联对
  及其累积对数似然权重。
- 每帧用 Murty K-best 全局分配算法扩展假设树，限制 top-K 条存活。
- N-scan 向后剪枝：所有 top-K 假设在 N 步前都同意的关联记录将被提交，
  相关分支合并。
- 轨迹管理：未关联量测产生候选轨迹，连续确认 confirm_frames 次后升级为
  确认轨迹；连续 delete_frames 次漏检后删除。

坐标约定与 phd_filter.py 相同：内部全部在归一化坐标系下运算。
"""

import copy as _copy
import heapq
import numpy as np
from scipy.optimize import linear_sum_assignment

from data_generation import COORD_SCALE


# ====================================================================
# 辅助：Murty K-best 全局分配
# ====================================================================

def _murty_k_best(cost: np.ndarray, K: int):
    """
    使用 Murty 算法求 n×m 代价矩阵的 K-best 最小代价分配。

    参数
    ----
    cost : (n, m)  代价矩阵（行=轨迹，列=量测 + 虚拟列）
    K    : 保留的最优分配数量

    返回
    ----
    list of (total_cost, assignment)
        assignment : ndarray[n]，assignment[i] = j 表示行 i 分配到列 j，
                     j >= m 表示"漏检"（分配到虚拟列）
    """
    n, m = cost.shape
    if n == 0:
        return [(0.0, np.array([], dtype=int))]

    # 扩展代价矩阵：在右侧追加 n 列虚拟"漏检"列（代价 = miss_cost_per_row）
    # 每行的漏检代价已经编码在传入的 cost 中（最后 n 列）
    # 直接求 (n × (m+n)) 矩阵上的 K-best

    results = []

    # 优先队列：(total_cost, constraint_list, excluded_set)
    # constraint_list：list of (row, col) 必须分配
    # excluded_set   ：set of (row, col) 不允许分配
    INF = 1e18

    def _solve_constrained(forced: list, excluded: set):
        """
        在 forced 和 excluded 约束下求最优分配。
        返回 (total_cost, assignment) 或 None（无可行解）。
        """
        c = cost.copy().astype(float)
        # 强制分配：令其他同行/同列代价设为 INF
        for (r, col) in forced:
            c[r, :] = INF
            c[:, col] = INF
            c[r, col] = 0.0
        # 排除分配
        for (r, col) in excluded:
            c[r, col] = INF
        row_idx, col_idx = linear_sum_assignment(c)
        total = c[row_idx, col_idx].sum()
        if total >= INF * 0.9:
            return None
        asgn = np.full(n, -1, dtype=int)
        asgn[row_idx] = col_idx
        return total, asgn

    base = _solve_constrained([], set())
    if base is None:
        return [(0.0, np.arange(n, dtype=int) + m)]   # 全部漏检

    heap = []
    counter = [0]

    def push(cost_val, forced, excluded):
        heapq.heappush(heap, (cost_val, counter[0], forced, excluded))
        counter[0] += 1

    base_cost, base_asgn = base
    push(base_cost, [], set())
    seen = set()

    while heap and len(results) < K:
        total_c, _, forced, excluded = heapq.heappop(heap)
        sol = _solve_constrained(forced, excluded)
        if sol is None:
            continue
        total_c, asgn = sol
        key = tuple(asgn.tolist())
        if key in seen:
            continue
        seen.add(key)
        results.append((total_c, asgn.copy()))

        # Murty 分裂：固定前 k 行，第 k 行排除当前分配列
        for k in range(n):
            if asgn[k] < 0:
                continue
            new_forced   = list(forced) + [(r, asgn[r]) for r in range(k)]
            new_excluded = set(excluded) | {(k, asgn[k])}
            sol2 = _solve_constrained(new_forced, new_excluded)
            if sol2 is not None:
                push(sol2[0], new_forced, new_excluded)

    return results if results else [(0.0, np.arange(n, dtype=int) + m)]


# ====================================================================
# 单条轨迹（Kalman 滤波器状态）
# ====================================================================

class _Track:
    """内部轨迹对象，维护 Kalman 滤波器状态。"""

    _id_counter = 0

    def __init__(self, state: np.ndarray, P: np.ndarray):
        _Track._id_counter += 1
        self.id      = _Track._id_counter
        self.state   = state.copy()    # [6] 归一化
        self.P       = P.copy()        # [6×6]
        self.hits    = 1               # 连续命中次数
        self.misses  = 0               # 连续漏检次数
        self.age     = 0               # 总帧数
        self.confirmed = False


# ====================================================================
# MHT 主体
# ====================================================================

class MHTTracker:
    """
    N-scan 多假设跟踪器。

    参数
    ----
    dt              : 采样间隔（s）
    sigma_q         : 过程噪声谱密度（m/s^{3/2}，同 GM-PHD）
    sigma_r         : 测量噪声标准差（m）
    P_d             : 检测概率
    P_s             : 存活概率
    lambda_c        : 每帧杂波期望数
    gate_sigma      : 门限（Mahalanobis 距离的 sigma 倍数，通常 3~5）
    n_scan          : N-scan 回溯剪枝深度
    max_hypotheses  : 保留的全局假设数 K
    confirm_frames  : 候选轨迹升级为确认轨迹所需连续命中帧数
    delete_frames   : 确认轨迹连续漏检超此帧数则删除
    """

    def __init__(
        self,
        dt:             float = 1.0,
        sigma_q:        float = 0.5,
        sigma_r:        float = 50.0,
        P_d:            float = 0.95,
        P_s:            float = 0.99,
        lambda_c:       float = 10.0,
        gate_sigma:     float = 4.0,
        n_scan:         int   = 2,
        max_hypotheses: int   = 20,
        confirm_frames: int   = 2,
        delete_frames:  int   = 3,
    ):
        _Track._id_counter = 0          # 每次实例化重置 ID

        C = COORD_SCALE
        self.dt             = dt
        self.P_d            = P_d
        self.P_s            = P_s
        self.lambda_c       = lambda_c
        self.gate_sigma     = gate_sigma
        self.n_scan         = n_scan
        self.max_hypotheses = max_hypotheses
        self.confirm_frames = confirm_frames
        self.delete_frames  = delete_frames

        self.sigma_q_n = sigma_q / C
        self.sigma_r_n = sigma_r / C

        # 运动模型
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        q  = self.sigma_q_n ** 2
        dt2 = dt ** 2; dt3 = dt ** 3; dt4 = dt ** 4
        self.Q = q * np.array([
            [dt4/4, 0,     0,     dt3/2, 0,     0    ],
            [0,     dt4/4, 0,     0,     dt3/2, 0    ],
            [0,     0,     dt4/4, 0,     0,     dt3/2],
            [dt3/2, 0,     0,     dt2,   0,     0    ],
            [0,     dt3/2, 0,     0,     dt2,   0    ],
            [0,     0,     dt3/2, 0,     0,     dt2  ],
        ])

        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)
        self.R = (self.sigma_r_n ** 2) * np.eye(3)

        v_init_n     = 500.0 / C
        self.P_birth = np.diag([self.sigma_r_n**2]*3 + [v_init_n**2]*3)

        # ── 假设集：list of {'log_w': float, 'tracks': dict{id: _Track},
        #                      'assignment': list[(frame, tid, meas_idx)]}
        self.hypotheses: list = [{'log_w': 0.0, 'tracks': {}, 'assignment': []}]
        self.frame_idx: int   = 0

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _gauss_pdf(z, nu, S):
        d  = len(z)
        dz = z - nu
        try:
            S_inv = np.linalg.solve(S + np.eye(d) * 1e-12, np.eye(d))
            sign, logdet = np.linalg.slogdet(S + np.eye(d) * 1e-12)
        except np.linalg.LinAlgError:
            return 1e-300
        if sign <= 0:
            return 1e-300
        log_p = -0.5 * (d * np.log(2 * np.pi) + logdet + dz @ S_inv @ dz)
        return float(np.exp(np.clip(log_p, -700, 700)))

    def _kalman_predict(self, track: _Track):
        m_pred = self.F @ track.state
        P_pred = self.F @ track.P @ self.F.T + self.Q
        t_new  = _copy.copy(track)
        t_new.state = m_pred
        t_new.P     = P_pred
        t_new.age  += 1
        t_new.misses = 0   # reset before update step
        return t_new

    def _kalman_update(self, track: _Track, z: np.ndarray):
        nu  = self.H @ track.state
        S   = self.H @ track.P @ self.H.T + self.R
        try:
            S_inv = np.linalg.solve(S + np.eye(3) * 1e-12, np.eye(3))
        except np.linalg.LinAlgError:
            S_inv = np.eye(3)
        K     = track.P @ self.H.T @ S_inv
        P_upd = (np.eye(6) - K @ self.H) @ track.P
        m_upd = track.state + K @ (z - nu)
        t_new  = _copy.copy(track)
        t_new.state = m_upd
        t_new.P     = P_upd
        return t_new

    def _log_likelihood(self, track: _Track, z: np.ndarray) -> float:
        """测量 z 归属到 track 的对数似然。"""
        nu = self.H @ track.state
        S  = self.H @ track.P @ self.H.T + self.R
        return float(np.log(max(self._gauss_pdf(z, nu, S), 1e-300)))

    def _gate(self, track: _Track, z: np.ndarray) -> bool:
        """判断测量 z 是否在轨迹 track 的门限内。"""
        nu  = self.H @ track.state
        S   = self.H @ track.P @ self.H.T + self.R
        try:
            S_inv = np.linalg.solve(S + np.eye(3) * 1e-12, np.eye(3))
        except np.linalg.LinAlgError:
            return False
        dz   = z - nu
        mah2 = float(dz @ S_inv @ dz)
        return mah2 <= self.gate_sigma ** 2

    # ------------------------------------------------------------------
    # 单帧处理
    # ------------------------------------------------------------------

    def step(self, measurements: list) -> dict:
        """
        处理一帧量测，更新假设集合。

        返回：best hypothesis 中各确认轨迹的状态
            {track_id: state[6]}
        """
        t = self.frame_idx
        n_meas = len(measurements)

        new_hypotheses = []

        for hyp in self.hypotheses:
            tracks = hyp['tracks']  # {id: _Track}

            # 1. 预测所有轨迹
            pred_tracks = {tid: self._kalman_predict(tr) for tid, tr in tracks.items()}
            track_ids   = list(pred_tracks.keys())
            n_tracks    = len(track_ids)

            # 2. 构建代价矩阵（行=轨迹，前 n_meas 列=量测，后 n_tracks 列=漏检）
            #    代价 = -对数似然
            miss_cost = float(-np.log(max(1.0 - self.P_d, 1e-300)))  # per track
            clutter_density = self.lambda_c / max(n_meas, 1)
            clutter_log     = float(np.log(max(clutter_density, 1e-300)))

            if n_tracks > 0 and n_meas > 0:
                cost = np.full((n_tracks, n_meas + n_tracks), miss_cost)
                for i, tid in enumerate(track_ids):
                    tr = pred_tracks[tid]
                    for j, z in enumerate(measurements):
                        if self._gate(tr, z):
                            ll = self._log_likelihood(tr, z) + np.log(max(self.P_d, 1e-300)) - clutter_log
                            cost[i, j] = -ll
                    cost[i, n_meas + i] = miss_cost  # 漏检虚拟列

                k_best = _murty_k_best(cost, self.max_hypotheses)
            elif n_tracks > 0:
                # 无量测：全部漏检
                asgn = np.arange(n_tracks, dtype=int) + n_meas
                k_best = [(miss_cost * n_tracks, asgn)]
            else:
                k_best = [(0.0, np.array([], dtype=int))]

            for delta_cost, asgn in k_best:
                new_tracks    = {}
                assignment_rec = list(hyp['assignment'])  # deep copy history

                # 已关联量测集合（避免同一量测被多轨迹使用）
                used_meas = set()
                valid = True

                for i, tid in enumerate(track_ids):
                    tr  = _copy.copy(pred_tracks[tid])
                    col = int(asgn[i]) if i < len(asgn) else (n_meas + i)

                    if col < n_meas:      # 关联到真实量测
                        if col in used_meas:
                            valid = False
                            break
                        used_meas.add(col)
                        tr = self._kalman_update(tr, measurements[col])
                        tr.hits   += 1
                        tr.misses  = 0
                        if tr.hits >= self.confirm_frames:
                            tr.confirmed = True
                        assignment_rec.append((t, tid, col))
                    else:                  # 漏检
                        tr.misses += 1
                        assignment_rec.append((t, tid, None))

                    new_tracks[tid] = tr

                if not valid:
                    continue

                # 3. 为未关联量测创建新候选轨迹
                for j, z in enumerate(measurements):
                    if j not in used_meas:
                        m0    = np.zeros(6)
                        m0[:3] = z
                        new_tr = _Track(m0, self.P_birth.copy())
                        assignment_rec.append((t, new_tr.id, j))
                        new_tracks[new_tr.id] = new_tr

                # 4. 删除连续漏检过多的轨迹
                new_tracks = {
                    tid: tr for tid, tr in new_tracks.items()
                    if tr.misses <= self.delete_frames
                }

                log_w = hyp['log_w'] - delta_cost
                new_hypotheses.append({
                    'log_w':      log_w,
                    'tracks':     new_tracks,
                    'assignment': assignment_rec,
                })

        # 5. 归一化权重，保留 top-K
        if not new_hypotheses:
            self.hypotheses = [{'log_w': 0.0, 'tracks': {}, 'assignment': []}]
        else:
            max_log = max(h['log_w'] for h in new_hypotheses)
            new_hypotheses.sort(key=lambda h: -h['log_w'])
            self.hypotheses = new_hypotheses[:self.max_hypotheses]

        # 6. N-scan 剪枝：裁剪历史记录（只保留最近 n_scan 帧）
        cutoff = t - self.n_scan + 1
        for hyp in self.hypotheses:
            hyp['assignment'] = [rec for rec in hyp['assignment'] if rec[0] >= cutoff]

        self.frame_idx += 1

        # 返回最优假设中确认轨迹的当前状态
        best = self.hypotheses[0]
        return {tid: tr.state for tid, tr in best['tracks'].items() if tr.confirmed}

    def get_all_track_states(self) -> dict:
        """返回最优假设中所有轨迹（含候选）的状态。"""
        best = self.hypotheses[0]
        return {tid: tr.state.copy() for tid, tr in best['tracks'].items()}


# ====================================================================
# 场景评估入口（与 evaluate_3d.evaluate_scenario 格式一致）
# ====================================================================

def evaluate_scenario_mht(
    scenario,
    sigma_q:        float = 0.5,
    sigma_r:        float = 50.0,
    P_d:            float = 0.95,
    P_s:            float = 0.99,
    lambda_c:       float = 10.0,
    gate_sigma:     float = 4.0,
    n_scan:         int   = 2,
    max_hypotheses: int   = 20,
    confirm_frames: int   = 2,
    delete_frames:  int   = 3,
    tau:            int   = 4,
    use_gt_init:    bool  = False,
) -> dict:
    """
    使用 N-scan MHT 评估单个场景。

    返回 dict 格式与 evaluate_3d.evaluate_scenario() 完全一致。

    说明
    ----
    - MHT 维护显式轨迹 ID；评估结束后用总体位置相似度匹配到真实轨迹。
    - frame_assoc_acc：该帧中被 MHT 最优假设正确关联的真实目标量测比例。
    - tau 前各帧以真实轨迹状态填充 tracked_states。
    - use_gt_init=True：前 tau 帧只向跟踪器喂真实目标量测（过滤杂波），
      与 BAIT 的真值热启动等价，can 大幅降低起始阶段的虚假轨迹数量。
    """
    scenario = _copy.deepcopy(scenario)
    trajectories, measurements, gt_associations = scenario
    num_frames = len(measurements)

    # ── 归一化 ──
    for traj in trajectories:
        traj['states'][:, :3] /= COORD_SCALE
    for fm in measurements:
        if len(fm) > 0:
            fm[:] = fm / COORD_SCALE

    init_mode = "真值热启动" if use_gt_init else "纯量测自主"
    print(f"[MHT] 目标数: {len(trajectories)}  帧数: {num_frames}  起始模式: {init_mode}")
    print(f"[MHT] sigma_r={sigma_r}m  P_d={P_d}  gate_sigma={gate_sigma}  K={max_hypotheses}")

    tracker = MHTTracker(
        dt=1.0,
        sigma_q=sigma_q, sigma_r=sigma_r,
        P_d=P_d, P_s=P_s, lambda_c=lambda_c,
        gate_sigma=gate_sigma, n_scan=n_scan,
        max_hypotheses=max_hypotheses,
        confirm_frames=confirm_frames,
        delete_frames=delete_frames,
    )

    # ── 逐帧运行 ──
    # 记录每帧最优假设中各轨迹的状态以及量测关联
    frame_mht_states: list = []    # list[dict{tid: state[6]}]（包含候选轨迹）
    frame_mht_assign: list = []    # list[dict{tid: meas_idx or None}]

    for t in range(num_frames):
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
        confirmed = tracker.step(z_list)
        all_states = tracker.get_all_track_states()
        frame_mht_states.append(all_states)

        # 当前帧的量测关联（来自最优假设最新一步记录）
        best  = tracker.hypotheses[0]
        curr_assign = {}
        for (frm, tid, midx) in best['assignment']:
            if frm == t:
                curr_assign[tid] = midx
        frame_mht_assign.append(curr_assign)

        n_confirmed = len(confirmed)
        print(f"  帧{t:3d} | 量测={len(z_list):2d}  确认轨迹={n_confirmed:2d}  "
              f"总轨迹={len(all_states):2d}")

    # ── 初始化 tracked_states（前 tau 帧填真值）──
    tracked_states = {traj['label']: [] for traj in trajectories}
    for traj in trajectories:
        lbl = traj['label']
        for t in range(tau):
            if traj['birth_frame'] <= t <= traj['death_frame']:
                tracked_states[lbl].append(traj['states'][t, :3].copy())
            else:
                tracked_states[lbl].append(np.zeros(3))

    # ── 全局匹配：将 MHT 轨迹 ID 映射到真实轨迹标签 ──
    # 策略：对 tau 后所有帧，统计每个 (tid, true_label) 对的平均距离，
    # 用匈牙利算法找最优映射
    all_tids = set()
    for states in frame_mht_states[tau:]:
        all_tids.update(states.keys())
    all_tids   = sorted(all_tids)
    true_labels = [tr['label'] for tr in trajectories]

    if all_tids and true_labels:
        dist_accum = np.full((len(all_tids), len(true_labels)), 1e9)
        count_mat  = np.zeros_like(dist_accum)

        for fi in range(tau, num_frames):
            states = frame_mht_states[fi]
            for i, tid in enumerate(all_tids):
                if tid not in states:
                    continue
                est = states[tid][:3]
                for j, tr in enumerate(trajectories):
                    if tr['birth_frame'] <= fi <= tr['death_frame']:
                        d = np.linalg.norm(est - tr['states'][fi, :3])
                        if dist_accum[i, j] == 1e9:
                            dist_accum[i, j] = d
                            count_mat[i, j]  = 1
                        else:
                            dist_accum[i, j] += d
                            count_mat[i, j]  += 1

        avg_dist = np.where(count_mat > 0, dist_accum / (count_mat + 1e-9), 1e9)
        row_ind, col_ind = linear_sum_assignment(avg_dist)
        tid_to_label = {}
        for r, c in zip(row_ind, col_ind):
            if avg_dist[r, c] < 1e8:
                tid_to_label[all_tids[r]] = true_labels[c]
    else:
        tid_to_label = {}

    # ── 构建 frame_pred_xyz / frame_true_xyz 及更新 tracked_states ──
    frame_pred_xyz   = []
    frame_true_xyz   = []
    frame_assoc_acc  = []
    frame_assoc_detail = []

    for frame_idx in range(tau, num_frames):
        active_trajs = [tr for tr in trajectories
                        if tr['birth_frame'] <= frame_idx <= tr['death_frame']]
        true_xyz = np.array([tr['states'][frame_idx, :3] for tr in active_trajs]) \
                   if active_trajs else np.empty((0, 3))

        states = frame_mht_states[frame_idx]

        # 以匹配关系填充预测位置，无对应轨迹的目标用 NaN（让曲线断开）
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

        # ── 关联正确率 ──
        gt_assoc_frame = gt_associations[frame_idx]
        meas_frame     = measurements[frame_idx]
        n_meas         = len(meas_frame)
        curr_assign    = frame_mht_assign[frame_idx]   # {tid: meas_idx}

        # 反转：meas_idx -> tid
        meas_to_tid = {v: k for k, v in curr_assign.items() if v is not None}

        acc         = 0.0
        detail_rows = []
        valid_idx   = np.where(gt_assoc_frame[:n_meas] > 0)[0] if n_meas > 0 else np.array([])

        if len(valid_idx) > 0:
            correct = 0
            for i in valid_idx:
                gt_lbl = int(gt_assoc_frame[i])
                tid    = meas_to_tid.get(int(i), None)
                if tid is not None:
                    mapped_lbl = tid_to_label.get(tid, -1)
                    matched    = (mapped_lbl == gt_lbl)
                else:
                    matched = False
                if matched:
                    correct += 1
                detail_rows.append({
                    'meas_idx':   int(i),
                    'pos_real':   (meas_frame[i] * COORD_SCALE).tolist(),
                    'pred_label': tid_to_label.get(meas_to_tid.get(int(i), -1), -1),
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
