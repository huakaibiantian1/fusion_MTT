"""
多场景训练数据生成模块

支持 4 种场景类型：
  1. crossing      - 星形交叉：3~10 条轨迹在同一空间点同时汇聚，经过后各自机动散开
  2. many_targets  - 大量目标场景（CV 模型）
  3. high_maneuver - 高机动目标（CA 模型，3 段随机加速度，每段 T/3 帧）
  4. spindle       - 纺锤形（两目标从远处相向接近 → 贴近 → 分离）
                     其中 50% 为「交叉纺锤」：平行段中间两目标互换左右，中点处真实交叉

各类型等量生成后混合，同时按类型分别保存测试集 pkl，供 evaluate_3d.py 分类验证。
"""

import copy
import numpy as np
from scipy.stats import poisson
from torch.utils.data import DataLoader

from data_generation import (
    MTTDataGenerator, MTTDataset,
    spherical_to_cartesian, cartesian_to_spherical,
    COORD_SCALE,
)
from data_generation_with_crossing import MTTDataGeneratorWithCrossing

# 场景类型名称列表（顺序即生成顺序）
SCENARIO_TYPES = ['crossing', 'many_targets', 'high_maneuver', 'spindle']


# ============================================================
# 多场景生成器
# ============================================================

class MTTDataGeneratorMultiScenario(MTTDataGeneratorWithCrossing):
    """
    4 种场景类型生成器

    继承 MTTDataGeneratorWithCrossing，复用其交叉场景逻辑；
    新增 many_targets / high_maneuver / spindle 三种场景。
    """

    def __init__(
        self,
        high_maneuver_accel_range=(5, 15),   # m/s²，每段加速度幅度
        many_targets_range=(3, 5),            # 多目标场景的目标数范围
        sep_far_range=(3000, 7000),           # 纺锤形最远横向间距范围 (m)
        sep_near_range=(200, 600),            # 纺锤形最近横向间距范围 (m)
        n_cross_traj=None,                    # 星形交叉固定轨迹数（None=随机3~15）
        spindle_crossing=None,                # 纺锤是否交叉：True/False/None(随机50%)
        **kwargs
    ):
        super().__init__(**kwargs)
        self.high_maneuver_accel_range = high_maneuver_accel_range
        self.many_targets_range        = many_targets_range
        self.sep_far_range             = sep_far_range
        self.sep_near_range            = sep_near_range
        self.n_cross_traj              = n_cross_traj
        self.spindle_crossing          = spindle_crossing

    # ----------------------------------------------------------
    # 统一入口
    # ----------------------------------------------------------

    def generate_by_type(self, scenario_type: str):
        if scenario_type == 'crossing':
            scenario = self._generate_crossing_scenario()
        elif scenario_type == 'many_targets':
            scenario = self._generate_many_targets_scenario()
        elif scenario_type == 'high_maneuver':
            scenario = self._generate_high_maneuver_scenario()
        elif scenario_type == 'spindle':
            scenario = self._generate_spindle_scenario()
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type!r}")

        trajectories, _, _ = scenario
        trajectories = self._randomize_trajectory_lifetimes(trajectories)
        if not trajectories:
            fallback = self._generate_single_trajectory(1)
            trajectories = self._randomize_trajectory_lifetimes([fallback]) if fallback else []
        meas, assoc = self._make_measurements(trajectories)
        return trajectories, meas, assoc

    def _make_measurements(self, trajectories):
        """统一的测量生成逻辑（球坐标加噪→xyz，含杂波）"""
        measurements, associations = [], []
        for t in range(self.num_frames):
            fm, fa = [], []
            for traj in trajectories:
                if traj['birth_frame'] <= t <= traj['death_frame']:
                    if np.random.rand() < self.P_d:
                        fm.append(self._measure_3d(traj['states'][t, :3]))
                        fa.append(traj['label'])
            for _ in range(poisson.rvs(self.lambda_c)):
                fm.append(self._generate_clutter())
                fa.append(0)
            if fm:
                idx = np.random.permutation(len(fm))
                fm  = [fm[i] for i in idx]
                fa  = [fa[i] for i in idx]
            measurements.append(np.array(fm)  if fm  else np.empty((0, 3)))
            associations.append(np.array(fa)  if fa  else np.empty(0))
        return measurements, associations

    # ----------------------------------------------------------
    # 场景 1: 多目标（many_targets）
    # ----------------------------------------------------------

    def _generate_many_targets_scenario(self):
        """大量目标场景：8~15 个 CV 目标同时存在"""
        n = np.random.randint(self.many_targets_range[0],
                              self.many_targets_range[1] + 1)
        trajectories = []
        for i in range(n):
            traj = self._generate_single_trajectory(i + 1)
            if traj is not None:
                trajectories.append(traj)
        if not trajectories:
            traj = self._generate_single_trajectory(1)
            if traj:
                trajectories = [traj]
        meas, assoc = self._make_measurements(trajectories)
        return trajectories, meas, assoc

    # ----------------------------------------------------------
    # 场景 2: 高机动（high_maneuver，CA 模型）
    # ----------------------------------------------------------

    def _generate_high_maneuver_scenario(self):
        """高机动场景：每个目标使用 3 段 CA 模型（每段随机加速方向）"""
        n = max(1, poisson.rvs(self.lambda_0))
        trajectories = []
        for i in range(n):
            traj = self._generate_high_maneuver_trajectory(i + 1)
            if traj is not None:
                trajectories.append(traj)
        if not trajectories:
            traj = self._generate_single_trajectory(1)
            if traj:
                trajectories = [traj]
        meas, assoc = self._make_measurements(trajectories)
        return trajectories, meas, assoc

    def _generate_high_maneuver_trajectory(self, label):
        """
        单条高机动轨迹（3 段 CA）：
          - 共 num_frames 帧，分成 3 段，每段独立加速度
          - 加速度方向偏向视野中心（向心分量），避免快速飞出视野
          - 最多尝试 50 次；失败则回退到普通 CV 轨迹（保证不卡死）
        """
        dt      = self.dt
        n_seg   = 3
        seg_len = self.num_frames // n_seg
        Q       = self._cv_Q(dt)
        r_mid   = (self.r_min + self.r_max) / 2   # 视野中间距离，用于向心约束

        for _ in range(50):
            # 初始位置（视野中间区域，给机动留余量）
            r0  = np.random.uniform(self.r_min * 1.4, self.r_max * 0.6)
            a0  = np.random.uniform(-self.alpha_max * 0.5, self.alpha_max * 0.5)
            b0  = np.random.uniform(0, 2 * np.pi)
            xyz0 = np.array(spherical_to_cartesian(r0, a0, b0))

            # 初始速度（中等偏低，给加速留空间）
            v_mag   = np.random.uniform(80, 200)
            phi_v   = np.random.uniform(np.pi * 0.35, np.pi * 0.65)
            theta_v = np.random.uniform(0, 2 * np.pi)
            v0 = v_mag * np.array([
                np.sin(phi_v) * np.cos(theta_v),
                np.sin(phi_v) * np.sin(theta_v),
                np.cos(phi_v),
            ])

            states    = np.zeros((self.num_frames, 6))
            states[0] = np.concatenate([xyz0, v0])

            valid = True
            for seg in range(n_seg):
                # 加速度 = 随机横向分量（机动）+ 向心分量（防飞出）
                a_mag    = np.random.uniform(*self.high_maneuver_accel_range)
                phi_a    = np.random.uniform(0, np.pi)
                theta_a  = np.random.uniform(0, 2 * np.pi)
                rand_dir = np.array([
                    np.sin(phi_a) * np.cos(theta_a),
                    np.sin(phi_a) * np.sin(theta_a),
                    np.cos(phi_a),
                ])

                t_start = seg * seg_len + 1
                t_end   = (seg + 1) * seg_len if seg < n_seg - 1 else self.num_frames

                for t in range(t_start, t_end):
                    prev = states[t - 1]
                    pos  = prev[:3]

                    # 向心修正：若当前偏离视野中间距离，加一个拉回分量
                    r_cur = np.linalg.norm(pos) + 1e-8
                    centripetal_dir = -pos / r_cur          # 指向原点方向
                    overshoot = (r_cur - r_mid) / r_mid     # 正：偏远，负：偏近
                    centripetal = centripetal_dir * overshoot * a_mag * 0.5

                    accel = rand_dir * a_mag + centripetal

                    new_pos = pos + prev[3:] * dt + 0.5 * accel * dt ** 2
                    new_vel = prev[3:] + accel * dt
                    noise   = np.random.multivariate_normal(np.zeros(6), Q)
                    states[t] = np.concatenate([new_pos, new_vel]) + noise

                    if not self._in_fov(states[t, :3]):
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                return {
                    'label': label, 'states': states,
                    'birth_frame': 0, 'death_frame': self.num_frames - 1,
                }

        # 50 次仍失败 → 回退到普通 CV 轨迹，保证不卡死
        return self._generate_single_trajectory(label)

    # ----------------------------------------------------------
    # 场景 1（覆盖父类）: 星形多轨迹交叉（crossing）
    # ----------------------------------------------------------

    def _generate_crossing_scenario(self):
        """
        星形交叉场景：3~15 条轨迹在同一空间点同时汇聚，经过后各自机动散开。

        - 随机选取交叉点（空域中心区域）和交叉时刻 t_cross ∈ [T/4, 3T/4]
        - 各轨迹进入方位角均匀分布在 [0, 2π)，仰角随机略有差异
        - 交叉前：CV 直线飞向交叉点（起点由运动学逆推）
        - 交叉后：CA 机动，方向大致与进入方向相反，各自独立散开
        """
        for _ in range(30):
            n_traj = (self.n_cross_traj if self.n_cross_traj is not None
                      else np.random.randint(3, 16))  # 3~15 条

            # ── 交叉点（空域中心区域） ──
            r_cross   = np.random.uniform(self.r_min * 1.2, self.r_max * 0.8)
            a_cross   = np.random.uniform(-self.alpha_max * 0.4, self.alpha_max * 0.4)
            b_cross   = np.random.uniform(0, 2 * np.pi)
            cross_xyz = np.array(spherical_to_cartesian(r_cross, a_cross, b_cross))

            # ── 随机交叉时刻 ──
            t_cross = np.random.randint(self.num_frames // 4,
                                        3 * self.num_frames // 4)

            # ── 各轨迹进入方位角：均匀分布 ──
            base_az = np.random.uniform(0, 2 * np.pi)

            trajectories = []
            ok = True
            for i in range(n_traj):
                az   = base_az + i * 2 * np.pi / n_traj
                traj = self._generate_star_trajectory(
                    label=i + 1,
                    cross_xyz=cross_xyz,
                    t_cross=t_cross,
                    azimuth_approach=az,
                )
                if traj is None:
                    ok = False
                    break
                trajectories.append(traj)

            if ok and trajectories:
                meas, assoc = self._make_measurements(trajectories)
                return trajectories, meas, assoc

        # 重试失败 → 回退到父类两两交叉
        return super()._generate_crossing_scenario()

    def _generate_star_trajectory(self, label, cross_xyz, t_cross, azimuth_approach):
        """
        单条星形交叉轨迹：

        Phase 1 (0 → t_cross)   : CV 直线飞向交叉点
                                   起点 = cross_xyz + approach_dir * speed * t_cross * dt
                                   速度 = -approach_dir * speed（指向交叉点）
        Phase 2 (t_cross → T-1) : CA 机动，方向大致与进入方向相反，带向心修正
        """
        dt    = self.dt
        T     = self.num_frames
        Q     = self._cv_Q(dt)
        r_mid = (self.r_min + self.r_max) / 2

        for _ in range(20):
            # 进入仰角（各轨迹略有差异，整体大致水平）
            phi_approach = np.random.uniform(np.pi * 0.35, np.pi * 0.65)

            # 从交叉点指向起始位置的单位向量
            approach_dir = np.array([
                np.sin(phi_approach) * np.cos(azimuth_approach),
                np.sin(phi_approach) * np.sin(azimuth_approach),
                np.cos(phi_approach),
            ])

            speed_in  = np.random.uniform(*self.velocity_range)
            v_in      = -approach_dir * speed_in          # 指向交叉点
            start_pos = cross_xyz - v_in * (t_cross * dt) # = cross + approach_dir * speed * t

            if not self._in_fov(start_pos):
                continue

            states    = np.zeros((T, 6))
            states[0] = np.concatenate([start_pos, v_in])

            # ── Phase 1: CV ──
            valid = True
            for t in range(1, t_cross + 1):
                prev    = states[t - 1]
                new_pos = prev[:3] + prev[3:] * dt
                new_vel = prev[3:].copy()
                noise   = np.random.multivariate_normal(np.zeros(6), Q)
                states[t] = np.concatenate([new_pos, new_vel]) + noise
                if not self._in_fov(states[t, :3]):
                    valid = False
                    break

            if not valid:
                continue

            # ── Phase 2: CA 散开（方向大致与进入方向相反 + 随机偏转） ──
            az_leave  = azimuth_approach + np.pi + np.random.uniform(-np.pi / 3, np.pi / 3)
            phi_leave = np.random.uniform(np.pi * 0.3, np.pi * 0.7)
            leave_dir = np.array([
                np.sin(phi_leave) * np.cos(az_leave),
                np.sin(phi_leave) * np.sin(az_leave),
                np.cos(phi_leave),
            ])
            a_mag = np.random.uniform(*self.high_maneuver_accel_range)
            accel = a_mag * leave_dir

            for t in range(t_cross + 1, T):
                prev      = states[t - 1]
                pos       = prev[:3]
                r_cur     = np.linalg.norm(pos) + 1e-8
                cent_dir  = -pos / r_cur
                overshoot = (r_cur - r_mid) / r_mid
                cent      = cent_dir * overshoot * a_mag * 0.5
                total_a   = accel + cent
                new_pos   = pos + prev[3:] * dt + 0.5 * total_a * dt ** 2
                new_vel   = prev[3:] + total_a * dt
                noise     = np.random.multivariate_normal(np.zeros(6), Q)
                states[t] = np.concatenate([new_pos, new_vel]) + noise
                if not self._in_fov(states[t, :3]):
                    valid = False
                    break

            if valid:
                return {
                    'label': label, 'states': states,
                    'birth_frame': 0, 'death_frame': T - 1,
                }

        return None

    # ----------------------------------------------------------
    # 场景 3: 纺锤形（spindle）
    # ----------------------------------------------------------

    def _generate_spindle_scenario(self):
        """
        纺锤形场景（两种变体各占 50%）：
          标准纺锤形：接近 → 贴近平行 → 分开
          交叉纺锤形：接近 → 平行段中间互换左右（中点处真实交叉）→ 从对侧分开
        补充若干随机 CV 背景目标。
        """
        trajectories = []
        # spindle_crossing: True=强制交叉, False=强制标准, None=随机50%
        do_cross = (self.spindle_crossing if self.spindle_crossing is not None
                    else np.random.rand() < 0.5)
        if do_cross:
            t1, t2 = self._generate_spindle_pair_crossing(1)    # 交叉纺锤
        else:
            t1, t2 = self._generate_spindle_pair(1)             # 标准纺锤

        if t1 is not None:
            trajectories.append(t1)
        if t2 is not None:
            trajectories.append(t2)

        # 补充背景目标
        extra = max(0, np.random.poisson(max(0, self.lambda_0 - 2)))
        lc = len(trajectories) + 1
        for _ in range(extra):
            traj = self._generate_single_trajectory(lc)
            if traj is not None:
                trajectories.append(traj)
                lc += 1

        if not trajectories:
            return super().generate_single_scenario()  # fallback

        meas, assoc = self._make_measurements(trajectories)
        return trajectories, meas, assoc

    def _generate_spindle_pair(self, start_label,
                               approach_frac=0.25, parallel_frac=0.50):
        """
        生成纺锤形轨迹对（三段余弦平滑，平行贴近段可控）：

        Phase 1 靠近段  (0 → T_app)          : 余弦平滑，sep_far → sep_near，端点导数=0
        Phase 2 平行段  (T_app → T_app+T_par) : sep = sep_near 恒定，导数=0（贴近飞行）
        Phase 3 分离段  (T_app+T_par → T)     : 余弦平滑，sep_near → sep_far，端点导数=0

        默认比例：靠近 25% | 平行 50% | 分离 25%，可通过参数调整。
        三段衔接处速度/加速度均连续，无拐点突变。
        """
        T       = self.num_frames
        dt      = self.dt
        Q_small = self._cv_Q(dt) * 0.02

        # 三段帧数
        T_app = max(1, int(T * approach_frac))
        T_par = max(1, int(T * parallel_frac))
        T_dep = max(1, T - T_app - T_par)

        for _ in range(50):
            # —— 中心轨迹参考点（平行段中心时刻的位置） ——
            r_m = np.random.uniform(self.r_min * 1.4, self.r_max * 0.6)
            a_m = np.random.uniform(-self.alpha_max * 0.3, self.alpha_max * 0.3)
            b_m = np.random.uniform(0, 2 * np.pi)
            center_mid = np.array(spherical_to_cartesian(r_m, a_m, b_m))

            # —— 主飞行方向与横向方向 ——
            phi     = np.random.uniform(np.pi * 0.35, np.pi * 0.65)
            theta   = np.random.uniform(0, 2 * np.pi)
            par_dir = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            ])
            cross_dir = np.cross(par_dir, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(cross_dir) < 0.1:
                cross_dir = np.cross(par_dir, np.array([0.0, 1.0, 0.0]))
            cross_dir /= np.linalg.norm(cross_dir)

            # —— 间距参数 ——
            sep_near = np.random.uniform(*self.sep_near_range)   # 平行段横向间距 (m)
            sep_far  = np.random.uniform(*self.sep_far_range)    # 起终点横向间距 (m)
            speed    = np.random.uniform(150, 300)               # 主方向飞行速度 (m/s)

            # center_start：让平行段中心帧 t = T_app + T_par/2 时位于 center_mid
            t_mid        = T_app + T_par / 2
            center_start = center_mid - par_dir * speed * t_mid * dt

            # 视野边角检查（起终点最大间距处）
            center_end = center_start + par_dir * speed * (T - 1) * dt
            p1_start   = center_start - cross_dir * sep_far / 2
            p2_start   = center_start + cross_dir * sep_far / 2
            p1_end     = center_end   - cross_dir * sep_far / 2
            p2_end     = center_end   + cross_dir * sep_far / 2

            if not all(self._in_fov(p) for p in [p1_start, p2_start, p1_end, p2_end]):
                continue

            # —— 逐帧解析计算 ——
            s1    = np.zeros((T, 6))
            s2    = np.zeros((T, 6))
            valid = True

            for t in range(T):
                if t <= T_app:
                    # Phase 1: 余弦靠近  sep_far → sep_near
                    frac     = t / T_app                            # [0, 1]
                    alpha    = 0.5 * (1.0 - np.cos(np.pi * frac))  # 0 → 1
                    sep_t    = sep_far - (sep_far - sep_near) * alpha
                    # d(sep)/dt = -(sep_far-sep_near) * 0.5*sin(π*frac) * π/(T_app*dt)
                    d_sep_dt = (
                        -(sep_far - sep_near)
                        * 0.5 * np.sin(np.pi * frac)
                        * np.pi / max(T_app * dt, 1e-8)
                    )

                elif t <= T_app + T_par:
                    # Phase 2: 平行贴近  sep = sep_near 恒定
                    sep_t    = sep_near
                    d_sep_dt = 0.0

                else:
                    # Phase 3: 余弦分离  sep_near → sep_far
                    t_dep    = t - T_app - T_par                    # [0, T_dep]
                    frac     = t_dep / T_dep                        # [0, 1]
                    alpha    = 0.5 * (1.0 - np.cos(np.pi * frac))  # 0 → 1
                    sep_t    = sep_near + (sep_far - sep_near) * alpha
                    d_sep_dt = (
                        (sep_far - sep_near)
                        * 0.5 * np.sin(np.pi * frac)
                        * np.pi / max(T_dep * dt, 1e-8)
                    )

                center_t = center_start + par_dir * speed * t * dt
                p1 = center_t - cross_dir * sep_t / 2
                p2 = center_t + cross_dir * sep_t / 2
                v1 = par_dir * speed - cross_dir * d_sep_dt / 2
                v2 = par_dir * speed + cross_dir * d_sep_dt / 2

                n1    = np.random.multivariate_normal(np.zeros(6), Q_small)
                n2    = np.random.multivariate_normal(np.zeros(6), Q_small)
                s1[t] = np.concatenate([p1, v1]) + n1
                s2[t] = np.concatenate([p2, v2]) + n2

                if not (self._in_fov(s1[t, :3]) and self._in_fov(s2[t, :3])):
                    valid = False
                    break

            if valid:
                traj1 = {'label': start_label,     'states': s1,
                         'birth_frame': 0, 'death_frame': T - 1}
                traj2 = {'label': start_label + 1, 'states': s2,
                         'birth_frame': 0, 'death_frame': T - 1}
                return traj1, traj2

        return None, None

    def _generate_spindle_pair_crossing(self, start_label,
                                        approach_frac=0.25, parallel_frac=0.50):
        """
        交叉纺锤形轨迹对（三段余弦平滑，平行段中间两目标互换左右）：

        Phase 1 靠近段  (0 → T_app)           : 余弦平滑，sep_far → sep_near，端点导数=0
                                                  目标1 在 -cross_dir 侧，目标2 在 +cross_dir 侧
        Phase 2 交叉段  (T_app → T_app+T_par) : 两目标对称地互换左右
                                                  offset₁ = -sep_near/2 · cos(π·t'/T_par)
                                                  offset₂ = +sep_near/2 · cos(π·t'/T_par)
                                                  中点 t'=T_par/2 处两者位置重合（真实交叉点）
        Phase 3 分离段  (T_app+T_par → T)     : 从交叉后的对侧各自余弦散开
                                                  目标1 从 +sep_near/2 → +sep_far/2
                                                  目标2 从 -sep_near/2 → -sep_far/2

        三段衔接处速度连续（余弦端点导数=0），无拐点突变。
        """
        T       = self.num_frames
        dt      = self.dt
        Q_small = self._cv_Q(dt) * 0.02

        T_app = max(1, int(T * approach_frac))
        T_par = max(1, int(T * parallel_frac))
        T_dep = max(1, T - T_app - T_par)

        for _ in range(50):
            # —— 中心轨迹参考点 ——
            r_m = np.random.uniform(self.r_min * 1.4, self.r_max * 0.6)
            a_m = np.random.uniform(-self.alpha_max * 0.3, self.alpha_max * 0.3)
            b_m = np.random.uniform(0, 2 * np.pi)
            center_mid = np.array(spherical_to_cartesian(r_m, a_m, b_m))

            # —— 主飞行方向与横向方向 ——
            phi   = np.random.uniform(np.pi * 0.35, np.pi * 0.65)
            theta = np.random.uniform(0, 2 * np.pi)
            par_dir = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            ])
            cross_dir = np.cross(par_dir, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(cross_dir) < 0.1:
                cross_dir = np.cross(par_dir, np.array([0.0, 1.0, 0.0]))
            cross_dir /= np.linalg.norm(cross_dir)

            # —— 间距参数 ——
            sep_near = np.random.uniform(*self.sep_near_range)
            sep_far  = np.random.uniform(*self.sep_far_range)
            speed    = np.random.uniform(150, 300)

            t_mid        = T_app + T_par / 2
            center_start = center_mid - par_dir * speed * t_mid * dt
            center_end   = center_start + par_dir * speed * (T - 1) * dt

            # 视野检查：起点 / 终点的最大间距处
            # 交叉后 t1 在 +cross_dir 侧，t2 在 -cross_dir 侧
            chk = [
                center_start - cross_dir * sep_far / 2,   # t1 起点
                center_start + cross_dir * sep_far / 2,   # t2 起点
                center_end   + cross_dir * sep_far / 2,   # t1 终点（交叉后在 + 侧）
                center_end   - cross_dir * sep_far / 2,   # t2 终点（交叉后在 - 侧）
            ]
            if not all(self._in_fov(p) for p in chk):
                continue

            # —— 逐帧解析计算 ——
            s1    = np.zeros((T, 6))
            s2    = np.zeros((T, 6))
            valid = True

            for t in range(T):
                center_t = center_start + par_dir * speed * t * dt

                if t <= T_app:
                    # Phase 1: 余弦靠近 sep_far → sep_near（与标准纺锤相同）
                    frac    = t / T_app
                    alpha   = 0.5 * (1.0 - np.cos(np.pi * frac))
                    sep_t   = sep_far - (sep_far - sep_near) * alpha
                    d_sep   = (-(sep_far - sep_near)
                               * 0.5 * np.sin(np.pi * frac)
                               * np.pi / max(T_app * dt, 1e-8))
                    # t1 在 - 侧，t2 在 + 侧
                    off1 = -sep_t / 2;  d_off1 = -d_sep / 2
                    off2 = +sep_t / 2;  d_off2 = +d_sep / 2

                elif t <= T_app + T_par:
                    # Phase 2: 余弦互换左右
                    # off1: -sep_near/2·cos(π·t'/T_par)  →  -sep_near/2 .. 0 .. +sep_near/2
                    # off2: +sep_near/2·cos(π·t'/T_par)  →  +sep_near/2 .. 0 .. -sep_near/2
                    t_dep  = t - T_app
                    cos_v  = np.cos(np.pi * t_dep / T_par)
                    sin_v  = np.sin(np.pi * t_dep / T_par)
                    dcos   = -sin_v * np.pi / max(T_par * dt, 1e-8)  # d(cos)/dt

                    off1   = -sep_near / 2 * cos_v
                    d_off1 = -sep_near / 2 * dcos
                    off2   = +sep_near / 2 * cos_v
                    d_off2 = +sep_near / 2 * dcos

                else:
                    # Phase 3: 从交叉后对侧分离
                    # t1: +sep_near/2 → +sep_far/2
                    # t2: -sep_near/2 → -sep_far/2
                    t_dep  = t - T_app - T_par
                    frac   = t_dep / T_dep
                    delta  = (sep_far - sep_near) / 2 * 0.5 * (1.0 - np.cos(np.pi * frac))
                    d_delta = ((sep_far - sep_near) / 2
                               * 0.5 * np.sin(np.pi * frac)
                               * np.pi / max(T_dep * dt, 1e-8))
                    off1   = +sep_near / 2 + delta;  d_off1 = +d_delta
                    off2   = -sep_near / 2 - delta;  d_off2 = -d_delta

                p1 = center_t + cross_dir * off1
                p2 = center_t + cross_dir * off2
                v1 = par_dir * speed + cross_dir * d_off1
                v2 = par_dir * speed + cross_dir * d_off2

                n1    = np.random.multivariate_normal(np.zeros(6), Q_small)
                n2    = np.random.multivariate_normal(np.zeros(6), Q_small)
                s1[t] = np.concatenate([p1, v1]) + n1
                s2[t] = np.concatenate([p2, v2]) + n2

                if not (self._in_fov(s1[t, :3]) and self._in_fov(s2[t, :3])):
                    valid = False
                    break

            if valid:
                traj1 = {'label': start_label,     'states': s1,
                         'birth_frame': 0, 'death_frame': T - 1}
                traj2 = {'label': start_label + 1, 'states': s2,
                         'birth_frame': 0, 'death_frame': T - 1}
                return traj1, traj2

        return None, None


# ============================================================
# 多场景数据集
# ============================================================

class MTTDatasetMultiScenario(MTTDataset):
    """
    混合 4 种场景类型的数据集。

    继承 MTTDataset 以复用 __getitem__（归一化、padding、标签映射）；
    不调用 MTTDataset.__init__，手动设置所需属性并自行生成场景。
    """

    def __init__(
        self,
        num_scenarios_per_type: int,
        tau:                    int   = 4,
        max_targets:            int   = 20,
        max_measurements:       int   = 30,
        task_type:              int   = 1,
        seed:                   int   = None,
        crossing_probability:   float = 0.7,
        scenario_types:         list  = None,
    ):
        # 不调用 MTTDataset.__init__，只设置 __getitem__ 依赖的属性
        if scenario_types is None:
            scenario_types = SCENARIO_TYPES

        self.tau              = tau
        self.max_targets      = max_targets
        self.max_measurements = max_measurements
        self.num_scenarios    = num_scenarios_per_type * len(scenario_types)

        if seed is not None:
            np.random.seed(seed)

        self.generator = MTTDataGeneratorMultiScenario(
            task_type=task_type,
            crossing_probability=crossing_probability,
            seed=None,   # seed 已由上面 np.random.seed 设置
        )

        print(f"\n{'='*60}")
        print(f"多场景数据集: {len(scenario_types)} 种类型 × "
              f"{num_scenarios_per_type} = {self.num_scenarios} 条场景")
        print(f"{'='*60}")

        # 按类型生成（分类存储，方便按类型保存测试 pkl）
        self.scenarios_by_type: dict = {}
        raw_scenarios: list          = []

        for stype in scenario_types:
            print(f"  生成 {stype:<14} × {num_scenarios_per_type} 条...")
            type_list = []
            for i in range(num_scenarios_per_type):
                if (i + 1) % 200 == 0:
                    print(f"    {i+1}/{num_scenarios_per_type}")
                type_list.append(self.generator.generate_by_type(stype))
            self.scenarios_by_type[stype] = type_list
            raw_scenarios.extend(type_list)

        # 训练时打乱，使不同类型均匀分布在各 batch 中
        perm = np.random.permutation(len(raw_scenarios))
        self.scenarios = [raw_scenarios[i] for i in perm]

        print("数据集生成完毕！")
        self._create_sample_indices()


# ============================================================
# DataLoader 工厂
# ============================================================

def create_dataloaders_multi_scenario(
    num_train_scenarios:  int   = 2000,
    num_val_scenarios:    int   = 200,
    num_test_scenarios:   int   = 200,
    batch_size:           int   = 64,
    tau:                  int   = 4,
    max_targets:          int   = 20,
    max_measurements:     int   = 30,
    task_type:            int   = 1,
    num_workers:          int   = 0,
    crossing_probability: float = 0.7,
    scenario_types:       list  = None,
    **kwargs   # 向后兼容，忽略多余参数
):
    """
    创建 4 种场景混合的 DataLoader。

    Returns:
        (train_loader, val_loader, test_loader, test_scenarios_by_type, train_scenarios_by_type, val_scenarios_by_type)

        *_scenarios_by_type : dict  {type_name: [scenario, ...]}
            用于在 train.py 中按类型保存 pkl 文件或按类型做验证。
    """
    if scenario_types is None:
        scenario_types = SCENARIO_TYPES

    n_types          = len(scenario_types)
    n_per_type_train = max(1, num_train_scenarios // n_types)
    n_per_type_val   = max(1, num_val_scenarios   // n_types)
    n_per_type_test  = max(1, num_test_scenarios  // n_types)

    common = dict(
        tau=tau,
        max_targets=max_targets,
        max_measurements=max_measurements,
        task_type=task_type,
        crossing_probability=crossing_probability,
        scenario_types=scenario_types,
    )

    print("\n创建多场景训练集（seed=42）...")
    train_ds = MTTDatasetMultiScenario(num_scenarios_per_type=n_per_type_train, seed=42,  **common)
    print("\n创建多场景验证集（seed=142）...")
    val_ds   = MTTDatasetMultiScenario(num_scenarios_per_type=n_per_type_val,   seed=142, **common)
    print("\n创建多场景测试集（seed=242）...")
    test_ds  = MTTDatasetMultiScenario(num_scenarios_per_type=n_per_type_test,  seed=242, **common)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers),
        test_ds.scenarios_by_type,    # 测试集，按类型分组
        train_ds.scenarios_by_type,   # 训练集，按类型分组（用于数据泄露验证）
        val_ds.scenarios_by_type,     # 验证集，按类型分组（用于按类型统计准确率）
    )


# ============================================================
# 从已有场景列表构建 Dataset / DataLoader（用于按类型验证）
# ============================================================

class MTTDatasetFromScenarios(MTTDataset):
    """
    直接从已生成好的场景列表构造 Dataset，不重新生成数据。
    用于在训练过程中对每种场景类型单独计算验证指标。
    """
    def __init__(self, scenarios: list, tau: int, max_targets: int,
                 max_measurements: int, task_type: int = 1):
        # 跳过 MTTDataset.__init__ 的生成逻辑，手动设置属性
        self.scenarios        = list(scenarios)
        self.num_scenarios    = len(scenarios)
        self.tau              = tau
        self.max_targets      = max_targets
        self.max_measurements = max_measurements
        # 只需要 generator 的 dt / T / r_max 等参数，不需要它生成数据
        self.generator        = MTTDataGeneratorMultiScenario(task_type=task_type)
        self._create_sample_indices()


def create_pertype_val_loaders(
    val_scenarios_by_type: dict,
    tau: int,
    max_targets: int,
    max_measurements: int,
    task_type: int = 1,
    batch_size: int = 64,
    num_workers: int = 0,
) -> dict:
    """
    为每种场景类型创建独立的验证 DataLoader。

    Args:
        val_scenarios_by_type: {type_name: [scenario, ...]}

    Returns:
        {type_name: DataLoader}
    """
    loaders = {}
    for stype, scenarios in val_scenarios_by_type.items():
        ds = MTTDatasetFromScenarios(
            scenarios=scenarios,
            tau=tau,
            max_targets=max_targets,
            max_measurements=max_measurements,
            task_type=task_type,
        )
        loaders[stype] = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers)
    return loaders


# ============================================================
# 快速测试
# ============================================================

if __name__ == '__main__':
    print("Testing MTTDataGeneratorMultiScenario...")
    gen = MTTDataGeneratorMultiScenario(
        task_type=1, seed=42, crossing_probability=0.7
    )

    for stype in SCENARIO_TYPES:
        trajs, meas, assoc = gen.generate_by_type(stype)
        n_tgt  = len(trajs)
        n_meas = sum(len(m) for m in meas)
        print(f"  {stype:<14}: {n_tgt} 目标, 共 {n_meas} 条测量")

        if stype == 'crossing':
            print(f"    星形交叉目标数: {n_tgt}（期望 3~15）")

        if stype == 'spindle' and len(trajs) >= 2:
            p1 = trajs[0]['states'][:, :3]
            p2 = trajs[1]['states'][:, :3]
            n  = min(len(p1), len(p2))
            d  = np.linalg.norm(p1[:n] - p2[:n], axis=1)
            print(f"    纺锤对最近距离: {d.min():.1f} m（第 {d.argmin()} 帧）")

    # 验证交叉纺锤生成正确（中点距离应接近 0）
    print("\n验证交叉纺锤形（中点应接近 0）...")
    np.random.seed(0)
    for _ in range(5):
        t1, t2 = gen._generate_spindle_pair_crossing(1)
        if t1 is not None:
            d = np.linalg.norm(
                t1['states'][:, :3] - t2['states'][:, :3], axis=1
            )
            print(f"  最近距离: {d.min():.1f} m  帧号: {d.argmin()}")

    print("\n验证星形交叉（每场景 3~15 条轨迹）...")
    np.random.seed(1)
    for _ in range(5):
        trajs, _, _ = gen._generate_crossing_scenario()
        print(f"  轨迹数: {len(trajs)}")

    print("\nTesting DataLoader (small)...")
    train_loader, val_loader, test_loader, test_by_type, _, _ = \
        create_dataloaders_multi_scenario(
            num_train_scenarios=40, num_val_scenarios=8, num_test_scenarios=8,
            batch_size=4, crossing_probability=0.7,
        )
    batch = next(iter(train_loader))
    print(f"past_states:          {batch['past_states'].shape}")
    print(f"current_measurements: {batch['current_measurements'].shape}")
    print(f"test scenarios by type: { {k: len(v) for k, v in test_by_type.items()} }")
    print("\nAll tests passed!")
