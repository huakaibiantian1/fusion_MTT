"""
数据生成模块（3D雷达场景版本）

观测模型：
  - 观测站位于坐标原点 (0, 0, 0)
  - 观测量：R（斜距）、alpha（俯仰角）、beta（方位角）
  - 三路观测均含独立高斯噪声
  - 测量转换：(R, alpha, beta) → (x, y, z) 后输入模型

目标参数：
  - 距离范围：R ∈ [10000, 50000] m
  - 速度范围：100 ~ 500 m/s（航空目标量级）
  - 俯仰角范围：alpha ∈ [-30°, +30°]
  - 方位角范围：beta ∈ [0°, 360°]
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import poisson


# ============================================================
# 坐标系转换
# ============================================================

def spherical_to_cartesian(r, alpha, beta):
    """球坐标 → 笛卡尔坐标"""
    x = r * np.cos(alpha) * np.cos(beta)
    y = r * np.cos(alpha) * np.sin(beta)
    z = r * np.sin(alpha)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    """笛卡尔坐标 → 球坐标"""
    r     = np.sqrt(x**2 + y**2 + z**2)
    alpha = np.arcsin(z / (r + 1e-10))
    beta  = np.arctan2(y, x)
    return r, alpha, beta


# ============================================================
# 归一化常数（与坐标范围匹配：R_max ≈ 50000 m）
# ============================================================
COORD_SCALE = 50000.0


# ============================================================
# 数据生成器
# ============================================================

class MTTDataGenerator:
    """
    多目标跟踪数据生成器（3D雷达场景）

    轨迹状态 ：[x, y, z, vx, vy, vz]（6维）
    测量过程 ：真实位置 → 球坐标加噪声 (σ_R, σ_alpha, σ_beta) → 转回笛卡尔 [x, y, z]
    """

    def __init__(
        self,
        task_type=1,
        r_min=10000.0,             # 最小观测距离 (m)
        r_max=50000.0,             # 最大观测距离 (m)
        alpha_max=np.pi / 6,       # 最大俯仰角 ±30°
        velocity_range=(100, 500), # 目标速度范围 (m/s)
        dt=1.0,                    # 采样周期 (s)，1秒/帧（雷达扫描间隔）
        T=30.0,                    # 场景持续时间 (s) → 30帧
        lambda_0=3,                # 目标数泊松参数
        P_d=1.0,                   # 检测概率
        seed=None
    ):
        self.task_type     = task_type
        self.r_min         = r_min
        self.r_max         = r_max
        self.alpha_max     = alpha_max
        self.velocity_range = velocity_range
        self.dt            = dt
        self.T             = T
        self.lambda_0      = lambda_0
        self.P_d           = P_d

        # 任务特定参数
        if task_type == 1:
            self.q_s      = 0.5    # 过程噪声强度 (m²/s³)
            self.sigma_r  = 50.0   # 距离测量噪声标准差 (m)
            self.sigma_alpha = 0.001  # 俯仰角噪声 (rad) ≈ 0.057°
            self.sigma_beta  = 0.001  # 方位角噪声 (rad) ≈ 0.057°
            self.lambda_c = 10     # 杂波强度（每帧期望数量）
        elif task_type == 2:
            self.q_s      = 1.0
            self.sigma_r  = 100.0
            self.sigma_alpha = 0.002
            self.sigma_beta  = 0.002
            self.lambda_c = 20
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        self.num_frames = int(T / dt)

        if seed is not None:
            np.random.seed(seed)

    # ----------------------------------------------------------
    # 场景生成
    # ----------------------------------------------------------

    def generate_single_scenario(self):
        """
        生成单个场景。
        Returns:
            trajectories : list of dict（含 label / states[N,6] / birth_frame / death_frame）
            measurements : list of [num_meas, 3]   每帧的 xyz 测量（已含噪声，未归一化）
            associations : list of [num_meas]       每帧测量的真实目标标签（0=杂波）
        """
        num_targets = max(1, poisson.rvs(self.lambda_0))

        trajectories = [self._generate_single_trajectory(i + 1) for i in range(num_targets)]

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
                fm  = [fm[i]  for i in idx]
                fa  = [fa[i]  for i in idx]

            measurements.append(np.array(fm)  if fm  else np.empty((0, 3)))
            associations.append(np.array(fa)  if fa  else np.empty(0))

        return trajectories, measurements, associations

    # ----------------------------------------------------------
    # 3D 测量：球坐标加噪声 → xyz
    # ----------------------------------------------------------

    def _measure_3d(self, true_xyz):
        """真实位置 → 球坐标加噪声 → 返回含噪 xyz"""
        r_t, a_t, b_t = cartesian_to_spherical(*true_xyz)

        r_m = max(1.0, r_t + np.random.randn() * self.sigma_r)
        a_m = a_t + np.random.randn() * self.sigma_alpha
        b_m = b_t + np.random.randn() * self.sigma_beta

        return np.array(spherical_to_cartesian(r_m, a_m, b_m))

    # ----------------------------------------------------------
    # 单条轨迹生成（3D CV 模型）
    # ----------------------------------------------------------

    def _generate_single_trajectory(self, label):
        """
        生成一条 3D 匀速运动轨迹，起点在球坐标观测体积内。
        状态向量：[x, y, z, vx, vy, vz]
        """
        dt = self.dt
        F  = self._cv_F(dt)
        Q  = self._cv_Q(dt)

        for _ in range(200):
            # 球坐标均匀初始位置
            r0    = np.random.uniform(self.r_min, self.r_max)
            a0    = np.random.uniform(-self.alpha_max, self.alpha_max)
            b0    = np.random.uniform(0, 2 * np.pi)
            xyz0  = np.array(spherical_to_cartesian(r0, a0, b0))

            # 随机 3D 速度方向（球面均匀采样）
            v_mag = np.random.uniform(*self.velocity_range)
            phi   = np.random.uniform(0, np.pi)
            theta = np.random.uniform(0, 2 * np.pi)
            v0    = v_mag * np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])

            state  = np.concatenate([xyz0, v0])
            states = np.zeros((self.num_frames, 6))
            states[0] = state

            valid = True
            for t in range(1, self.num_frames):
                state  = F @ state + np.random.multivariate_normal(np.zeros(6), Q)
                states[t] = state
                if not self._in_fov(state[:3]):
                    valid = False
                    break

            if valid:
                return {'label': label, 'states': states,
                        'birth_frame': 0, 'death_frame': self.num_frames - 1}

        # 保底：切向低速轨迹（确保始终在视野内）
        r0   = np.random.uniform(self.r_min * 1.1, self.r_max * 0.9)
        a0   = 0.0
        b0   = np.random.uniform(0, 2 * np.pi)
        xyz0 = np.array(spherical_to_cartesian(r0, a0, b0))
        # 切向速度（在 xy 平面内旋转）
        v_mag = np.random.uniform(*self.velocity_range)
        v0    = v_mag * np.array([-np.sin(b0), np.cos(b0), 0.0])

        state  = np.concatenate([xyz0, v0])
        states = np.zeros((self.num_frames, 6))
        states[0] = state
        for t in range(1, self.num_frames):
            state  = F @ state
            states[t] = state

        return {'label': label, 'states': states,
                'birth_frame': 0, 'death_frame': self.num_frames - 1}

    # ----------------------------------------------------------
    # 杂波生成（球坐标均匀采样 → xyz）
    # ----------------------------------------------------------

    def _generate_clutter(self):
        r    = np.random.uniform(self.r_min, self.r_max)
        a    = np.random.uniform(-self.alpha_max, self.alpha_max)
        b    = np.random.uniform(0, 2 * np.pi)
        return np.array(spherical_to_cartesian(r, a, b))

    # ----------------------------------------------------------
    # 视野检查
    # ----------------------------------------------------------

    def _in_fov(self, pos):
        """检查 3D 位置是否在球坐标观测体积内"""
        r, alpha, _ = cartesian_to_spherical(*pos)
        return (self.r_min <= r <= self.r_max) and (abs(alpha) <= self.alpha_max)

    # 保留旧名称兼容性
    def _in_field_of_view_3d(self, pos):
        return self._in_fov(pos)

    # ----------------------------------------------------------
    # 3D CV 矩阵
    # ----------------------------------------------------------

    @staticmethod
    def _cv_F(dt):
        return np.array([
            [1, 0, 0, dt, 0,  0 ],
            [0, 1, 0, 0,  dt, 0 ],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0 ],
            [0, 0, 0, 0,  1,  0 ],
            [0, 0, 0, 0,  0,  1 ],
        ])

    def _cv_Q(self, dt):
        return self.q_s * np.array([
            [dt**3/3, 0,       0,       dt**2/2, 0,       0      ],
            [0,       dt**3/3, 0,       0,       dt**2/2, 0      ],
            [0,       0,       dt**3/3, 0,       0,       dt**2/2],
            [dt**2/2, 0,       0,       dt,      0,       0      ],
            [0,       dt**2/2, 0,       0,       dt,      0      ],
            [0,       0,       dt**2/2, 0,       0,       dt     ],
        ])


# ============================================================
# 数据集
# ============================================================

class MTTDataset(Dataset):
    """
    多目标跟踪数据集（3D雷达场景）

    past_states  : [tau * max_targets, 5]  — [label, x_norm, y_norm, z_norm, t]
    current_meas : [max_measurements,  3]  — [x_norm, y_norm, z_norm]
    gt_states    : [max_targets,       3]  — [x_norm, y_norm, z_norm]
    归一化因子    : COORD_SCALE = 50000 m
    """

    def __init__(self, num_scenarios=1000, tau=4, max_targets=20,
                 max_measurements=30, task_type=1, seed=None):
        self.num_scenarios    = num_scenarios
        self.tau              = tau
        self.max_targets      = max_targets
        self.max_measurements = max_measurements
        self.generator        = MTTDataGenerator(task_type=task_type, seed=seed)

        print(f"Generating {num_scenarios} scenarios (3D radar, R=[{self.generator.r_min/1000:.0f},{self.generator.r_max/1000:.0f}]km)...")
        self.scenarios = []
        for i in range(num_scenarios):
            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{num_scenarios} scenarios")
            self.scenarios.append(self.generator.generate_single_scenario())
        print("Dataset generation completed!")

        self._create_sample_indices()

    def _create_sample_indices(self):
        self.sample_indices = []
        for s_idx, (_, meas, _) in enumerate(self.scenarios):
            for f_idx in range(self.tau, len(meas)):
                self.sample_indices.append((s_idx, f_idx))

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        s_idx, f_idx = self.sample_indices[idx]
        trajectories, measurements, associations = self.scenarios[s_idx]

        # ── 1. 过去 tau 帧的状态 [tau*max_targets, 5] ──
        past_list       = []
        num_past_targets = []

        for t in range(f_idx - self.tau, f_idx):
            frame_states = []
            for traj in trajectories:
                if traj['birth_frame'] <= t <= traj['death_frame']:
                    xyz = traj['states'][t, :3]
                    frame_states.append(np.array([
                        traj['label'], xyz[0], xyz[1], xyz[2],
                        (t * self.generator.dt) / self.generator.T  # 时间戳归一化到[0,1]
                    ]))
            num_past_targets.append(len(frame_states))
            while len(frame_states) < self.max_targets:
                frame_states.append(np.zeros(5))
            past_list.extend(frame_states[:self.max_targets])

        past_states = np.array(past_list)   # [tau*max_targets, 5]

        # ── 2. 当前帧测量 ──────────────────────────────────────
        cur_meas  = measurements[f_idx].copy()
        cur_assoc = associations[f_idx].copy()
        n_meas    = len(cur_meas)

        # 标签映射
        lmap, lc = {}, 1
        for traj in trajectories:
            if traj['label'] not in lmap:
                lmap[traj['label']] = min(lc, self.max_targets)
                lc += 1
        assoc_mapped = np.zeros_like(cur_assoc)
        for i, lbl in enumerate(cur_assoc):
            assoc_mapped[i] = 0 if lbl == 0 else lmap.get(lbl, self.max_targets)
        cur_assoc = assoc_mapped

        if n_meas < self.max_measurements:
            pad_m = np.zeros((self.max_measurements - n_meas, 3))
            pad_a = np.zeros(self.max_measurements - n_meas)
            cur_meas  = np.vstack([cur_meas, pad_m]) if n_meas > 0 else pad_m
            cur_assoc = np.concatenate([cur_assoc, pad_a])
        else:
            cur_meas  = cur_meas[:self.max_measurements]
            cur_assoc = cur_assoc[:self.max_measurements]
            n_meas    = self.max_measurements

        # ── 3. Ground truth 状态 ───────────────────────────────
        gt_list = []
        for traj in trajectories:
            if traj['birth_frame'] <= f_idx <= traj['death_frame']:
                gt_list.append(traj['states'][f_idx, :3])
        n_gt = len(gt_list)
        while len(gt_list) < self.max_targets:
            gt_list.append(np.zeros(3))
        gt_states = np.array(gt_list[:self.max_targets])

        # ── 4. 归一化 ──────────────────────────────────────────
        past_states[:, 1:4] /= COORD_SCALE   # x, y, z
        cur_meas            /= COORD_SCALE
        gt_states           /= COORD_SCALE

        return {
            'past_states':              torch.FloatTensor(past_states),
            'current_measurements':     torch.FloatTensor(cur_meas),
            'gt_associations':          torch.LongTensor(cur_assoc.astype(np.int64)),
            'gt_states':                torch.FloatTensor(gt_states),
            'num_past_targets':         torch.LongTensor(num_past_targets),
            'num_current_measurements': torch.LongTensor([n_meas]),
            'num_current_targets':      torch.LongTensor([n_gt])
        }


# ============================================================
# DataLoader 工厂
# ============================================================

def create_dataloaders(num_train_scenarios=800, num_val_scenarios=100,
                       num_test_scenarios=100, batch_size=16, tau=4,
                       max_targets=20, max_measurements=30, task_type=1,
                       num_workers=0):
    kw = dict(tau=tau, max_targets=max_targets,
              max_measurements=max_measurements, task_type=task_type)
    train_ds = MTTDataset(num_scenarios=num_train_scenarios, seed=42,  **kw)
    val_ds   = MTTDataset(num_scenarios=num_val_scenarios,   seed=43,  **kw)
    test_ds  = MTTDataset(num_scenarios=num_test_scenarios,  seed=44,  **kw)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    print("Testing 3D radar data generation...")
    gen = MTTDataGenerator(task_type=1, seed=42)
    trajs, meas, assoc = gen.generate_single_scenario()

    print(f"\n生成目标数: {len(trajs)}")
    print(f"帧数:       {len(meas)}")
    print(f"轨迹0状态形状: {trajs[0]['states'].shape}")
    print(f"帧0测量形状:   {meas[0].shape}")

    # 打印第一条轨迹的起始 R/alpha/beta
    xyz0 = trajs[0]['states'][0, :3]
    r, a, b = cartesian_to_spherical(*xyz0)
    print(f"\n轨迹0起始球坐标: R={r:.0f}m  alpha={np.degrees(a):.1f}°  beta={np.degrees(b):.1f}°")
    spd = np.linalg.norm(trajs[0]['states'][0, 3:])
    print(f"轨迹0起始速度:   {spd:.1f} m/s")

    ds = MTTDataset(num_scenarios=10, tau=4, max_targets=20, max_measurements=30,
                    task_type=1, seed=42)
    sample = ds[0]
    print(f"\n样本形状:")
    print(f"  past_states:          {sample['past_states'].shape}")
    print(f"  current_measurements: {sample['current_measurements'].shape}")
    print(f"  gt_states:            {sample['gt_states'].shape}")
    print(f"\n3D radar data generation test passed!")
