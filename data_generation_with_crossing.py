"""
带两两交叉轨迹的数据生成（3D雷达场景）

交叉策略：
  - 以 crossing_probability 的概率生成交叉场景
  - 交叉场景中生成 1~2 对两两交叉轨迹
  - 不再使用星形交叉

观测模型与 data_generation.py 完全一致：
  R / alpha / beta 加噪声后转换为 xyz 输入模型
"""

import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import poisson

from data_generation import (
    MTTDataGenerator, MTTDataset,
    spherical_to_cartesian, cartesian_to_spherical,
    COORD_SCALE
)


# ============================================================
# 带交叉轨迹的生成器
# ============================================================

class MTTDataGeneratorWithCrossing(MTTDataGenerator):
    """在 MTTDataGenerator 基础上增加两两交叉场景"""

    def __init__(self, crossing_probability=0.5, **kwargs):
        """
        Args:
            crossing_probability: 生成交叉场景的概率（0~1）
        """
        super().__init__(**kwargs)
        self.crossing_probability = crossing_probability

    def generate_single_scenario(self):
        if np.random.rand() < self.crossing_probability:
            return self._generate_crossing_scenario()
        return super().generate_single_scenario()

    # ----------------------------------------------------------
    # 交叉场景
    # ----------------------------------------------------------

    def _generate_crossing_scenario(self):
        """生成 1~2 对两两交叉轨迹的3D场景"""
        trajectories  = []
        label_counter = 1

        num_pairs = np.random.randint(1, 3)   # 1 或 2 对
        for _ in range(num_pairs):
            t1, t2 = self._generate_crossing_pair(label_counter)
            if t1 is not None:
                trajectories.append(t1)
                label_counter += 1
            if t2 is not None:
                trajectories.append(t2)
                label_counter += 1

        # 补充随机目标，使总数接近 lambda_0
        extra = max(0, np.random.poisson(self.lambda_0) - len(trajectories))
        for _ in range(extra):
            traj = self._generate_single_trajectory(label_counter)
            if traj is not None:
                trajectories.append(traj)
                label_counter += 1

        trajectories = self._randomize_trajectory_lifetimes(trajectories)
        if not trajectories:
            return super().generate_single_scenario()

        # 生成测量（与父类逻辑一致：球坐标加噪声 → xyz）
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
    # 两两交叉对
    # ----------------------------------------------------------

    def _generate_crossing_pair(self, start_label):
        """
        生成两条在3D空间中交叉的轨迹。

        策略：
          1. 在观测体积中心区域选取交叉点（球坐标）
          2. 随机选择两个方向不同的速度向量（角度差 ≥ 30°）
          3. 向后推算起点：start = cross_xyz - v * (T/2)
        """
        # 交叉点（球坐标，位于观测体积中间区域）
        r_cross = np.random.uniform(self.r_min * 1.2, self.r_max * 0.8)
        a_cross = np.random.uniform(-self.alpha_max * 0.5, self.alpha_max * 0.5)
        b_cross = np.random.uniform(0, 2 * np.pi)
        cross_xyz = np.array(spherical_to_cartesian(r_cross, a_cross, b_cross))

        # 第一个速度方向（球面均匀采样）
        theta1 = np.random.uniform(0, 2 * np.pi)
        phi1   = np.random.uniform(np.pi * 0.3, np.pi * 0.7)   # 避免过于极端的仰角

        # 第二个方向：与第一个相差至少 30°（在方位角上）
        theta2 = theta1 + np.random.uniform(np.pi / 6, np.pi * 5 / 6)
        phi2   = np.clip(phi1 + np.random.uniform(-np.pi / 4, np.pi / 4),
                         np.pi * 0.2, np.pi * 0.8)

        speed1 = np.random.uniform(*self.velocity_range)
        speed2 = np.random.uniform(*self.velocity_range)

        v1 = speed1 * np.array([
            np.sin(phi1) * np.cos(theta1),
            np.sin(phi1) * np.sin(theta1),
            np.cos(phi1)
        ])
        v2 = speed2 * np.array([
            np.sin(phi2) * np.cos(theta2),
            np.sin(phi2) * np.sin(theta2),
            np.cos(phi2)
        ])

        half_time = self.T / 2
        start1 = cross_xyz - v1 * half_time
        start2 = cross_xyz - v2 * half_time

        traj1 = self._generate_trajectory_with_params(start_label,     start1, v1)
        traj2 = self._generate_trajectory_with_params(start_label + 1, start2, v2)
        return traj1, traj2

    # ----------------------------------------------------------
    # 给定初态生成轨迹（3D CV）
    # ----------------------------------------------------------

    def _generate_trajectory_with_params(self, label, start_pos, velocity):
        """给定起点和速度生成一条3D CV轨迹"""
        dt     = self.dt
        F      = self._cv_F(dt)
        Q      = self._cv_Q(dt)
        state  = np.concatenate([start_pos, velocity])
        states = np.zeros((self.num_frames, 6))
        states[0] = state

        for t in range(1, self.num_frames):
            state  = F @ state + np.random.multivariate_normal(np.zeros(6), Q)
            states[t] = state
            if not self._in_fov(state[:3]):
                # 轨迹出界，截断
                return {
                    'label': label, 'states': states[:t],
                    'birth_frame': 0, 'death_frame': t - 1
                }

        return {'label': label, 'states': states,
                'birth_frame': 0, 'death_frame': self.num_frames - 1}


# ============================================================
# 带交叉场景的数据集
# ============================================================

class MTTDatasetWithCrossing(MTTDataset):
    """包含两两交叉场景的多目标跟踪数据集（3D雷达场景）"""

    def __init__(
        self,
        num_scenarios=1000,
        tau=4,
        max_targets=20,
        max_measurements=30,
        task_type=1,
        seed=None,
        crossing_probability=0.5,
        **kwargs    # 忽略旧版的 star 相关参数
    ):
        self.num_scenarios    = num_scenarios
        self.tau              = tau
        self.max_targets      = max_targets
        self.max_measurements = max_measurements

        self.generator = MTTDataGeneratorWithCrossing(
            task_type=task_type,
            seed=seed,
            crossing_probability=crossing_probability,
        )

        print(f"Generating {num_scenarios} scenarios "
              f"(3D radar, crossing={crossing_probability*100:.0f}%)...")
        self.scenarios = []
        for i in range(num_scenarios):
            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{num_scenarios} scenarios")
            self.scenarios.append(self.generator.generate_single_scenario())
        print("Dataset generation completed!")

        self._create_sample_indices()

    # __getitem__ 完全继承 MTTDataset（已支持3D）


# ============================================================
# DataLoader 工厂
# ============================================================

def create_dataloaders_with_crossing(
    num_train_scenarios=800,
    num_val_scenarios=100,
    num_test_scenarios=100,
    batch_size=16,
    tau=4,
    max_targets=20,
    max_measurements=30,
    task_type=1,
    num_workers=0,
    crossing_probability=0.5,
    **kwargs    # 向后兼容，忽略旧版 star 参数
):
    """
    创建包含两两交叉场景的训练/验证/测试 DataLoader（3D雷达场景）

    Args:
        crossing_probability: 交叉场景比例 (0~1)，推荐 0.5
    """
    print("\n" + "=" * 60)
    print("创建3D雷达交叉场景数据集")
    print("=" * 60)
    print(f"交叉场景概率 : {crossing_probability * 100:.0f}%  (两两交叉)")
    print(f"训练/验证/测试: {num_train_scenarios}/{num_val_scenarios}/{num_test_scenarios}")
    print("=" * 60 + "\n")

    common = dict(
        tau=tau, max_targets=max_targets,
        max_measurements=max_measurements,
        task_type=task_type,
        crossing_probability=crossing_probability,
    )

    print("Creating training dataset...")
    train_ds = MTTDatasetWithCrossing(num_scenarios=num_train_scenarios, seed=42,  **common)
    print("\nCreating validation dataset...")
    val_ds   = MTTDatasetWithCrossing(num_scenarios=num_val_scenarios,   seed=142, **common)
    print("\nCreating test dataset...")
    test_ds  = MTTDatasetWithCrossing(num_scenarios=num_test_scenarios,  seed=242, **common)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    import numpy as np
    print("Testing 3D radar crossing scenario generation...")

    gen = MTTDataGeneratorWithCrossing(crossing_probability=1.0, task_type=1, seed=42)
    trajs, meas, assoc = gen.generate_single_scenario()

    print(f"\n生成轨迹数: {len(trajs)}")
    print(f"帧数:       {len(meas)}")
    for tr in trajs:
        xyz0 = tr['states'][0, :3]
        r, a, b = cartesian_to_spherical(*xyz0)
        spd = np.linalg.norm(tr['states'][0, 3:])
        print(f"  轨迹{tr['label']}: R={r:.0f}m  alpha={np.degrees(a):.1f}°  "
              f"beta={np.degrees(b):.1f}°  速度={spd:.1f}m/s")

    if len(trajs) >= 2:
        p1 = trajs[0]['states'][:, :3]
        p2 = trajs[1]['states'][:, :3]
        n  = min(len(p1), len(p2))
        d  = np.linalg.norm(p1[:n] - p2[:n], axis=1)
        print(f"\n轨迹1-2 最近距离: {d.min():.1f}m (第{d.argmin()}帧)")

    print("\n" + "=" * 60)
    print("Testing DataLoader...")
    train_loader, _, _ = create_dataloaders_with_crossing(
        num_train_scenarios=10, num_val_scenarios=5, num_test_scenarios=5,
        batch_size=4, crossing_probability=0.5
    )
    batch = next(iter(train_loader))
    print(f"past_states:          {batch['past_states'].shape}")
    print(f"current_measurements: {batch['current_measurements'].shape}")
    print(f"gt_states:            {batch['gt_states'].shape}")
    print("\nAll tests passed!")
