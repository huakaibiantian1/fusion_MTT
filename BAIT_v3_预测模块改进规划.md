# BAIT v3 预测模块改进规划

> **基于版本**：BAIT v2（双流架构 + Track Query + Sinkhorn）  
> **改进焦点**：预测模块（Track Encoder）的结构缺陷与闭环反馈断路问题  
> **状态**：规划文档，待实现

---

## 目录

1. [v2 预测模块现状与问题](#1-v2-预测模块现状与问题)
2. [问题一：Teacher Forcing 与推理断路](#2-问题一teacher-forcing-与推理断路)
3. [问题二：时空结构被打平](#3-问题二时空结构被打平)
4. [问题三：位置差分信息未被显式利用](#4-问题三位置差分信息未被显式利用)
5. [改进 A：Scheduled Sampling（渐进关闭 Teacher Forcing）](#5-改进-a-scheduled-sampling渐进关闭-teacher-forcing)
6. [改进 B：GRU 轨迹记忆（显式预测-更新闭环）](#6-改进-b-gru-轨迹记忆显式预测-更新闭环)
7. [改进 C：帧感知二维位置编码](#7-改进-c-帧感知二维位置编码)
8. [改进 D：分层时序编码器](#8-改进-d-分层时序编码器)
9. [改进 E：位置差分隐式速度特征](#9-改进-e-位置差分隐式速度特征)
10. [改进 F：多步预测辅助损失](#10-改进-f-多步预测辅助损失)
11. [改进 G：因果注意力掩码](#11-改进-g-因果注意力掩码)
12. [实施路线图](#12-实施路线图)

---

## 1. v2 预测模块现状与问题

### 当前数据流

```
仿真器真实状态 traj['states'][t] = [x, y, z, vx, vy, vz]
                    ↓ __getitem__ 只取 [:3]，速度丢弃
past_states = [label, x_norm, y_norm, z_norm, t_norm]   ← 5维，仅位置

    tau=4 帧 × max_targets=20 个槽 = 80 个 token
                    ↓ state_embedding: Linear(5→256)
                    ↓ pos_encoder: 1D 正弦 PE（按 0~79 线性编号）
              Track Encoder（6层 Transformer）
                    ↓ encoder_output [B, 80, 256]
         ┌──────────┴──────────┐
  Associate Decoder      Filtering Decoder
  （测量-轨迹关联）         （状态更新）
         ↓                    ↓
       MPM               filtered_states [B, T, 3]
```

### 三个核心问题

| 问题 | 位置 | 影响 |
|------|------|------|
| Teacher Forcing：训练用 GT、推理用模型自身输出 | `__getitem__` | 训推分布不一致，长时跟踪误差积累 |
| 时空结构打平：帧内目标关系 ≠ 跨帧时序关系，但 PE 无法区分 | `pos_encoder` | 预测建模效率低，交叉场景帧内关系混淆 |
| 位置差分未利用：相邻帧位置差（隐式速度）未显式输入 | `__getitem__` | 模型须从注意力隐式推断速度，收敛慢 |

---

## 2. 问题一：Teacher Forcing 与推理断路

### 根本原因

这是序列模型的经典 **Exposure Bias** 问题：

```
训练模式（Teacher Forcing）：
  GT_{t-4} → GT_{t-3} → GT_{t-2} → GT_{t-1}
       └──────────────────────────────┘
               Track Encoder              → 预测第 t 帧
  ↑ 永远使用真实值，模型从未接触自身历史误差

推理模式（Autoregressive）：
  pred_{t-4} → pred_{t-3} → pred_{t-2} → pred_{t-1}
         └────────────────────────────────┘
                 Track Encoder               → 预测第 t 帧
  ↑ 使用自身历史预测，误差随时间积累
```

**对比卡尔曼滤波的闭环结构**：

| 步骤 | 标准卡尔曼滤波 | BAIT v2 |
|------|--------------|---------|
| 预测步 | `x̂⁻_t = F · x̂_{t-1}`，用**滤波后**状态预测 | Track Encoder 训练时用 GT 状态输入 |
| 更新步 | `x̂_t = x̂⁻_t + K(z_t - H·x̂⁻_t)` | Filtering Decoder 用 Sinkhorn 测量修正 ✅ |
| 闭环反馈 | 滤波后状态显式传递给下一帧预测 ✅ | **训练时断路，推理靠外部拼接** ❌ |

### 问题表现

- 场景越长（帧数越多），位置误差越倾向于累积
- 高机动场景中若某帧关联出错，误差会在后续帧的 Track Encoder 中被当作"正确历史"处理
- 模型从未在训练中学会"从错误中恢复"

---

## 3. 问题二：时空结构被打平

### 当前 1D PE 的语义混乱

80 个 token 的位置编码：

```
位置 0  = 目标槽1_帧0      位置 1  = 目标槽2_帧0  …  位置 19 = 目标槽20_帧0
位置 20 = 目标槽1_帧1      位置 21 = 目标槽2_帧1  …  位置 39 = 目标槽20_帧1
位置 40 = 目标槽1_帧2      …
位置 60 = 目标槽1_帧3      …
```

**语义错误**：
- "目标槽1_帧0"（位置0）和"目标槽1_帧3"（位置60）是**同一目标的时序演化**，PE 距离却是 60
- "目标槽1_帧0"（位置0）和"目标槽2_帧0"（位置1）是**同帧不同目标的空间关系**，PE 距离却只有 1
- 时序关系（距离大）与空间关系（距离小）被 PE 完全颠倒

这导致 Transformer 无法高效区分"我应该沿时间轴聚合同一目标的运动轨迹"和"我应该在帧内理解目标群体的空间格局"这两种截然不同的操作。

---

## 4. 问题三：位置差分信息未被显式利用

数据生成器中，每帧轨迹状态包含真实速度 `[vx, vy, vz]`，但：

1. **直接输入速度不现实**：实际雷达只观测位置（R, α, β → x, y, z），没有直接的速度观测
2. **但位置差分是合法的**：从连续帧的位置估计计算 `Δpos / Δt`，这是从可观测量推导的特征，是标准跟踪算法（α-β 滤波、卡尔曼滤波）的基础操作

当前模型只能通过注意力机制隐式地从位置序列推断运动趋势，而非显式利用位置差分信息。

---

## 5. 改进 A：Scheduled Sampling（渐进关闭 Teacher Forcing）

### 原理

训练时以概率 $p$（**教师强迫率**，随训练进度从 1 衰减到 0）决定是否使用 GT 状态作为 past_states 的输入，其余时候使用模型自身上一步的 `filtered_states`：

$$p(k) = \max\left(p_{\min},\ p_0 \cdot \lambda^k\right)$$

其中 $k$ 为当前训练步数，$\lambda$ 为衰减率，$p_{\min}$ 为最低保留比例（通常 0.1~0.3）。

### 实现方案

**`train.py` 修改——`train_one_step` 函数**：

```python
def train_one_step(model, batch, criterion, optimizer, device,
                   grad_clip=1.0, teacher_forcing_ratio=1.0):
    model.train()

    past_states              = batch['past_states'].to(device)
    current_measurements     = batch['current_measurements'].to(device)
    gt_associations          = batch['gt_associations'].to(device)
    gt_states                = batch['gt_states'].to(device)
    num_past_targets         = batch['num_past_targets'].to(device)
    num_current_measurements = batch['num_current_measurements'].squeeze(-1).to(device)
    num_current_targets      = batch['num_current_targets'].squeeze(-1).to(device)

    # Scheduled Sampling：以 (1 - teacher_forcing_ratio) 的概率
    # 用上一帧模型输出替换 past_states 的最后一帧
    if teacher_forcing_ratio < 1.0 and random.random() > teacher_forcing_ratio:
        with torch.no_grad():
            # 用上一帧的 GT past_states 先推一步，得到模型估计的"上一帧状态"
            _, prev_filtered, _ = model(
                past_states, current_measurements,
                num_past_targets, num_current_measurements
            )
        # 将 past_states 最后一帧（位置部分）替换为模型估计
        T = past_states.size(1) // 4   # max_targets
        # 替换最后一帧的 xyz 列（索引 1:4）
        last_frame_slice = slice(-T, None)
        past_states_ss = past_states.clone()
        past_states_ss[:, last_frame_slice, 1:4] = prev_filtered.detach()
        past_states = past_states_ss

    match_prob_matrix, filtered_states, existence_probs = model(
        past_states, current_measurements,
        num_past_targets, num_current_measurements
    )
    ...
```

**学习率调度器同步衰减**（在 `main()` 的训练循环中）：

```python
# teacher_forcing_ratio 衰减调度
def get_teacher_forcing_ratio(step, total_steps,
                               p0=1.0, p_min=0.1, decay_start=0.3):
    """
    前 decay_start 比例的 step 保持 p0=1（纯 Teacher Forcing）
    之后线性衰减到 p_min
    """
    start = int(total_steps * decay_start)
    if step < start:
        return p0
    progress = (step - start) / (total_steps - start)
    return max(p_min, p0 - (p0 - p_min) * progress)

# 在训练循环中：
tf_ratio = get_teacher_forcing_ratio(step, num_steps)
loss_dict = train_one_step(..., teacher_forcing_ratio=tf_ratio)
writer.add_scalar('train/teacher_forcing_ratio', tf_ratio, step)
```

### 超参数推荐

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `p0` | 1.0 | 初始完全 Teacher Forcing |
| `p_min` | 0.1~0.2 | 训练末期保留少量 GT 防止完全崩塌 |
| `decay_start` | 0.3 | 前 30% 的步数用纯 GT，之后渐进 |
| 衰减形状 | 线性或余弦 | 余弦衰减更平滑 |

---

## 6. 改进 B：GRU 轨迹记忆（显式预测-更新闭环）

### 原理

为每个轨迹槽维护一个 GRU 隐状态 $h_t$，在 Filtering Decoder 每次更新后通过 GRU 传递给下一帧 Track Encoder，建立真正的**递归预测-更新闭环**：

```
第 t 帧：
  [past_states + h_{t-1}] → Track Encoder → encoder_output
                                               ↓
                                        Filtering Decoder → filt_out_t
                                               ↓
                                       GRU: h_t = GRU(filt_out_t, h_{t-1})
第 t+1 帧：
  [past_states + h_t] → Track Encoder → ...（闭环）
```

与卡尔曼滤波的对应关系：
- `h_t`：等价于卡尔曼滤波的后验状态 $\hat{x}_t$（含误差协方差）
- GRU 更新：等价于卡尔曼更新步 `x̂_t = x̂⁻_t + K(z - Hx̂⁻_t)`
- Track Encoder with `h_{t-1}`：等价于卡尔曼预测步 `x̂⁻_t = Fx̂_{t-1}`

### 实现方案

**`bait_model.py` 新增 `TrackGRUMemory` 模块**：

```python
class TrackGRUMemory(nn.Module):
    """
    每个轨迹槽的跨帧 GRU 记忆
    在 Filtering Decoder 输出后更新，在下一帧 Track Encoder 前注入

    实现预测-更新闭环：
      Filtering Decoder 输出（更新后状态）→ GRU → 下一帧 Track Encoder 输入
    """
    def __init__(self, d_model, max_targets):
        super().__init__()
        self.gru_cell = nn.GRUCell(d_model, d_model)
        # 可学习初始隐状态（所有槽共用初始值）
        self.h0 = nn.Parameter(torch.zeros(1, d_model))

    def get_initial_hidden(self, batch_size, max_targets, device):
        """返回初始隐状态 [B, T, D]"""
        return self.h0.unsqueeze(0).expand(
            batch_size, max_targets, -1
        ).contiguous()

    def forward(self, filt_out, prev_hidden, existence_mask):
        """
        Args:
            filt_out:       [B, T, D]  Filtering Decoder 当前帧输出
            prev_hidden:    [B, T, D]  上一帧 GRU 隐状态
            existence_mask: [B, T]     True = 轨迹存在，False = 消亡（重置隐状态）
        Returns:
            new_hidden:     [B, T, D]  更新后的 GRU 隐状态
        """
        B, T, D = filt_out.shape
        new_h = self.gru_cell(
            filt_out.view(B * T, D),
            prev_hidden.view(B * T, D)
        ).view(B, T, D)

        # 消亡轨迹重置隐状态（避免错误历史污染新目标）
        mask = existence_mask.unsqueeze(-1).float()   # [B, T, 1]
        h0   = self.h0.expand(B, T, -1)              # [B, T, D]
        new_hidden = new_h * mask + h0 * (1.0 - mask)
        return new_hidden
```

**在 `BAIT.__init__` 中新增**：

```python
# GRU 轨迹记忆模块（创新 B）
self.track_gru = TrackGRUMemory(d_model, max_targets)

# 隐状态注入投影：将 GRU 隐状态投影并加入 Track Encoder 输入
self.gru_inject = nn.Linear(d_model, d_model)
```

**在 `BAIT.forward` 中修改**：

```python
def forward(self, past_states, current_measurements,
            num_past_targets, num_current_measurements,
            prev_hidden=None):    # ← 新增参数：上一帧 GRU 隐状态
    B = past_states.size(0)

    # Track Stream
    track_emb = self.state_embedding(past_states)
    track_emb = self.pos_encoder(track_emb)

    # ── 注入 GRU 隐状态（跨帧记忆）──────────────────────────────
    if prev_hidden is not None:
        # prev_hidden: [B, T, D]，广播到整个 tau*T 序列
        # 只对最后一帧的 T 个槽注入（最近的历史记忆）
        T = self.max_targets
        gru_signal = self.gru_inject(prev_hidden)   # [B, T, D]
        # 将隐状态加到 past_states 对应最后一帧的 token 上
        track_emb[:, -T:, :] = track_emb[:, -T:, :] + gru_signal

    track_feats = self.track_encoder(track_emb)
    # ... 后续流程不变 ...

    # ── 更新 GRU 隐状态 ────────────────────────────────────────
    existence_mask = (existence_probs > 0.5)          # [B, T]
    if prev_hidden is None:
        prev_hidden = self.track_gru.get_initial_hidden(
            B, self.max_targets, past_states.device
        )
    new_hidden = self.track_gru(filt_out, prev_hidden, existence_mask)

    return match_prob_matrix_norm, filtered_states, existence_probs, new_hidden
```

**推理时的调用方式**：

```python
# 推理循环
hidden = model.track_gru.get_initial_hidden(B, max_targets, device)

for frame_data in streaming_frames:
    mpm, states, exist, hidden = model(
        past_states, measurements,
        num_past, num_meas,
        prev_hidden=hidden      # 传入上一帧的 GRU 记忆
    )
    # hidden 会被传给下一帧
```

### GRU 记忆 vs 扩大 tau 的本质区别

| 方式 | 历史容量 | 记忆衰减 | 参数量 | 对消亡目标处理 |
|------|---------|---------|--------|--------------|
| 增大 tau（如 tau=8） | 线性增长 | 无（均等权重） | 序列长度翻倍 | 无（padding 填充） |
| GRU 记忆 | 理论无限 | 指数自然衰减 | 仅 GRU 参数 | 显式重置隐状态 |

GRU 记忆用极少的参数维护了"压缩后的完整轨迹历史"，是闭环反馈的关键。

---

## 7. 改进 C：帧感知二维位置编码

### 问题回顾

当前 1D 正弦 PE 按 `0~79` 线性编号，时序关系（同目标跨帧）距离远，帧内关系（同帧不同目标）距离近，语义完全颠倒。

### 实现方案

```python
class FrameAwarePositionalEncoding(nn.Module):
    """
    帧感知二维位置编码

    每个 token 的 PE = 帧时间编码（d_model/2） + 目标槽编码（d_model/2）

    使模型明确区分：
      - 同目标跨帧（时序演化）：相同的目标槽编码
      - 同帧不同目标（空间关系）：相同的帧时间编码
    """
    def __init__(self, d_model, max_targets=20, max_tau=16):
        super().__init__()
        assert d_model % 2 == 0
        # 帧时间编码：区分第几帧（tau 方向）
        self.frame_pe  = nn.Embedding(max_tau,         d_model // 2)
        # 目标槽编码：区分第几个目标槽（target 方向）
        self.target_pe = nn.Embedding(max_targets + 1, d_model // 2)  # +1 for padding

    def forward(self, x, tau, max_targets):
        """
        x: [B, tau*max_targets, D]
        """
        device = x.device
        # 生成每个 token 的 (frame_id, target_id)
        frame_ids  = torch.arange(tau,         device=device).repeat_interleave(max_targets)
        target_ids = torch.arange(max_targets, device=device).repeat(tau)

        pe = torch.cat([
            self.frame_pe(frame_ids),    # [tau*T, D/2]
            self.target_pe(target_ids),  # [tau*T, D/2]
        ], dim=-1)                        # [tau*T, D]

        return x + pe.unsqueeze(0)        # [B, tau*T, D]
```

**在 `BAIT.__init__` 中替换**：

```python
# 原来：
# self.pos_encoder = PositionalEncoding(d_model)

# 改为：
self.pos_encoder      = PositionalEncoding(d_model)        # 保留给 Meas Encoder 用
self.track_pos_encoder = FrameAwarePositionalEncoding(
    d_model, max_targets=max_targets, max_tau=16
)
```

**在 `forward` 中修改 Track Stream**：

```python
# Track Stream
track_emb = self.state_embedding(past_states)
# 用帧感知 PE 替换原来的 1D PE
track_emb = self.track_pos_encoder(track_emb, tau=tau, max_targets=self.max_targets)
track_feats = self.track_encoder(track_emb)
```

---

## 8. 改进 D：分层时序编码器

### 原理

将原来的单层 Track Encoder 替换为**两级注意力**结构，显式区分两种关系：

```
Level 1（帧内空间注意力）：
  每帧内 max_targets 个 token → self-attention
  捕获：目标群体的空间分布格局（对交叉/密集场景尤为重要）

Level 2（跨帧时序注意力）：
  同一目标槽 tau 个时刻的特征 → self-attention
  捕获：单目标的运动轨迹和速度趋势
```

### 实现方案

```python
class HierarchicalTrackEncoder(nn.Module):
    """
    分层时序编码器

    Level 1: 帧内空间注意力（每帧目标间关系）
    Level 2: 跨帧时序注意力（同目标时序演化）

    对应贝叶斯滤波的状态空间建模（Level 1）和时序预测（Level 2）
    """
    def __init__(self, d_model, nhead,
                 num_spatial_layers=2,    # Level 1 层数
                 num_temporal_layers=4,   # Level 2 层数
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()

        # Level 1: 帧内空间 Encoder
        _sp_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(
            _sp_layer, num_layers=num_spatial_layers
        )

        # Level 2: 跨帧时序 Encoder
        _tp_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            _tp_layer, num_layers=num_temporal_layers
        )

        # 帧级时间位置编码（可学习）
        self.frame_time_pe = nn.Embedding(16, d_model)   # 最多支持 tau=16

    def forward(self, track_emb, tau, max_targets):
        """
        track_emb: [B, tau*max_T, D]  已经经过 state_embedding 的嵌入
        """
        B, L, D = track_emb.shape
        T = max_targets

        # reshape: [B, tau, T, D]
        x = track_emb.view(B, tau, T, D)

        # ── Level 1: 帧内空间注意力 ─────────────────────────────
        # 每帧独立处理，T 个目标槽做 self-attention
        spatial_out = []
        for i in range(tau):
            fi = self.spatial_encoder(x[:, i, :, :])   # [B, T, D]
            spatial_out.append(fi)
        x = torch.stack(spatial_out, dim=1)            # [B, tau, T, D]

        # ── Level 2: 跨帧时序注意力 ─────────────────────────────
        # 对同一目标槽，tau 个时刻的特征做 self-attention + 时间 PE
        t_ids = torch.arange(tau, device=track_emb.device)
        t_pe  = self.frame_time_pe(t_ids)              # [tau, D]

        # [B, T, tau, D] → 以每个目标槽为独立序列
        x = x.permute(0, 2, 1, 3)                     # [B, T, tau, D]
        x = x + t_pe.unsqueeze(0).unsqueeze(0)        # 加时间 PE

        x_flat = x.contiguous().view(B * T, tau, D)   # [B*T, tau, D]
        x_flat = self.temporal_encoder(x_flat)         # [B*T, tau, D]
        x = x_flat.view(B, T, tau, D)

        # 取最后一帧（最新时刻）的特征作为预测输出
        # 同时保留完整序列供 Decoder memory 使用
        x_last   = x[:, :, -1, :]                     # [B, T, D]   最新预测
        x_full   = x.permute(0, 2, 1, 3).contiguous().view(B, tau*T, D)  # [B, tau*T, D]

        return x_full, x_last
```

**层数配置建议**（保持总参数量与原 6 层 Track Encoder 相近）：

| 配置 | Level 1（空间） | Level 2（时序） | 总层数等效 |
|------|----------------|----------------|-----------|
| 轻量版 | 2 层 | 3 层 | 5 层 |
| 标准版 | 2 层 | 4 层 | 6 层 |
| 强化版 | 3 层 | 5 层 | 8 层 |

---

## 9. 改进 E：位置差分隐式速度特征

### 合法性说明

实际雷达只能观测位置，但**连续帧估计位置的差分**是合法的推导量，是标准跟踪算法（α-β 滤波、CV 卡尔曼）的基础：

$$\hat{v}_t = \frac{\hat{x}_t - \hat{x}_{t-1}}{\Delta t}$$

这不是直接观测速度，而是从位置序列推导的运动特征。

### 实现方案

**`data_generation.py` `__getitem__` 修改**：

```python
for k, t in enumerate(range(f_idx - self.tau, f_idx)):
    frame_states = []
    for traj in trajectories:
        if traj['birth_frame'] <= t <= traj['death_frame']:
            xyz_cur = traj['states'][t, :3]

            # 计算位置差分（推导速度，非直接观测）
            if k == 0 or t - 1 < traj['birth_frame']:
                dxyz = np.zeros(3)           # 第一帧或出生帧，无先验差分
            else:
                xyz_pre = traj['states'][t - 1, :3]
                dxyz = (xyz_cur - xyz_pre) / self.generator.dt  # m/s

            frame_states.append(np.array([
                traj['label'],
                xyz_cur[0], xyz_cur[1], xyz_cur[2],     # 位置
                dxyz[0],    dxyz[1],    dxyz[2],         # 推导速度
                (t * self.generator.dt) / self.generator.T,  # 时间戳
            ]))
    ...
```

**归一化**：

```python
# past_states 归一化（__getitem__ 末尾）
past_states[:, 1:4] /= COORD_SCALE        # 位置 → [-1, 1]
past_states[:, 4:7] /= 500.0              # 速度归一化，最大速度约 500 m/s
```

**`config_multi_scenario.json` 修改**：

```json
"model": {
    "state_dim": 9
}
```

`state_dim` 从 5 改为 9（`[label, x, y, z, dx, dy, dz, t]`），`state_embedding` 自动从 `Linear(5→256)` 变为 `Linear(9→256)`，其余架构不变。

---

## 10. 改进 F：多步预测辅助损失

### 原理

给 Track Encoder 增加一个**直接监督信号**：除了通过 Filtering Decoder 间接优化，还让 Encoder 直接预测下一帧目标位置，迫使其真正学会运动预测。

```
encoder_output [B, tau*T, D]
        ↓ 取最后 T 个 token（对应最新帧）
prediction_head: Linear(D → 3)
        ↓
pred_next_pos [B, T, 3]   ← 预测下一帧位置
        ↓
L_pred = MSE(pred_next_pos, gt_states_next)
```

### 实现方案

**`bait_model.py` 修改**：

```python
# __init__ 新增
self.prediction_head = nn.Linear(d_model, 3)
nn.init.xavier_uniform_(self.prediction_head.weight, gain=0.01)
nn.init.zeros_(self.prediction_head.bias)

# forward 新增
# 取 encoder_output 最后 max_targets 个 token（最新帧的预测）
last_frame_enc = track_feats[:, -self.max_targets:, :]   # [B, T, D]
pred_next_pos  = self.prediction_head(last_frame_enc)    # [B, T, 3]

# 在返回值中增加
return match_prob_matrix_norm, filtered_states, existence_probs, pred_next_pos
```

**`data_generation.py` `__getitem__` 新增 `gt_states_next`**：

```python
# 当前帧的下一帧真值（用于预测损失监督）
gt_next_list = []
next_f = f_idx + 1
if next_f < len(measurements):
    for traj in trajectories:
        if traj['birth_frame'] <= next_f <= traj['death_frame']:
            gt_next_list.append(traj['states'][next_f, :3])
while len(gt_next_list) < self.max_targets:
    gt_next_list.append(np.zeros(3))
gt_states_next = np.array(gt_next_list[:self.max_targets]) / COORD_SCALE
else:
    gt_states_next = gt_states.copy()  # 最后一帧用当前帧 GT 代替

return {
    ...
    'gt_states_next': torch.FloatTensor(gt_states_next),  # 新增
}
```

**`BAITLoss` 新增预测损失项**：

```python
def _prediction_loss(self, pred_next_pos, gt_states_next, num_targets):
    """
    一步预测损失：督促 Track Encoder 真正学会运动预测
    量级与 filtering_loss 一致（米空间 MSE / 1e9）
    """
    coord_scale = 50000.0
    T = pred_next_pos.size(1)
    mask = (torch.arange(T, device=pred_next_pos.device).unsqueeze(0)
            < num_targets.unsqueeze(1))
    diff = (pred_next_pos - gt_states_next) * coord_scale
    loss = ((diff**2).sum(dim=-1) * mask.float()).sum()
    loss = loss / mask.float().sum().clamp(min=1.0)
    return loss / 1e9
```

---

## 11. 改进 G：因果注意力掩码

### 原理

当前 Track Encoder 做**双向** self-attention：帧0的 token 能看到帧3的内容（未来信息），而实际推理时帧3在帧0之后才到达。这造成**训练与推理的信息不对称**。

加入因果掩码后，帧 $i$ 的 token 只能 attend 到帧 $j \leq i$ 的 token。

### 实现方案

```python
def _build_causal_mask(self, tau, max_targets, device):
    """
    构造帧级因果掩码
    帧 i 的所有 token 可以看到帧 0~i 的所有 token
    帧 i 的所有 token 不能看到帧 i+1~tau-1 的 token

    返回: [tau*T, tau*T]  True = 屏蔽位置
    """
    T = max_targets
    L = tau * T
    mask = torch.zeros(L, L, dtype=torch.bool, device=device)

    for i in range(tau):
        for j in range(i + 1, tau):
            # 帧 i 的 token 不能看帧 j（j > i）的 token
            mask[i*T:(i+1)*T, j*T:(j+1)*T] = True

    return mask

# forward 中使用
causal_mask = self._build_causal_mask(tau, self.max_targets, past_states.device)
track_feats  = self.track_encoder(track_emb, mask=causal_mask)
```

> **注意**：PyTorch Transformer 的 `mask` 参数中 `True` 表示屏蔽位置。

---

## 12. 实施路线图

### 改进分类与优先级

| 改进 | 类型 | 改动量 | 预期收益 | 优先级 |
|------|------|--------|---------|--------|
| C：帧感知二维 PE | 结构 | 小 | 修复时空语义混乱 | ★★★ P1 |
| E：位置差分速度特征 | 数据 | 极小 | 补充运动学信息 | ★★★ P1 |
| G：因果注意力掩码 | 结构 | 极小 | 消除未来信息泄露 | ★★★ P1 |
| A：Scheduled Sampling | 训练策略 | 小 | 修复 Exposure Bias | ★★ P2 |
| F：多步预测辅助损失 | 损失 | 小 | 直接监督 Encoder 预测 | ★★ P2 |
| D：分层时序编码器 | 结构 | 中 | 显式时空建模分离 | ★★ P2 |
| B：GRU 轨迹记忆 | 结构 | 大 | 真正的递归闭环反馈 | ★ P3 |

### 推荐实施顺序

**Phase 1（低风险快速收益）**：C + E + G 同时实施

- 三者改动量均极小，且相互独立
- C 和 G 只改 `bait_model.py`，E 只改 `data_generation.py`
- 合计约 60 行代码改动

**Phase 2（中等改动）**：A + F 同时实施

- A（Scheduled Sampling）需修改 `train.py` 的训练循环
- F（多步预测损失）需同步修改 `data_generation.py`、`bait_model.py`、`BAITLoss`
- 合计约 100 行代码改动

**Phase 3（较大重构）**：D 或 B 二选一

- D（分层时序编码器）：替换 Track Encoder，参数量不变，但推理逻辑改变
- B（GRU 记忆）：引入有状态推理，需重写推理循环，工程复杂度最高
- 建议先做 D，验证分层结构收益后再评估是否需要 B

### 消融实验建议

为了定量评估各改进的单独贡献，建议按以下顺序做消融：

```
Baseline (v2)
    ↓ + C（帧感知 PE）→ 验证帧边界感知的贡献
    ↓ + E（差分速度）→ 验证运动特征的贡献
    ↓ + G（因果掩码）→ 验证因果性的贡献
    ↓ + A（SS）      → 验证 Exposure Bias 修复的贡献
    ↓ + F（预测损失）→ 验证直接监督的贡献
    ↓ = v3 完整版
```

每个 checkpoint 在 4 种场景类型上分别计算 OSPA / 关联准确率，可以清晰看出每项改进在哪类场景上贡献最大。
