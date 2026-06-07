"""
BAIT v2: Bayesian Inference using Transformers for Multi-Target Tracking
改进版本，包含三项结构/算法创新：

  1. Dual-Stream 双流并行架构
       - Track Stream:   独立的 Transformer Encoder 编码历史轨迹
       - Meas Stream:    独立的轻量 Transformer Encoder 编码当前帧测量
       - Cross-Bridge:   双向交叉注意力桥，使两路特征在关联前互相感知

  2. Track Query 动态轨迹槽（DETR 范式）
       - 可学习的 nn.Embedding 作为每条轨迹槽的先验嵌入
       - 替换原来被动接受 argmax 测量的 Filtering Decoder 输入
       - 各槽主动"寻找"对应目标，天然具备置换等变性

  3. Sinkhorn 可微最优传输关联
       - 用 Sinkhorn-Knopp 迭代替换硬 argmax (_match_and_sort)
       - 软赋值矩阵全程可微，改善 Filtering Decoder 的梯度流
       - 可学习温度参数自适应调整软/硬分配程度

注意：v2 模型结构与 v1 (原始 BAIT) 不兼容，v1 checkpoints 无法直接加载。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: [B, seq_len, d_model]"""
        return x + self.pe[:, :x.size(1), :]



class CrossBridge(nn.Module):
    """
    Dual-Stream 双向交叉注意力桥

    两个方向各自包含一个 Pre-LN 残差块：
      - Track → Meas:  历史轨迹特征关注当前测量（"预测位置附近有哪些测量？"）
      - Meas  → Track: 当前测量特征关注历史轨迹（"这个测量附近有哪些活跃轨迹？"）
    两次交叉注意力独立进行，输出分别通过 FFN + LayerNorm 强化。
    """

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        self.track_cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.track_norm1 = nn.LayerNorm(d_model)
        self.track_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.track_norm2 = nn.LayerNorm(d_model)

        self.meas_cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.meas_norm1 = nn.LayerNorm(d_model)
        self.meas_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.meas_norm2 = nn.LayerNorm(d_model)

    def forward(self, track_feats, meas_feats, meas_key_padding_mask=None):
        """
        Args:
            track_feats:           [B, tau*max_T, D]   Track stream 特征
            meas_feats:            [B, M, D]            Meas stream 特征
            meas_key_padding_mask: [B, M]               True = padding（应忽略的位置）
        Returns:
            enhanced_track: [B, tau*max_T, D]
            enhanced_meas:  [B, M, D]
        """
        t_ctx, _ = self.track_cross_attn(
            query=track_feats,
            key=meas_feats,
            value=meas_feats,
            key_padding_mask=meas_key_padding_mask,
        )
        track_feats = self.track_norm1(track_feats + t_ctx)
        track_feats = self.track_norm2(track_feats + self.track_ffn(track_feats))

        m_ctx, _ = self.meas_cross_attn(
            query=meas_feats,
            key=track_feats,
            value=track_feats,
        )
        meas_feats = self.meas_norm1(meas_feats + m_ctx)
        meas_feats = self.meas_norm2(meas_feats + self.meas_ffn(meas_feats))

        return track_feats, meas_feats



class BAIT(nn.Module):
    """
    BAIT v2 —— 三项创新融合版

    架构流程：
      past_states   → state_embedding → Track Encoder  ─┐
                                                          CrossBridge
      measurements  → meas_embedding  → Meas  Encoder  ─┘
                                              ↓ (enhanced meas_feats)
                                       Associate Decoder (query=meas)
                                              ↓
                                    Match Prob Matrix (MPM)
                                              ↓
                                    Sinkhorn 软赋值
                                              ↓
                              Track Query + Weighted Meas Emb
                                              ↓ (enhanced track_feats)
                                       Filtering Decoder
                                              ↓
                             filtered_states (位置) + existence_probs
    """

    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,               # Track Encoder 层数
        num_meas_encoder_layers=2,          # Meas Encoder 层数（轻量）
        num_associate_decoder_layers=3,
        num_filtering_decoder_layers=6,
        dim_feedforward_encoder=2048,
        dim_feedforward_meas_encoder=1024,  # Meas Encoder FFN 宽度
        dim_feedforward_bridge=1024,        # Cross-Bridge FFN 宽度
        dim_feedforward_associate=1024,
        dim_feedforward_filtering=2048,
        dropout=0.1,
        max_targets=20,
        state_dim=5,           # [label, x, y, z, t]
        measurement_dim=3,     # [x, y, z]
        output_state_dim=4,    # [label, x, y, z]（保留用于兼容性）
        sinkhorn_iters=5,      # Sinkhorn 迭代次数
    ):
        super().__init__()

        self.d_model          = d_model
        self.max_targets      = max_targets
        self.state_dim        = state_dim
        self.measurement_dim  = measurement_dim
        self.output_state_dim = output_state_dim
        self.sinkhorn_iters   = sinkhorn_iters
        self.coord_scale      = 50000.0  # 坐标归一化因子

        self.state_embedding       = nn.Linear(state_dim, d_model)
        self.measurement_embedding = nn.Linear(measurement_dim, d_model)
        self.pos_encoder           = PositionalEncoding(d_model)

        _track_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout, batch_first=True,
        )
        self.track_encoder = nn.TransformerEncoder(
            _track_enc_layer, num_layers=num_encoder_layers
        )

        _meas_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward_meas_encoder,
            dropout=dropout, batch_first=True,
        )
        self.meas_encoder = nn.TransformerEncoder(
            _meas_enc_layer, num_layers=num_meas_encoder_layers
        )

        self.cross_bridge = CrossBridge(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward_bridge,
            dropout=dropout,
        )

        _assoc_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward_associate,
            dropout=dropout, batch_first=True,
        )
        self.associate_decoder = nn.TransformerDecoder(
            _assoc_layer, num_layers=num_associate_decoder_layers
        )
        self.match_prob_head = nn.Linear(d_model, max_targets + 1)

        self.log_sinkhorn_temp = nn.Parameter(torch.zeros(1))

        self.track_queries = nn.Embedding(max_targets, d_model)

        _filt_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward_filtering,
            dropout=dropout, batch_first=True,
        )
        self.filtering_decoder = nn.TransformerDecoder(
            _filt_layer, num_layers=num_filtering_decoder_layers
        )

        self.state_output_head = nn.Linear(d_model, 3)

        self.existence_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """参数初始化"""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

        nn.init.xavier_uniform_(self.state_output_head.weight, gain=0.01)
        nn.init.zeros_(self.state_output_head.bias)

        nn.init.normal_(self.track_queries.weight, mean=0.0, std=0.02)

        nn.init.zeros_(self.log_sinkhorn_temp)

    def forward(self, past_states, current_measurements,
                num_past_targets, num_current_measurements):
        """
        前向传播（外部接口与原 BAIT v1 完全兼容）

        Args:
            past_states:               [B, tau*max_T, state_dim]     历史帧状态
            current_measurements:      [B, max_M, measurement_dim]   当前帧测量
            num_past_targets:          [B, tau]    每帧实际目标数（当前仅用于兼容性）
            num_current_measurements:  [B]         当前帧实际测量数

        Returns:
            match_prob_matrix:  [B, max_M, max_T+1]   关联概率矩阵（含 clutter 列）
            filtered_states:    [B, max_T, 3]          估计3D位置 [x,y,z]（归一化）
            existence_probs:    [B, max_T]             轨迹存在性概率 [0,1]
        """
        B     = past_states.size(0)
        max_M = current_measurements.size(1)

        meas_pad_mask = (
            torch.arange(max_M, device=current_measurements.device).unsqueeze(0)
            >= num_current_measurements.unsqueeze(1)
        )  # [B, max_M]


        track_emb = self.state_embedding(past_states)    # [B, tau*T, D]
        track_emb = self.pos_encoder(track_emb)
        track_feats = self.track_encoder(track_emb)      # [B, tau*T, D]

        meas_emb = self.measurement_embedding(current_measurements)  # [B, M, D]
        meas_emb = self.pos_encoder(meas_emb)
        meas_feats = self.meas_encoder(
            meas_emb,
            src_key_padding_mask=meas_pad_mask,
        )  # [B, M, D]

        track_feats, meas_feats = self.cross_bridge(
            track_feats, meas_feats,
            meas_key_padding_mask=meas_pad_mask,
        )


        assoc_out = self.associate_decoder(
            meas_feats,     # query
            track_feats,    # memory
        )  # [B, M, D]

        match_prob_matrix = self.match_prob_head(assoc_out)   # [B, M, T+1]
        match_prob_matrix_norm = F.softmax(match_prob_matrix, dim=-1)


        filtering_queries, soft_assignment = self._sinkhorn_assign(
            match_prob_matrix_norm,
            current_measurements,
            num_current_measurements,
        )


        track_q = self.track_queries.weight.unsqueeze(0).expand(B, -1, -1)

        weighted_meas_emb = self.measurement_embedding(filtering_queries)  # [B, T, D]

        filt_input = self.pos_encoder(track_q + weighted_meas_emb)  # [B, T, D]

        filt_out = self.filtering_decoder(
            filt_input,     # query: 槽位先验 + 当前观测信息
            track_feats,    # memory: 增强的历史轨迹特征（含测量上下文）
        )  # [B, T, D]

        state_correction = self.state_output_head(filt_out)    # [B, T, 3]
        filtered_states  = filtering_queries + state_correction  # [B, T, 3]

        existence_probs = self.existence_head(filt_out).squeeze(-1)  # [B, T]

        return match_prob_matrix_norm, filtered_states, existence_probs


    def _sinkhorn_assign(self, match_prob_matrix, measurements, num_measurements):
        """
        Sinkhorn-Knopp 迭代软赋值（替代原 _match_and_sort 中的硬 argmax）

        原理：
          将 [B, M, T] 匹配概率矩阵通过交替行/列对数归一化，
          趋近双随机矩阵（行列均归一化 → 满足"一个测量只能属于一条轨迹"的软约束）。
          整个过程全程可微，Filtering Decoder 通过软赋值获得连续梯度。

        vs 原始 argmax：
          argmax 直接截断梯度流，Filtering Decoder 无法通过关联决策反传梯度；
          Sinkhorn 软赋值保留概率分布信息，梯度可以完整传播。

        Args:
            match_prob_matrix: [B, M, T+1]  MPM（已 softmax 归一化，含 clutter 列 0）
            measurements:      [B, M, 3]    当前帧测量坐标
            num_measurements:  [B]          每帧有效测量数

        Returns:
            filtering_queries: [B, T, 3]   软赋值加权后的测量位置（Filtering Decoder 输入）
            soft_P:            [B, M, T]   软赋值矩阵（训练可微；推理时可 argmax 得硬赋值）
        """
        B, M, T_plus1 = match_prob_matrix.shape
        T = T_plus1 - 1

        log_p = torch.log(match_prob_matrix[:, :, 1:].clamp(min=1e-8))  # [B, M, T]

        temperature = self.log_sinkhorn_temp.exp().clamp(min=0.02, max=2.0)
        log_p = log_p / temperature

        pad_mask = (
            torch.arange(M, device=measurements.device).unsqueeze(0)
            >= num_measurements.unsqueeze(1)
        )  # [B, M]
        log_p = log_p.masked_fill(pad_mask.unsqueeze(-1), -1e9)

        for _ in range(self.sinkhorn_iters):
            log_p = log_p - torch.logsumexp(log_p, dim=2, keepdim=True)
            log_p = log_p.masked_fill(pad_mask.unsqueeze(-1), -1e9)

            log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
            log_p = log_p.masked_fill(pad_mask.unsqueeze(-1), -1e9)

        soft_P = torch.exp(log_p)  # [B, M, T]  软赋值矩阵（padding 行 ≈ 0）

        col_sum  = soft_P.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [B, 1, T]
        soft_P_n = soft_P / col_sum                                   # [B, M, T] 列归一化

        filtering_queries = torch.einsum('bmt,bmd->btd', soft_P_n, measurements)  # [B, T, 3]

        return filtering_queries, soft_P



class BAITLoss(nn.Module):
    """
    BAIT v2 损失函数

    三项损失：
      1. Association Loss = CE Loss + Dice Loss  （测量-轨迹关联）
      2. Filtering Loss   = MSE（米空间）         （3D 位置精度）
      3. Existence Loss   = BCE                   （轨迹存在性，v2 新增）
    """

    def __init__(
        self,
        gamma=1.0,
        association_weight=1.0,
        filtering_weight=1.0,
        existence_weight=0.5,       # v2 新增：存在性损失权重
    ):
        super().__init__()
        self.gamma              = gamma
        self.association_weight = association_weight
        self.filtering_weight   = filtering_weight
        self.existence_weight   = existence_weight

    def forward(
        self,
        match_prob_matrix,
        filtered_states,
        gt_associations,
        gt_states,
        num_measurements,
        num_targets,
        existence_probs=None,
        gt_slot_mask=None,
    ):
        """
        Args:
            match_prob_matrix: [B, max_M, max_T+1]  预测关联概率
            filtered_states:   [B, max_T, 3]         预测3D位置（归一化）
            gt_associations:   [B, max_M]             GT 关联标签（0=clutter）
            gt_states:         [B, max_T, 3]          GT 3D位置（归一化）
            num_measurements:  [B]
            num_targets:       [B]
            existence_probs:   [B, max_T] | None      预测存在性概率（v2 新增）

        Returns:
            total_loss, loss_dict
        """
        association_loss = self._association_loss(
            match_prob_matrix, gt_associations, num_measurements
        )

        filtering_loss = self._filtering_loss(
            filtered_states, gt_states, num_targets, gt_slot_mask=gt_slot_mask
        )

        existence_loss = torch.tensor(0.0, device=filtered_states.device)
        if existence_probs is not None:
            existence_loss = self._existence_loss(
                existence_probs, num_targets, gt_slot_mask=gt_slot_mask
            )

        total_loss = (
            self.association_weight * association_loss
            + self.filtering_weight   * filtering_loss
            + self.existence_weight   * existence_loss
        )

        with torch.no_grad():
            max_meas = match_prob_matrix.size(1)
            meas_mask = (
                torch.arange(max_meas, device=match_prob_matrix.device).unsqueeze(0)
                < num_measurements.unsqueeze(1)
            )
            meas_mask = meas_mask & (gt_associations >= 0)
            pred_assoc = match_prob_matrix.argmax(dim=-1)
            correct = (pred_assoc == gt_associations.long()) & meas_mask
            association_acc = (
                correct.float().sum() / meas_mask.float().sum().clamp(min=1.0)
            )

            coord_scale = 50000.0
            max_targets = filtered_states.size(1)
            if gt_slot_mask is not None:
                tgt_mask = gt_slot_mask.to(filtered_states.device).bool()
            else:
                tgt_mask = (
                    torch.arange(max_targets, device=filtered_states.device).unsqueeze(0)
                    < num_targets.unsqueeze(1)
                )
            pos_error = torch.norm(
                (filtered_states - gt_states) * coord_scale, dim=-1
            )
            pos_error_m = (
                (pos_error * tgt_mask.float()).sum()
                / tgt_mask.float().sum().clamp(min=1.0)
            )

        loss_dict = {
            'total_loss':       total_loss.item(),
            'association_loss': association_loss.item(),
            'filtering_loss':   filtering_loss.item(),
            'existence_loss':   existence_loss.item(),
            'association_acc':  association_acc.item(),
            'pos_error_m':      pos_error_m.item(),
        }

        return total_loss, loss_dict

    def _association_loss(self, match_prob_matrix, gt_associations, num_measurements):
        """
        关联损失 = CE Loss + Dice Loss
        论文公式(22)：Loss_Association = Loss_CE + Loss_Dice
        """
        batch_size      = match_prob_matrix.size(0)
        max_meas        = match_prob_matrix.size(1)
        max_T_plus1     = match_prob_matrix.size(2)

        mask = (
            torch.arange(max_meas, device=match_prob_matrix.device).unsqueeze(0)
            < num_measurements.unsqueeze(1)
        )
        mask = mask & (gt_associations >= 0)
        gt_associations = torch.clamp(gt_associations, 0, max_T_plus1 - 1)

        b_idx = torch.arange(batch_size, device=match_prob_matrix.device).unsqueeze(1).expand(-1, max_meas)
        m_idx = torch.arange(max_meas,   device=match_prob_matrix.device).unsqueeze(0).expand(batch_size, -1)
        pred_probs = match_prob_matrix[b_idx, m_idx, gt_associations.long()]

        ce_loss_per = -torch.log(pred_probs.clamp(min=1e-8))
        ce_loss = (ce_loss_per * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

        numerator  = 2 * pred_probs + self.gamma
        denominator = pred_probs ** 2 + self.gamma
        dice_terms  = numerator / denominator
        dice_sum    = (dice_terms * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
        dice_loss   = F.relu(1 - dice_sum)

        return ce_loss + dice_loss

    def _filtering_loss(self, filtered_states, gt_states, num_targets, gt_slot_mask=None):
        """
        滤波损失 = MSE（米空间）

        在归一化空间计算时梯度极小（量级 ~1e-8），
        乘以 COORD_SCALE² 等价于在米空间计算 MSE，
        再除以 1e9 使量级与关联损失相当（~1）。
        """
        coord_scale = 50000.0
        max_targets = filtered_states.size(1)

        diff = (filtered_states - gt_states) * coord_scale
        loss = (diff ** 2).sum(dim=-1)  # [B, T]  L2² per target

        if gt_slot_mask is not None:
            mask = gt_slot_mask.to(filtered_states.device).float()
        else:
            mask = (
                torch.arange(max_targets, device=filtered_states.device).unsqueeze(0)
                < num_targets.unsqueeze(1)
            ).float()
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        return loss / 1e9

    def _existence_loss(self, existence_probs, num_targets, gt_slot_mask=None):
        """
        存在性 BCE 损失（v2 新增）

        GT 存在性：轨迹槽 t < num_targets[b] 的为 1，其余为 0
        训练模型在槽位先验基础上准确预测轨迹是否活跃。
        """
        max_T    = existence_probs.size(1)
        if gt_slot_mask is not None:
            exist_gt = gt_slot_mask.to(existence_probs.device).float()
        else:
            exist_gt = (
                torch.arange(max_T, device=existence_probs.device).unsqueeze(0)
                < num_targets.unsqueeze(1)
            ).float()
        return F.binary_cross_entropy(existence_probs, exist_gt, reduction='mean')



if __name__ == "__main__":
    print("Testing BAIT v2 model...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BAIT(
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_meas_encoder_layers=2,
        num_associate_decoder_layers=3,
        num_filtering_decoder_layers=6,
        dim_feedforward_encoder=2048,
        dim_feedforward_meas_encoder=1024,
        dim_feedforward_bridge=1024,
        dim_feedforward_associate=1024,
        dim_feedforward_filtering=2048,
        max_targets=10,
        sinkhorn_iters=5,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数数量: {num_params:,}")

    batch_size       = 4
    tau              = 4
    max_targets      = 10
    max_measurements = 15

    past_states              = torch.randn(batch_size, tau * max_targets, 5).to(device)
    current_measurements     = torch.randn(batch_size, max_measurements, 3).to(device)
    num_past_targets         = torch.randint(5, max_targets, (batch_size, tau)).to(device)
    num_current_measurements = torch.randint(8, max_measurements, (batch_size,)).to(device)

    match_prob_matrix, filtered_states, existence_probs = model(
        past_states, current_measurements,
        num_past_targets, num_current_measurements
    )

    print(f"match_prob_matrix shape: {match_prob_matrix.shape}")  # [B, max_M, T+1]
    print(f"filtered_states   shape: {filtered_states.shape}")    # [B, T, 3]
    print(f"existence_probs   shape: {existence_probs.shape}")    # [B, T]

    criterion = BAITLoss(existence_weight=0.5)
    gt_associations = torch.randint(0, max_targets + 1, (batch_size, max_measurements)).to(device)
    gt_states       = torch.randn(batch_size, max_targets, 3).to(device)
    num_targets     = torch.randint(3, max_targets, (batch_size,)).to(device)

    total_loss, loss_dict = criterion(
        match_prob_matrix, filtered_states,
        gt_associations, gt_states,
        num_current_measurements, num_targets,
        existence_probs=existence_probs,
    )

    print(f"\nLoss dict: {loss_dict}")

    total_loss.backward()
    grad_track_q = model.track_queries.weight.grad
    grad_sinkhorn_temp = model.log_sinkhorn_temp.grad
    print(f"\nTrack Query 梯度范数:   {grad_track_q.norm().item():.4f}")
    print(f"Sinkhorn 温度梯度:       {grad_sinkhorn_temp.item():.6f}")
    print("\nBAIT v2 test passed!")
