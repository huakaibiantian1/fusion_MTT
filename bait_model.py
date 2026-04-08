"""
BAIT: Bayesian Inference using Transformers for Multi-Target Tracking
Implementation based on the paper:
"Transformer-based Multi-Target Tracking with Bayesian Perspective"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class BAIT(nn.Module):
    """
    BAIT模型主体
    包含：
    1. Transformer Encoder（预测过程）
    2. Associate Decoder（数据关联）
    3. Filtering Decoder（更新过程）
    """
    def __init__(
        self,
        d_model=256,           # 状态维度增强后的维度
        nhead=8,               # 多头注意力的头数
        num_encoder_layers=6,  # Encoder层数
        num_associate_decoder_layers=3,  # Associate Decoder层数
        num_filtering_decoder_layers=6,  # Filtering Decoder层数
        dim_feedforward_encoder=2048,    # Encoder的FFN隐藏单元
        dim_feedforward_associate=1024,  # Associate Decoder的FFN隐藏单元
        dim_feedforward_filtering=2048,  # Filtering Decoder的FFN隐藏单元
        dropout=0.1,
        max_targets=20,        # 最大目标数
        state_dim=5,           # 输入状态维度 [label, x, y, z, t]
        measurement_dim=3,     # 测量维度 [x, y, z]
        output_state_dim=4,    # 输出状态维度 [label, x, y, z]
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_targets = max_targets
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.output_state_dim = output_state_dim
        
        # 坐标缩放因子（与data_generation.py保持一致）
        self.coord_scale = 50000.0  # 雷达场景 R_max=50000m

        # 状态嵌入层：将[label, x, y, z, t]映射到d_model维度
        self.state_embedding = nn.Linear(state_dim, d_model)

        # 测量嵌入层：将[x, y, z]映射到d_model维度
        self.measurement_embedding = nn.Linear(measurement_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # ===== 预测过程：Transformer Encoder =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # ===== 数据关联：Associate Decoder =====
        associate_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward_associate,
            dropout=dropout,
            batch_first=True
        )
        self.associate_decoder = nn.TransformerDecoder(
            associate_decoder_layer,
            num_layers=num_associate_decoder_layers
        )
        
        # 匹配概率矩阵输出层
        # 输出：每个测量对应每个轨迹的概率（包括clutter dummy trajectory）
        self.match_prob_head = nn.Linear(d_model, max_targets + 1)  # +1 for clutter
        
        # 存在性阈值计算（论文第217行提到的线性层）
        # 注意：这里简化实现，直接使用最大匹配概率作为存在概率
        # 在实际应用中可以添加一个阈值进行轨迹终止判断
        
        # ===== 更新过程：Filtering Decoder =====
        filtering_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward_filtering,
            dropout=dropout,
            batch_first=True
        )
        self.filtering_decoder = nn.TransformerDecoder(
            filtering_decoder_layer,
            num_layers=num_filtering_decoder_layers
        )
        
        # 状态输出层：输出[x, y, z]（label在match & sort中已确定）
        self.state_output_head = nn.Linear(d_model, 3)  # 输出3D位置 [x, y, z]
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """初始化参数"""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
        
        # 🔧 特别处理state_output_head：输出修正量，初始应接近0
        # 使用小的gain让初始修正量很小，从而稳定训练早期
        nn.init.xavier_uniform_(self.state_output_head.weight, gain=0.01)
        nn.init.zeros_(self.state_output_head.bias)
    
    def forward(self, past_states, current_measurements, num_past_targets, num_current_measurements):
        """
        前向传播

        Args:
            past_states: [batch_size, tau * max_targets, state_dim]
                过去tau帧的状态，state_dim = [label, x, y, z, t]
            current_measurements: [batch_size, max_measurements, measurement_dim]
                当前帧的测量值 [x, y, z]
            num_past_targets: [batch_size, tau] 每帧实际目标数
            num_current_measurements: [batch_size] 当前帧实际测量数

        Returns:
            match_prob_matrix: [batch_size, max_measurements, max_targets + 1]
            filtered_states:   [batch_size, max_targets, 3]  — [x, y, z]
            existence_probs:   [batch_size, max_targets]
        """
        batch_size = past_states.size(0)
        
        # ===== 1. 预测过程：Encoder处理过去的状态 =====
        # 嵌入过去的状态
        past_embedded = self.state_embedding(past_states)  # [B, tau*max_targets, d_model]
        past_embedded = self.pos_encoder(past_embedded)
        
        # Encoder生成预测嵌入
        encoder_output = self.encoder(past_embedded)  # [B, tau*max_targets, d_model]
        
        # ===== 2. 数据关联：Associate Decoder =====
        # 嵌入当前测量值
        measurement_embedded = self.measurement_embedding(current_measurements)  # [B, max_meas, d_model]
        measurement_embedded = self.pos_encoder(measurement_embedded)
        
        # Associate Decoder输出
        associate_output = self.associate_decoder(
            measurement_embedded,  # query
            encoder_output         # memory
        )  # [B, max_meas, d_model]
        
        # 生成匹配概率矩阵 (MPM)
        match_prob_matrix = self.match_prob_head(associate_output)  # [B, max_meas, max_targets+1]
        
        # ===== 3. Match & Sort机制 =====
        # 对每个测量，沿轨迹维度进行softmax归一化
        match_prob_matrix_normalized = F.softmax(match_prob_matrix, dim=-1)  # [B, max_meas, max_targets+1]
        
        # 为每个轨迹选择最可能的测量（argmax along measurements）
        # 这里我们需要为每个轨迹找到概率最高的测量
        filtering_queries = self._match_and_sort(
            match_prob_matrix_normalized,
            current_measurements,
            num_current_measurements
        )  # [B, max_targets, measurement_dim]
        
        # 计算存在性概率
        # 对每个轨迹，计算其对应的最大匹配概率
        max_probs_per_trajectory, _ = torch.max(
            match_prob_matrix_normalized[:, :, 1:],  # 排除clutter列
            dim=1
        )  # [B, max_targets]
        existence_probs = max_probs_per_trajectory  # [B, max_targets]
        
        # ===== 4. 更新过程：Filtering Decoder =====
        # 嵌入filtering queries
        filtering_query_embedded = self.measurement_embedding(filtering_queries)  # [B, max_targets, d_model]
        filtering_query_embedded = self.pos_encoder(filtering_query_embedded)
        
        # Filtering Decoder输出
        filtering_output = self.filtering_decoder(
            filtering_query_embedded,  # query
            encoder_output             # memory
        )  # [B, max_targets, d_model]
        
        # 输出过滤后的3D状态 + 残差连接
        # 模型学习对测量位置的"修正量"，类似Kalman滤波更新步骤
        state_correction = self.state_output_head(filtering_output)  # [B, max_targets, 3]
        filtered_states  = filtering_queries + state_correction       # 残差连接
        
        return match_prob_matrix_normalized, filtered_states, existence_probs
    
    def _match_and_sort(self, match_prob_matrix, measurements, num_measurements):
        """
        Match & Sort机制
        论文第215行："perform argmax along every column of MPM"
        为每个轨迹选择概率最高的测量
        
        Args:
            match_prob_matrix: [B, max_meas, max_targets+1] 归一化后的匹配概率
            measurements: [B, max_meas, measurement_dim]
            num_measurements: [B]
        
        Returns:
            filtering_queries: [B, max_targets, measurement_dim]
        """
        batch_size = match_prob_matrix.size(0)
        max_meas = match_prob_matrix.size(1)
        max_targets = match_prob_matrix.size(2) - 1  # 减去clutter列
        
        # 排除clutter列（索引0）
        match_prob_trajectories = match_prob_matrix[:, :, 1:]  # [B, max_meas, max_targets]
        
        # 论文：对每个轨迹（列），沿测量（行）方向找到概率最高的测量
        # 即对dim=1进行argmax，为每个轨迹（列）选择最佳测量（行）
        max_probs, max_indices = torch.max(match_prob_trajectories, dim=1)  # [B, max_targets]
        
        # 根据索引选择测量值
        batch_indices = torch.arange(batch_size, device=measurements.device).unsqueeze(1).expand(-1, max_targets)
        selected_measurements = measurements[batch_indices, max_indices]  # [B, max_targets, measurement_dim]
        
        return selected_measurements


class BAITLoss(nn.Module):
    """BAIT的损失函数"""
    def __init__(self, gamma=1.0, association_weight=1.0, filtering_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.association_weight = association_weight
        self.filtering_weight = filtering_weight
    
    def forward(self, match_prob_matrix, filtered_states,
                gt_associations, gt_states, num_measurements, num_targets):
        """
        计算总损失

        Args:
            match_prob_matrix: [B, max_meas, max_targets+1] 预测的匹配概率
            filtered_states:   [B, max_targets, 3]          预测的3D状态 [x,y,z]
            gt_associations:   [B, max_meas]                GT关联标签（0=clutter）
            gt_states:         [B, max_targets, 3]          GT 3D状态 [x,y,z]
            num_measurements:  [B]
            num_targets:       [B]

        Returns:
            total_loss, loss_dict
        """
        # ===== 1. Association Loss =====
        association_loss = self._association_loss(
            match_prob_matrix, gt_associations, num_measurements
        )
        
        # ===== 2. Filtering Loss =====
        filtering_loss = self._filtering_loss(
            filtered_states, gt_states, num_targets
        )
        
        # 总损失
        total_loss = (self.association_weight * association_loss + 
                     self.filtering_weight * filtering_loss)

        # ===== 3. 可解释指标（不参与梯度） =====
        with torch.no_grad():
            # 关联准确率：对有效测量，判断 argmax 是否等于 GT 标签
            max_meas = match_prob_matrix.size(1)
            meas_mask = (torch.arange(max_meas, device=match_prob_matrix.device)
                         .unsqueeze(0) < num_measurements.unsqueeze(1))  # [B, max_meas]
            pred_assoc = match_prob_matrix.argmax(dim=-1)                # [B, max_meas]
            correct = (pred_assoc == gt_associations.long()) & meas_mask
            association_acc = correct.float().sum() / meas_mask.float().sum().clamp(min=1.0)

            # 位置误差（反归一化到米）：对有效目标计算欧氏距离均值
            coord_scale = 50000.0
            max_targets = filtered_states.size(1)
            tgt_mask = (torch.arange(max_targets, device=filtered_states.device)
                        .unsqueeze(0) < num_targets.unsqueeze(1))         # [B, max_targets]
            pos_error = torch.norm((filtered_states - gt_states) * coord_scale, dim=-1)  # [B, max_targets]
            pos_error_m = (pos_error * tgt_mask.float()).sum() / tgt_mask.float().sum().clamp(min=1.0)

        loss_dict = {
            'total_loss': total_loss.item(),
            'association_loss': association_loss.item(),
            'filtering_loss': filtering_loss.item(),
            'association_acc': association_acc.item(),   # 关联正确率 0~1
            'pos_error_m': pos_error_m.item(),           # 位置误差（米）
        }
        
        return total_loss, loss_dict
    
    def _association_loss(self, match_prob_matrix, gt_associations, num_measurements):
        """
        关联损失 = CE Loss + Dice Loss
        论文公式(22):
        Loss_Association = Loss_CE + Loss_Dice
        = -1/m * Σ log(p_{i,ℓ_i}) + 1 - Σ (2*p_{i,ℓ_i} + γ) / (p_{i,ℓ_i}^2 + γ)
        
        Args:
            match_prob_matrix: [B, max_meas, max_targets+1]
            gt_associations: [B, max_meas] 值范围[0, max_targets]
            num_measurements: [B]
        """
        batch_size = match_prob_matrix.size(0)
        max_meas = match_prob_matrix.size(1)
        max_targets_plus_one = match_prob_matrix.size(2)
        
        # 确保gt_associations在有效范围内 [0, max_targets]
        gt_associations = torch.clamp(gt_associations, 0, max_targets_plus_one - 1)
        
        # 创建mask，只计算有效测量的损失
        mask = torch.arange(max_meas, device=match_prob_matrix.device).unsqueeze(0) < num_measurements.unsqueeze(1)
        
        # 获取预测概率 p_{i,ℓ_i}
        batch_indices = torch.arange(batch_size, device=match_prob_matrix.device).unsqueeze(1).expand(-1, max_meas)
        meas_indices = torch.arange(max_meas, device=match_prob_matrix.device).unsqueeze(0).expand(batch_size, -1)
        pred_probs = match_prob_matrix[batch_indices, meas_indices, gt_associations.long()]
        
        # CE Loss: -1/m * Σ log(p_{i,ℓ_i})
        # 注意：CE loss除以m（有效测量数）
        ce_loss_per_sample = -torch.log(pred_probs.clamp(min=1e-8))  # 防止log(0)
        ce_loss = (ce_loss_per_sample * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
        
        # Dice Loss: 1 - Σ (2*p_{i,ℓ_i} + γ) / (p_{i,ℓ_i}^2 + γ)
        # 注意：当 p→1 时原式会给出负值（1 - 1.5 = -0.5），导致梯度方向混乱；
        # 用 F.relu 截断，保证 Dice Loss ≥ 0，只在预测不准时产生梯度。
        numerator = 2 * pred_probs + self.gamma
        denominator = pred_probs ** 2 + self.gamma
        dice_terms = numerator / denominator
        dice_sum = (dice_terms * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
        dice_loss = F.relu(1 - dice_sum)   # clamp to non-negative

        return ce_loss + dice_loss
    
    def _filtering_loss(self, filtered_states, gt_states, num_targets):
        """
        过滤损失 = MSE（在米空间计算）

        坐标在归一化空间中（除以 50000），若直接算 Smooth L1 则梯度
        极小（200m 误差归一化后仅 0.004，Smooth L1 ≈ 0.000008/维）。
        乘以 COORD_SCALE² 等价于在原始米空间计算 MSE，保证滤波损失
        与关联损失量级可比，使 Filtering Decoder 获得足够梯度。

        Args:
            filtered_states: [B, max_targets, 3]  预测3D位置（归一化）
            gt_states:       [B, max_targets, 3]  真实3D位置（归一化）
            num_targets:     [B]
        """
        coord_scale = 50000.0
        max_targets = filtered_states.size(1)

        # MSE in meter space: scale² × mean((pred - gt)²)
        diff = (filtered_states - gt_states) * coord_scale   # 转换到米空间
        loss = (diff ** 2).sum(dim=-1)                       # [B, max_targets] L2² per target

        mask = torch.arange(max_targets, device=filtered_states.device).unsqueeze(0) < num_targets.unsqueeze(1)
        loss = (loss * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

        # 缩放到与关联损失同量级（除以 1e6，即 1000m² 量级 → ~1）
        return loss / 1e6


if __name__ == "__main__":
    # 测试模型
    print("Testing BAIT model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型参数
    model = BAIT(
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_associate_decoder_layers=3,
        num_filtering_decoder_layers=6,
        dim_feedforward_encoder=2048,
        dim_feedforward_associate=1024,
        dim_feedforward_filtering=2048,
        max_targets=10
    ).to(device)
    
    # 测试数据（3D版本）
    batch_size = 4
    tau = 4
    max_targets = 10
    max_measurements = 15

    past_states = torch.randn(batch_size, tau * max_targets, 5).to(device)  # [label, x, y, z, t]
    current_measurements = torch.randn(batch_size, max_measurements, 3).to(device)  # [x, y, z]
    num_past_targets = torch.randint(5, max_targets, (batch_size, tau)).to(device)
    num_current_measurements = torch.randint(10, max_measurements, (batch_size,)).to(device)

    match_prob_matrix, filtered_states, existence_probs = model(
        past_states, current_measurements, num_past_targets, num_current_measurements
    )

    print(f"Match Prob Matrix shape: {match_prob_matrix.shape}")  # [B, max_meas, max_targets+1]
    print(f"Filtered States shape:   {filtered_states.shape}")    # [B, max_targets, 3]
    print(f"Existence Probs shape:   {existence_probs.shape}")    # [B, max_targets]

    # 测试损失
    criterion = BAITLoss()
    gt_associations = torch.randint(0, max_targets + 1, (batch_size, max_measurements)).to(device)
    gt_states = torch.randn(batch_size, max_targets, 3).to(device)
    num_targets = torch.randint(5, max_targets, (batch_size,)).to(device)
    
    total_loss, loss_dict = criterion(
        match_prob_matrix, filtered_states,
        gt_associations, gt_states,
        num_current_measurements, num_targets
    )
    
    print(f"\nLoss dict: {loss_dict}")
    print("Model test passed!")
