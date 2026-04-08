# BAIT: Bayesian Inference using Transformers for Multi-Target Tracking

本项目实现了论文《Transformer-based Multi-Target Tracking with Bayesian Perspective》中提出的BAIT算法。

## 论文信息

**标题**: Transformer-based Multi-Target Tracking with Bayesian Perspective

**作者**: Xinwei Wei, Yiru Lin, Linao Zhang, Zhiyuan Zou, Jianwei Wei, Wei Yi

**机构**: University of Electronic Science and Technology of China

## 算法概述

BAIT是一个基于Transformer的多目标跟踪器，其架构模仿了贝叶斯推理的预测-更新结构：

1. **预测过程 (Prediction)**: Transformer Encoder分析过去τ帧的状态，生成预测嵌入
2. **数据关联 (Data Association)**: Associate Decoder匹配测量值与轨迹
3. **更新过程 (Update)**: Filtering Decoder基于关联结果更新目标状态

### 主要特点

- 结合了模型驱动的贝叶斯架构和数据驱动的深度学习
- 使用神经网络替代预设的运动和观测模型
- 递归地完成准确的预测和更新
- 在复杂的数据关联场景中表现优异

## 项目结构

```
fusion_MTT/
├── bait_model.py          # BAIT模型核心实现
├── data_generation.py     # 数据生成模块
├── metrics.py             # 评估指标(OSPA, OSPA2)
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── requirements.txt       # 依赖包
└── README.md              # 项目说明
```

## 环境要求

- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA (可选，用于GPU加速)

## 安装

1. 克隆或下载本项目

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

#### Task 1 (中等杂波强度)
```bash
python train.py \
    --task-type 1 \
    --num-train-scenarios 800 \
    --num-val-scenarios 100 \
    --batch-size 16 \
    --num-steps 800000 \
    --lr 1e-4 \
    --save-dir checkpoints/task1 \
    --log-dir logs/task1
```

#### Task 2 (高杂波强度)
```bash
python train.py \
    --task-type 2 \
    --num-train-scenarios 800 \
    --num-val-scenarios 100 \
    --batch-size 16 \
    --num-steps 800000 \
    --lr 1e-4 \
    --save-dir checkpoints/task2 \
    --log-dir logs/task2
```

#### 主要训练参数说明

- `--task-type`: 任务类型 (1: λ_c=10, 2: λ_c=20)
- `--num-steps`: 训练步数 (论文中为800k)
- `--batch-size`: 批次大小 (默认16)
- `--lr`: 学习率 (默认1e-4)
- `--tau`: 使用过去的帧数 (默认4)
- `--max-targets`: 最大目标数 (默认20)
- `--max-measurements`: 最大测量数 (默认30)

### 2. 评估模型

```bash
python evaluate.py \
    --checkpoint checkpoints/task1/best_model.pth \
    --task-type 1 \
    --num-scenarios 1000 \
    --output-dir evaluation_results/task1 \
    --visualize
```

评估脚本会：
- 在指定数量的Monte Carlo场景上评估模型
- 计算每帧的OSPA指标
- 生成评估报告 (JSON格式)
- 绘制OSPA曲线图 (如果启用 `--visualize`)

### 3. 监控训练

使用TensorBoard监控训练过程：
```bash
tensorboard --logdir logs
```

## 模型架构参数

根据论文IV.B节的设置：

| 组件 | 参数 | 值 |
|-----|------|-----|
| Encoder | 层数 | 6 |
| | 注意力头数 | 8 |
| | FFN隐藏单元 | 2048 |
| | 状态维度 d' | 256 |
| Associate Decoder | 层数 | 3 |
| | FFN隐藏单元 | 1024 |
| Filtering Decoder | 层数 | 6 |
| | FFN隐藏单元 | 2048 |

## 任务参数

根据论文IV.A节的设置：

### 共同参数
- 视野范围: [-30m, 30m] × [-30m, 30m]
- 速度范围: ±U(10, 20) m/s
- 采样周期: Δt = 0.1s
- 运动持续时间: T = 2s
- 检测概率: P_d = 0.95
- 初始目标数: 泊松分布 λ_0 = 8
- 过程噪声: q_s = 0.09 m²/s²
- 测量噪声: R = 0.01 m²

### Task 1
- 杂波强度: λ_c = 10

### Task 2
- 杂波强度: λ_c = 20 (更具挑战性)

## 损失函数

### 关联损失 (Association Loss)
```
Loss_Association = Loss_CE + Loss_Dice
```
- Cross-Entropy Loss: 测量-轨迹匹配的交叉熵
- Dice Loss: 提高难样本的学习效果

### 过滤损失 (Filtering Loss)
```
Loss_Filtering = Smooth_L1(预测状态, 真实状态)
```

## 评估指标

### OSPA (Optimal Sub-Pattern Assignment)
用于单帧评估，分解为：
- **定位误差 (Localization)**: 匹配目标的位置误差
- **基数误差 (Cardinality)**: 目标数量估计误差

### OSPA(2)
用于轨迹评估，考虑时间维度上的一致性

## 注意事项

1. **数据泄露防范**：
   - 数据生成时随机打乱测量值顺序
   - 不使用未来信息
   - 严格遵循因果性原则

2. **轨迹初始化**：
   - 当前实现假设目标已被初始化并分配标签
   - 在实际应用中需要额外的轨迹管理模块

3. **计算资源**：
   - 训练800k步大约需要数小时到数天（取决于硬件）
   - 建议使用GPU加速
   - 可以通过减少 `num_steps` 进行快速测试

## 实验结果参考

根据论文，在Task 1中：
- BAIT在整个跟踪过程中保持稳定的性能
- 在轨迹交叉等复杂场景中优于传统KF和MT3

在Task 2中（高杂波）：
- BAIT的基数误差明显低于KF
- 定位精度也有优势
- 展示了优秀的数据关联能力

## 引用

如果您使用本代码，请引用原论文：

```bibtex
@article{wei2024bait,
  title={Transformer-based Multi-Target Tracking with Bayesian Perspective},
  author={Wei, Xinwei and Lin, Yiru and Zhang, Linao and Zou, Zhiyuan and Wei, Jianwei and Yi, Wei},
  journal={...},
  year={2024}
}
```

## 许可证

本项目仅用于学术研究和学习目的。

## 联系方式

如有问题，请参考原论文或联系作者。
