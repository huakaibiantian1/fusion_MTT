"""
快速测试脚本
验证所有组件是否正常工作
"""

import torch
import numpy as np

print("="*60)
print("BAIT Implementation Quick Test")
print("="*60)

# 测试1: 导入模块
print("\n[1/5] Testing module imports...")
try:
    from bait_model import BAIT, BAITLoss
    from data_generation import MTTDataGenerator, MTTDataset, create_dataloaders
    from metrics import OSPAMetric, OSPA2Metric, TrackingMetrics
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# 测试2: 模型创建
print("\n[2/5] Testing model creation...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
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
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {num_params:,} parameters")
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    exit(1)

# 测试3: 数据生成
print("\n[3/5] Testing data generation...")
try:
    generator = MTTDataGenerator(task_type=1, seed=42)
    trajectories, measurements, associations = generator.generate_single_scenario()
    print(f"  Generated scenario with {len(trajectories)} targets")
    print(f"  Number of frames: {len(measurements)}")
    print(f"✓ Data generation successful")
except Exception as e:
    print(f"✗ Data generation failed: {e}")
    exit(1)

# 测试4: 前向传播
print("\n[4/5] Testing forward pass...")
try:
    batch_size = 2
    tau = 4
    max_targets = 10
    max_measurements = 15
    
    past_states = torch.randn(batch_size, tau * max_targets, 4).to(device)
    current_measurements = torch.randn(batch_size, max_measurements, 2).to(device)
    num_past_targets = torch.randint(5, max_targets, (batch_size, tau)).to(device)
    num_current_measurements = torch.randint(10, max_measurements, (batch_size,)).to(device)
    
    with torch.no_grad():
        match_prob_matrix, filtered_states, existence_probs = model(
            past_states, current_measurements, num_past_targets, num_current_measurements
        )
    
    print(f"  Output shapes:")
    print(f"    Match prob matrix: {match_prob_matrix.shape}")
    print(f"    Filtered states: {filtered_states.shape}")
    print(f"    Existence probs: {existence_probs.shape}")
    print(f"✓ Forward pass successful")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试5: 损失计算
print("\n[5/5] Testing loss calculation...")
try:
    criterion = BAITLoss(gamma=1.0)
    
    gt_associations = torch.randint(0, max_targets + 1, (batch_size, max_measurements)).to(device)
    gt_states = torch.randn(batch_size, max_targets, 2).to(device)
    num_targets = torch.randint(5, max_targets, (batch_size,)).to(device)
    
    total_loss, loss_dict = criterion(
        match_prob_matrix, filtered_states,
        gt_associations, gt_states,
        num_current_measurements, num_targets
    )
    
    print(f"  Loss values:")
    print(f"    Total: {loss_dict['total_loss']:.4f}")
    print(f"    Association: {loss_dict['association_loss']:.4f}")
    print(f"    Filtering: {loss_dict['filtering_loss']:.4f}")
    print(f"✓ Loss calculation successful")
except Exception as e:
    print(f"✗ Loss calculation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试6: OSPA指标
print("\n[Bonus] Testing OSPA metrics...")
try:
    ospa = OSPAMetric(c=1.0, p=1)
    
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    Y = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]])
    
    dist, loc, card = ospa(X, Y)
    print(f"  OSPA test: dist={dist:.4f}, loc={loc:.4f}, card={card:.4f}")
    print(f"✓ OSPA metrics working")
except Exception as e:
    print(f"✗ OSPA metrics failed: {e}")
    exit(1)

print("\n" + "="*60)
print("All tests passed! ✓")
print("="*60)
print("\nYou can now:")
print("1. Train the model: python train.py --task-type 1 --num-steps 1000")
print("2. Evaluate: python evaluate.py --checkpoint <path> --task-type 1")
print("3. Read README.md for detailed instructions")
print("="*60)
