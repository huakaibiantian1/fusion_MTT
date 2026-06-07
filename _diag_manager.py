"""快速诊断：Oracle vs TrackManager"""
import copy
import torch
import numpy as np
from bait_model import BAIT
from data_generation_multi_scenario import MTTDataGeneratorMultiScenario
from evaluate_3d import evaluate_scenario, evaluate_scenario_with_manager, COORD_SCALE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = 'checkpoints_v3/best_model.pth'
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
cfg = ckpt.get('config', {})
m = cfg.get('model', {}) if isinstance(cfg, dict) else {}

model = BAIT(
    d_model=m.get('d_model', 256),
    nhead=m.get('nhead', 8),
    max_targets=m.get('max_targets', 20),
).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

gen = MTTDataGeneratorMultiScenario(task_type=1, T=30.0, seed=42, crossing_probability=0.7)
scenario = gen.generate_by_type('crossing')
trajs, meas, assoc = scenario
print(f'场景: {len(trajs)} 目标, {len(meas)} 帧')
for t in trajs:
    print(f"  traj{t['label']}: birth={t['birth_frame']} death={t['death_frame']}")
print(f'每帧测量数: min={min(len(m) for m in meas)} max={max(len(m) for m in meas)} avg={np.mean([len(m) for m in meas]):.1f}')

max_meas = max(30, max(len(m) for m in meas))

print('\n=== Oracle 模式 ===')
r_oracle = evaluate_scenario(model, copy.deepcopy(scenario), 4, 20, max_meas, device, 0)
acc_o = np.mean(r_oracle['frame_assoc_acc'])
print(f'关联正确率均值: {acc_o*100:.1f}%')
n_pred = [len(p) for p in r_oracle['frame_pred_xyz']]
print(f'每帧预测目标数: min={min(n_pred)} max={max(n_pred)} avg={np.mean(n_pred):.1f}')

print('\n=== TrackManager + GRU 模式 ===')
r_mgr = evaluate_scenario_with_manager(
    model, copy.deepcopy(scenario), 4, 20, max_meas, device, 0,
    gru_model_path='lifecycle_gru.pth')
acc_m = np.mean(r_mgr['frame_assoc_acc'][4:])  # tau 后
print(f'关联正确率均值(tau后): {acc_m*100:.1f}%')
n_pred_m = [len(p) for p in r_mgr['frame_pred_xyz']]
print(f'每帧确认航迹数: min={min(n_pred_m)} max={max(n_pred_m)} avg={np.mean(n_pred_m):.1f}')
print('逐帧确认数(tau后):', n_pred_m[4:])
