import torch, numpy as np
from bait_model import BAIT
from data_generation_multi_scenario import MTTDataGeneratorMultiScenario
from evaluate_3d import COORD_SCALE, prepare_measurements
from track_manager import TrackManager, TrackManagerConfig

np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpt = torch.load('checkpoints_v3/best_model.pth', map_location=device, weights_only=False)
model = BAIT(d_model=256, nhead=8, max_targets=20).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

gen = MTTDataGeneratorMultiScenario(task_type=1, T=30.0, seed=42)
scenario = gen.generate_by_type('crossing')
trajs, meas, assoc = scenario
print(f'Targets: {len(trajs)}, Frames: {len(meas)}')
# GT labels and birth/death
gt_labels = {t['label'] for t in trajs}

for t in trajs:
    t['states'][:, :3] /= COORD_SCALE
for fm in meas:
    if len(fm):
        fm[:] = fm / COORD_SCALE

cfg = TrackManagerConfig(gru_model_path='lifecycle_gru.pth')
mgr = TrackManager(cfg)

tau = 4; max_targets = 20; max_measurements = 30; DT = 1.0; T_TOTAL = 30.0

# map track_id -> associated gt label (by proximity at confirmation)
track_gt_map = {}

with torch.no_grad():
    for frame_idx in range(len(meas)):
        meas_raw = meas[frame_idx]
        n_meas = min(len(meas_raw), max_measurements)
        meas_use = meas_raw[:n_meas]
        mgr.process_frame(meas_use, frame_idx)

        exist_np = None; match_np = None; filt_np = None

        if frame_idx >= tau:
            past_np, n_past = mgr.build_joint_past_states(frame_idx, tau, max_targets, DT, T_TOTAL)
            meas_pad = prepare_measurements(meas_use, max_measurements)
            pt  = torch.FloatTensor(past_np).unsqueeze(0).to(device)
            mt  = torch.FloatTensor(meas_pad).unsqueeze(0).to(device)
            npt = torch.LongTensor([n_past]).to(device)
            nmt = torch.LongTensor([n_meas]).to(device)
            match_pm, filt, exist_pr = model(pt, mt, npt, nmt)
            match_np = match_pm[0].cpu().numpy()
            exist_np = exist_pr[0].cpu().numpy()
            filt_np  = filt[0].cpu().numpy()
            mgr.bait_update(meas_use, match_np, exist_np, frame_idx,
                            filtered_states=filt_np, max_measurements=max_measurements)

        if frame_idx == 15 and exist_np is not None:
            # detailed dump at frame 15
            confirmed_sorted = sorted(mgr.get_confirmed_tracks(), key=lambda tr: tr.slot_no or 999)
            n_conf = len(confirmed_sorted)
            print(f'\n===== Frame 15 detail: {n_conf} confirmed tracks =====')
            print(f'  exist_np[:n_conf] = {exist_np[:n_conf]}')
            for ci, tr in enumerate(confirmed_sorted):
                positions = list(tr.positions.values())
                pos_last = positions[-1] if positions else None
                p = tr.p_alive if tr.p_alive is not None else -1
                ep = exist_np[ci] if ci < len(exist_np) else -1
                max_mp = float(np.max(match_np[:n_meas, ci+1])) if n_meas > 0 else -1
                # try to determine if real target by checking positions against gt states
                print(f'  slot={tr.slot_no:2d} ci={ci} ep={ep:.3f} max_mp={max_mp:.3f} p_alive={p:.2f}  miss={tr.miss_streak}')

n_conf = len(mgr.get_confirmed_tracks())
n_tent = sum(1 for t in mgr.tracks.values() if t.status == 'tentative')
print(f'\nFinal: conf={n_conf} tent={n_tent}')
