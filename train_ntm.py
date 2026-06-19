import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from data_generation import COORD_SCALE
from data_generation_multi_scenario import MTTDataGeneratorMultiScenario, SCENARIO_TYPES
from ntm_model import NTMModel


def track_features(state, age, hit=1.0, miss=0.0, confirmed=0.0):
    return np.array(
        [
            state[0], state[1], state[2],
            state[3], state[4], state[5],
            min(age, 80) / 80.0,
            min(hit, 10) / 10.0,
            min(miss, 10) / 10.0,
            confirmed,
        ],
        dtype=np.float32,
    )


def meas_features(z):
    r = float(np.linalg.norm(z))
    return np.array([z[0], z[1], z[2], r, 1.0], dtype=np.float32)


def pair_features(state, tf, z, gate_m=900.0):
    gate = float(gate_m) / COORD_SCALE
    dz = z - state[:3]
    dist = float(np.linalg.norm(dz))
    mf = meas_features(z)
    return np.array(
        [
            dz[0], dz[1], dz[2],
            abs(dz[0]), abs(dz[1]), abs(dz[2]),
            dist, dist / max(gate, 1e-6),
            tf[0], tf[1], tf[2],
            tf[3], tf[4], tf[5],
            tf[6], tf[7], tf[8], tf[9],
            mf[3], 1.0,
        ],
        dtype=np.float32,
    )


def state6_from_traj(traj, t):
    s = traj["states"][t].copy().astype(np.float64)
    s[:3] /= COORD_SCALE
    s[3:6] /= COORD_SCALE
    return s[:6]


def collect_scenario_buffers(scenario, tau=4):
    trajectories, measurements, associations = scenario
    for fm in measurements:
        if len(fm) > 0:
            fm[:] = fm / COORD_SCALE

    pair_x, pair_y, update_y, update_mask = [], [], [], []
    meas_x, birth_y = [], []
    track_x, motion_y, confirm_y, death_y = [], [], [], []

    by_label = {int(tr["label"]): tr for tr in trajectories}
    num_frames = len(measurements)

    for t in range(num_frames):
        meas = measurements[t]
        assoc = associations[t]

        for mi, z in enumerate(meas):
            meas_x.append(meas_features(z))
            lbl = int(assoc[mi]) if mi < len(assoc) else 0
            is_birth = False
            if lbl > 0 and lbl in by_label:
                is_birth = t <= int(by_label[lbl]["birth_frame"]) + 1
            birth_y.append(float(is_birth))

        for tr in trajectories:
            label = int(tr["label"])
            birth = int(tr["birth_frame"])
            death = int(tr["death_frame"])
            if not (birth < t <= death):
                continue

            prev_state = state6_from_traj(tr, t - 1)
            cur_state = state6_from_traj(tr, t)
            age = t - birth + 1
            confirmed = 1.0 if age >= tau else 0.0
            tf = track_features(prev_state, age=age, hit=min(age, 10), miss=0, confirmed=confirmed)

            track_x.append(tf)
            motion_y.append((cur_state - prev_state).astype(np.float32))
            confirm_y.append(float(age >= tau))
            death_y.append(0.0)

            for mi, z in enumerate(meas):
                px = pair_features(prev_state, tf, z)
                same = int(assoc[mi]) == label if mi < len(assoc) else False
                pair_x.append(px)
                pair_y.append(float(same))
                update_y.append((cur_state - prev_state).astype(np.float32))
                update_mask.append(float(same))

        for tr in trajectories:
            label = int(tr["label"])
            death = int(tr["death_frame"])
            if t != death + 1 or death <= int(tr["birth_frame"]):
                continue
            last_state = state6_from_traj(tr, death)
            tf = track_features(last_state, age=death - int(tr["birth_frame"]) + 1, hit=0, miss=1, confirmed=1.0)
            track_x.append(tf)
            motion_y.append(np.zeros(6, dtype=np.float32))
            confirm_y.append(0.0)
            death_y.append(1.0)

    return pair_x, pair_y, update_y, update_mask, meas_x, birth_y, track_x, motion_y, confirm_y, death_y


def sample_tensor(xs, ys=None, batch_size=512, device="cpu"):
    idx = np.random.randint(0, len(xs), size=batch_size)
    x = torch.as_tensor(np.asarray([xs[i] for i in idx]), dtype=torch.float32, device=device)
    if ys is None:
        return x
    y = torch.as_tensor(np.asarray([ys[i] for i in idx]), dtype=torch.float32, device=device)
    return x, y


def sample_multi(xs, y_lists, batch_size=512, device="cpu"):
    idx = np.random.randint(0, len(xs), size=batch_size)
    x = torch.as_tensor(np.asarray([xs[i] for i in idx]), dtype=torch.float32, device=device)
    ys = [
        torch.as_tensor(np.asarray([items[i] for i in idx]), dtype=torch.float32, device=device)
        for items in y_lists
    ]
    return x, ys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--scenarios", type=int, default=600)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", type=str, default="checkpoints_ntm")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame_desc = str(args.frames) if args.frames is not None else f"random[{args.min_frames},{args.max_frames}]"
    print(f"Generating NTM buffers: scenarios={args.scenarios}, frames={frame_desc}")
    buffers = tuple([] for _ in range(10))
    for i in range(args.scenarios):
        n_frames = int(args.frames) if args.frames is not None else random.randint(args.min_frames, args.max_frames)
        gen = MTTDataGeneratorMultiScenario(task_type=1, T=float(n_frames), seed=None)
        scenario = gen.generate_by_type(random.choice(SCENARIO_TYPES))
        sample = collect_scenario_buffers(scenario)
        for dst, src in zip(buffers, sample):
            dst.extend(src)
        if (i + 1) % 50 == 0:
            print(
                f"  {i+1}/{args.scenarios} | last_frames={n_frames} | "
                f"pairs={len(buffers[0])} tracks={len(buffers[6])}"
            )

    pair_x, pair_y, update_y, update_mask, meas_x, birth_y, track_x, motion_y, confirm_y, death_y = buffers
    model_kwargs = dict(hidden_dim=128, dropout=0.1)
    model = NTMModel(**model_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float("inf")

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        losses = []

        if pair_x:
            x_pair, (y_pair, y_update, y_update_mask) = sample_multi(
                pair_x, [pair_y, update_y, update_mask], args.batch_size, device
            )
            pair_logits = model.score_pairs(x_pair)
            update_pred = model.update_state_delta(x_pair)
            losses.append(F.binary_cross_entropy_with_logits(
                pair_logits, y_pair, pos_weight=torch.tensor(8.0, device=device)
            ))
            update_loss = F.smooth_l1_loss(update_pred, y_update, reduction="none").mean(dim=1)
            losses.append((update_loss * y_update_mask).sum() / y_update_mask.sum().clamp_min(1.0))

        if track_x:
            x_track, (y_motion, y_confirm, y_death) = sample_multi(
                track_x, [motion_y, confirm_y, death_y], args.batch_size, device
            )
            motion_pred = model.predict_motion(x_track)
            confirm_logits, death_logits = model.score_lifecycle(x_track)
            losses.append(F.smooth_l1_loss(motion_pred, y_motion))
            losses.append(F.binary_cross_entropy_with_logits(
                confirm_logits, y_confirm, pos_weight=torch.tensor(2.0, device=device)
            ))
            losses.append(F.binary_cross_entropy_with_logits(
                death_logits, y_death, pos_weight=torch.tensor(4.0, device=device)
            ))

        if meas_x:
            x_meas, y_birth = sample_tensor(meas_x, birth_y, args.batch_size, device)
            birth_logits = model.score_birth(x_meas)
            losses.append(F.binary_cross_entropy_with_logits(
                birth_logits, y_birth, pos_weight=torch.tensor(4.0, device=device)
            ))

        loss = sum(losses)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step == 1 or step % 100 == 0:
            val = float(loss.item())
            print(f"step {step:6d}/{args.steps} | loss={val:.4f}")
            if val < best_loss:
                best_loss = val
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_kwargs": model_kwargs,
                        "step": step,
                        "loss": best_loss,
                    },
                    os.path.join(args.save_dir, "best_ntm.pth"),
                )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": model_kwargs,
            "step": args.steps,
            "loss": float(loss.item()),
        },
        os.path.join(args.save_dir, "last_ntm.pth"),
    )
    print(f"Saved NTM checkpoints to {args.save_dir}")


if __name__ == "__main__":
    main()
