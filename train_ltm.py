import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from data_generation import COORD_SCALE
from data_generation_multi_scenario import MTTDataGeneratorMultiScenario, SCENARIO_TYPES
from ltm_bait_tracker import LTMBaitTracker, LTMBaitTrack
from ltm_model import LTMScorer


@dataclass
class TeacherTrack:
    x: np.ndarray
    P: np.ndarray
    birth_frame: int
    last_frame: int
    gt_label: int = 0
    status: str = "tentative"
    miss_streak: int = 0
    hit_streak: int = 1
    total_hits: int = 1

    def age(self, frame_idx):
        return int(frame_idx) - int(self.birth_frame) + 1


class TeacherFeatureBuilder:
    def __init__(self, tau=4, gate_m=700.0, sigma_r_m=80.0, sigma_q_m=50.0):
        self.tau = int(tau)
        self.gate = float(gate_m) / COORD_SCALE
        self.sigma_r = float(sigma_r_m) / COORD_SCALE
        self.sigma_q = float(sigma_q_m) / COORD_SCALE
        self.dt = 1.0
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[:3, :3] = np.eye(3)
        q = self.sigma_q ** 2
        self.Q = q * np.diag([0.25, 0.25, 0.25, 1.0, 1.0, 1.0])
        self.R = (self.sigma_r ** 2) * np.eye(3)
        self.tracks = []

        self.feature_proxy = LTMBaitTracker(
            ltm_model=LTMScorer(), device=torch.device("cpu"),
            tau=tau, gate_m=gate_m, sigma_r_m=sigma_r_m, sigma_q_m=sigma_q_m,
        )

    def _predict(self, tr):
        tr.x = self.F @ tr.x
        tr.P = self.F @ tr.P @ self.F.T + self.Q

    def _update(self, tr, z):
        y = z - self.H @ tr.x
        S = self.H @ tr.P @ self.H.T + self.R
        K = tr.P @ self.H.T @ np.linalg.solve(S + np.eye(3) * 1e-12, np.eye(3))
        tr.x = tr.x + K @ y
        tr.P = (np.eye(6) - K @ self.H) @ tr.P

    def _new_track(self, frame_idx, z, gt_label):
        x = np.zeros(6, dtype=np.float64)
        x[:3] = z
        v_init = 500.0 / COORD_SCALE
        P = np.diag([self.sigma_r ** 2] * 3 + [v_init ** 2] * 3)
        self.tracks.append(
            TeacherTrack(
                x=x, P=P, birth_frame=int(frame_idx), last_frame=int(frame_idx),
                gt_label=int(gt_label), hit_streak=1, total_hits=1,
            )
        )

    def _proxy_track(self, tr):
        return LTMBaitTrack(
            track_id=0, x=tr.x.copy(), P=tr.P.copy(), status=tr.status,
            birth_frame=tr.birth_frame, last_frame=tr.last_frame,
            miss_streak=tr.miss_streak, hit_streak=tr.hit_streak,
            total_hits=tr.total_hits,
        )

    def collect(self, scenario):
        trajectories, measurements, associations = scenario
        for traj in trajectories:
            traj["states"][:, :3] /= COORD_SCALE
        for fm in measurements:
            if len(fm) > 0:
                fm[:] = fm / COORD_SCALE

        pair_x, pair_y = [], []
        meas_x, birth_y = [], []
        track_x, confirm_y, death_y = [], [], []

        active_by_frame = []
        for t in range(len(measurements)):
            active_by_frame.append({
                int(tr["label"]) for tr in trajectories
                if int(tr["birth_frame"]) <= t <= int(tr["death_frame"])
            })

        for frame_idx, meas_frame in enumerate(measurements):
            meas = [np.asarray(z, dtype=np.float64) for z in meas_frame]
            assoc = [int(a) for a in associations[frame_idx]]
            live = [tr for tr in self.tracks if tr.status != "dead"]
            for tr in live:
                if tr.last_frame < frame_idx:
                    self._predict(tr)

            labels_with_live_track = {tr.gt_label for tr in live if tr.gt_label > 0}

            for tr in live:
                p = self._proxy_track(tr)
                track_x.append(self.feature_proxy._track_features(p, frame_idx))
                is_real = tr.gt_label > 0 and tr.gt_label in active_by_frame[frame_idx]
                confirm_y.append(float(is_real and tr.age(frame_idx) >= self.tau and tr.total_hits >= 2))
                death_y.append(float((tr.gt_label == 0 and tr.age(frame_idx) >= self.tau) or
                                     (tr.gt_label > 0 and tr.gt_label not in active_by_frame[frame_idx])))

            for mi, z in enumerate(meas):
                meas_x.append(self.feature_proxy._meas_features(z))
                birth_y.append(float(assoc[mi] > 0 and assoc[mi] not in labels_with_live_track))

            for tr in live:
                p = self._proxy_track(tr)
                for mi, z in enumerate(meas):
                    pair_x.append(self.feature_proxy._pair_features(p, z, frame_idx))
                    pair_y.append(float(tr.gt_label > 0 and mi < len(assoc) and assoc[mi] == tr.gt_label))

            assigned_tracks = set()
            assigned_meas = set()
            real_tracks = [tr for tr in live if tr.gt_label > 0]
            if real_tracks and meas:
                cost = np.full((len(real_tracks), len(meas)), 1e6, dtype=np.float64)
                for ti, tr in enumerate(real_tracks):
                    for mi, z in enumerate(meas):
                        if assoc[mi] == tr.gt_label:
                            cost[ti, mi] = np.linalg.norm(z - tr.x[:3])
                rows, cols = linear_sum_assignment(cost)
                for r, c in zip(rows, cols):
                    if cost[r, c] >= 1e6:
                        continue
                    tr = real_tracks[r]
                    self._update(tr, meas[c])
                    tr.last_frame = frame_idx
                    tr.miss_streak = 0
                    tr.hit_streak += 1
                    tr.total_hits += 1
                    assigned_tracks.add(id(tr))
                    assigned_meas.add(c)

            for tr in live:
                if id(tr) in assigned_tracks:
                    continue
                tr.miss_streak += 1
                tr.hit_streak = 0
                tr.last_frame = frame_idx
                if tr.miss_streak > 5:
                    tr.status = "dead"

            current_track_labels = {tr.gt_label for tr in live if tr.status != "dead" and tr.gt_label > 0}
            for mi, z in enumerate(meas):
                if mi in assigned_meas:
                    continue
                gt_label = assoc[mi] if assoc[mi] > 0 and assoc[mi] not in current_track_labels else 0
                self._new_track(frame_idx, z, gt_label)

        return pair_x, pair_y, meas_x, birth_y, track_x, confirm_y, death_y


def sample_batch(buffers, batch_size, device):
    pair_x, pair_y, meas_x, birth_y, track_x, confirm_y, death_y = buffers
    out = {}
    if pair_x:
        idx = np.random.randint(0, len(pair_x), size=batch_size)
        out["pair_x"] = torch.as_tensor(np.asarray([pair_x[i] for i in idx]), dtype=torch.float32, device=device)
        out["pair_y"] = torch.as_tensor(np.asarray([pair_y[i] for i in idx]), dtype=torch.float32, device=device)
    if meas_x:
        idx = np.random.randint(0, len(meas_x), size=batch_size)
        out["meas_x"] = torch.as_tensor(np.asarray([meas_x[i] for i in idx]), dtype=torch.float32, device=device)
        out["birth_y"] = torch.as_tensor(np.asarray([birth_y[i] for i in idx]), dtype=torch.float32, device=device)
    if track_x:
        idx = np.random.randint(0, len(track_x), size=batch_size)
        out["track_x"] = torch.as_tensor(np.asarray([track_x[i] for i in idx]), dtype=torch.float32, device=device)
        out["confirm_y"] = torch.as_tensor(np.asarray([confirm_y[i] for i in idx]), dtype=torch.float32, device=device)
        out["death_y"] = torch.as_tensor(np.asarray([death_y[i] for i in idx]), dtype=torch.float32, device=device)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--scenarios", type=int, default=400)
    parser.add_argument("--frames", type=int, default=None,
                        help="Use a fixed frame count. If omitted, frames are sampled from [--min-frames, --max-frames].")
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", type=str, default="checkpoints_ltm")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.frames is not None:
        frame_desc = str(args.frames)
    else:
        if args.min_frames > args.max_frames:
            raise ValueError("--min-frames must be <= --max-frames")
        frame_desc = f"random[{args.min_frames},{args.max_frames}]"

    print(f"Generating LTM training buffers: scenarios={args.scenarios}, frames={frame_desc}")
    buffers = ([], [], [], [], [], [], [])
    for i in range(args.scenarios):
        if args.frames is not None:
            n_frames = int(args.frames)
        else:
            n_frames = random.randint(int(args.min_frames), int(args.max_frames))
        gen = MTTDataGeneratorMultiScenario(task_type=1, T=float(n_frames), seed=None)
        scene_type = random.choice(SCENARIO_TYPES)
        scenario = gen.generate_by_type(scene_type)
        builder = TeacherFeatureBuilder(tau=4)
        sample = builder.collect(scenario)
        for dst, src in zip(buffers, sample):
            dst.extend(src)
        if (i + 1) % 50 == 0:
            print(
                f"  {i+1}/{args.scenarios} scenarios | "
                f"last_frames={n_frames} | pairs={len(buffers[0])} tracks={len(buffers[4])}"
            )

    model_kwargs = dict(hidden_dim=128, dropout=0.1)
    model = LTMScorer(**model_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float("inf")

    for step in range(1, args.steps + 1):
        batch = sample_batch(buffers, args.batch_size, device)
        opt.zero_grad(set_to_none=True)
        losses = []
        if "pair_x" in batch:
            logits = model.score_pairs(batch["pair_x"])
            losses.append(F.binary_cross_entropy_with_logits(logits, batch["pair_y"], pos_weight=torch.tensor(8.0, device=device)))
        if "meas_x" in batch:
            logits = model.score_birth(batch["meas_x"])
            losses.append(F.binary_cross_entropy_with_logits(logits, batch["birth_y"], pos_weight=torch.tensor(4.0, device=device)))
        if "track_x" in batch:
            confirm_logits, death_logits = model.score_track(batch["track_x"])
            losses.append(F.binary_cross_entropy_with_logits(confirm_logits, batch["confirm_y"], pos_weight=torch.tensor(2.0, device=device)))
            losses.append(F.binary_cross_entropy_with_logits(death_logits, batch["death_y"], pos_weight=torch.tensor(4.0, device=device)))
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
                    os.path.join(args.save_dir, "best_ltm.pth"),
                )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": model_kwargs,
            "step": args.steps,
            "loss": float(loss.item()),
        },
        os.path.join(args.save_dir, "last_ltm.pth"),
    )
    print(f"Saved LTM checkpoints to {args.save_dir}")


if __name__ == "__main__":
    main()
