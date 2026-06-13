from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from data_generation import COORD_SCALE


@dataclass
class LTMBaitTrack:
    track_id: int
    x: np.ndarray
    P: np.ndarray
    history: dict = field(default_factory=dict)
    status: str = "tentative"
    slot: int | None = None
    birth_frame: int | None = None
    confirmed_frame: int | None = None
    last_frame: int | None = None
    miss_streak: int = 0
    hit_streak: int = 0
    total_hits: int = 0

    def add_state(self, frame_idx, xyz):
        self.history[int(frame_idx)] = np.asarray(xyz, dtype=np.float32).copy()
        self.last_frame = int(frame_idx)

    def age(self, frame_idx):
        if self.birth_frame is None:
            return 0
        return max(0, int(frame_idx) - int(self.birth_frame) + 1)


class LTMBaitTracker:
    def __init__(
        self,
        ltm_model,
        device,
        tau=4,
        max_targets=20,
        max_missed=3,
        gate_m=700.0,
        sigma_q_m=50.0,
        sigma_r_m=80.0,
        assoc_threshold=0.45,
        birth_threshold=0.45,
        confirm_threshold=0.45,
        death_threshold=0.65,
        min_confirm_hits=2,
        proposal_birth=True,
        dt=1.0,
    ):
        self.model = ltm_model
        self.device = device
        self.tau = int(tau)
        self.max_targets = int(max_targets)
        self.max_missed = int(max_missed)
        self.gate = float(gate_m) / COORD_SCALE
        self.assoc_threshold = float(assoc_threshold)
        self.birth_threshold = float(birth_threshold)
        self.confirm_threshold = float(confirm_threshold)
        self.death_threshold = float(death_threshold)
        self.min_confirm_hits = int(min_confirm_hits)
        self.proposal_birth = bool(proposal_birth)
        self.dt = float(dt)
        self.sigma_q = float(sigma_q_m) / COORD_SCALE
        self.sigma_r = float(sigma_r_m) / COORD_SCALE

        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[:3, :3] = np.eye(3)
        q = self.sigma_q ** 2
        self.Q = q * np.diag([0.25, 0.25, 0.25, 1.0, 1.0, 1.0])
        self.R = (self.sigma_r ** 2) * np.eye(3)

        self.tracks = {}
        self.next_track_id = 1
        self.next_slot = 0

    def _new_track(self, frame_idx, z):
        x = np.zeros(6, dtype=np.float64)
        x[:3] = z
        v_init = 500.0 / COORD_SCALE
        P = np.diag([self.sigma_r ** 2] * 3 + [v_init ** 2] * 3)
        tr = LTMBaitTrack(
            track_id=self.next_track_id,
            x=x,
            P=P,
            birth_frame=int(frame_idx),
            last_frame=int(frame_idx),
            hit_streak=1,
            total_hits=1,
        )
        self.next_track_id += 1
        tr.add_state(frame_idx, x[:3])
        self.tracks[tr.track_id] = tr

    def _predict_track(self, tr):
        tr.x = self.F @ tr.x
        tr.P = self.F @ tr.P @ self.F.T + self.Q

    def _update_track(self, tr, z):
        y = z - self.H @ tr.x
        S = self.H @ tr.P @ self.H.T + self.R
        K = tr.P @ self.H.T @ np.linalg.solve(S + np.eye(3) * 1e-12, np.eye(3))
        tr.x = tr.x + K @ y
        tr.P = (np.eye(6) - K @ self.H) @ tr.P

    def _track_features(self, tr, frame_idx):
        age = min(tr.age(frame_idx), 60) / 60.0
        hit = min(tr.hit_streak, 10) / 10.0
        miss = min(tr.miss_streak, 10) / 10.0
        confirmed = 1.0 if tr.status == "confirmed" else 0.0
        return np.array(
            [
                tr.x[0], tr.x[1], tr.x[2],
                tr.x[3], tr.x[4], tr.x[5],
                age, hit, miss, confirmed,
            ],
            dtype=np.float32,
        )

    def _meas_features(self, z):
        r = float(np.linalg.norm(z))
        return np.array([z[0], z[1], z[2], r, 1.0], dtype=np.float32)

    def _pair_features(self, tr, z, frame_idx):
        dz = z - tr.x[:3]
        dist = float(np.linalg.norm(dz))
        tf = self._track_features(tr, frame_idx)
        mf = self._meas_features(z)
        return np.array(
            [
                dz[0], dz[1], dz[2],
                abs(dz[0]), abs(dz[1]), abs(dz[2]),
                dist, dist / max(self.gate, 1e-6),
                tf[6], tf[7], tf[8], tf[9],
                tr.x[3], tr.x[4], tr.x[5],
                mf[3], self.gate, 1.0,
            ],
            dtype=np.float32,
        )

    def _score(self, pair_features=None, track_features=None, meas_features=None):
        with torch.no_grad():
            pair_prob = None
            confirm_prob = None
            death_prob = None
            birth_prob = None
            if pair_features is not None and len(pair_features) > 0:
                x = torch.as_tensor(pair_features, dtype=torch.float32, device=self.device)
                pair_prob = torch.sigmoid(self.model.score_pairs(x)).cpu().numpy()
            if track_features is not None and len(track_features) > 0:
                x = torch.as_tensor(track_features, dtype=torch.float32, device=self.device)
                c, d = self.model.score_track(x)
                confirm_prob = torch.sigmoid(c).cpu().numpy()
                death_prob = torch.sigmoid(d).cpu().numpy()
            if meas_features is not None and len(meas_features) > 0:
                x = torch.as_tensor(meas_features, dtype=torch.float32, device=self.device)
                birth_prob = torch.sigmoid(self.model.score_birth(x)).cpu().numpy()
        return pair_prob, confirm_prob, death_prob, birth_prob

    def step(self, frame_idx, measurements_norm):
        measurements = [np.asarray(z, dtype=np.float64) for z in measurements_norm]
        live_tracks = [tr for tr in self.tracks.values() if tr.status in ("tentative", "confirmed")]

        for tr in live_tracks:
            if tr.last_frame is not None and tr.last_frame < frame_idx:
                self._predict_track(tr)

        assigned_tracks = set()
        assigned_meas = set()
        if live_tracks and measurements:
            pair_rows = []
            pair_index = []
            for ti, tr in enumerate(live_tracks):
                for mi, z in enumerate(measurements):
                    pair_rows.append(self._pair_features(tr, z, frame_idx))
                    pair_index.append((ti, mi))
            pair_prob, _, _, _ = self._score(pair_features=np.asarray(pair_rows, dtype=np.float32))
            score = np.full((len(live_tracks), len(measurements)), -1e6, dtype=np.float64)
            dists = np.zeros_like(score)
            for k, (ti, mi) in enumerate(pair_index):
                dist = np.linalg.norm(measurements[mi] - live_tracks[ti].x[:3])
                dists[ti, mi] = dist
                if dist <= self.gate:
                    score[ti, mi] = float(pair_prob[k])
            rows, cols = linear_sum_assignment(-score)
            for r, c in zip(rows, cols):
                if score[r, c] < self.assoc_threshold or dists[r, c] > self.gate:
                    continue
                tr = live_tracks[r]
                self._update_track(tr, measurements[c])
                tr.add_state(frame_idx, tr.x[:3])
                tr.miss_streak = 0
                tr.hit_streak += 1
                tr.total_hits += 1
                assigned_tracks.add(tr.track_id)
                assigned_meas.add(c)

        for tr in live_tracks:
            if tr.track_id in assigned_tracks:
                continue
            tr.miss_streak += 1
            tr.hit_streak = 0
            tr.add_state(frame_idx, tr.x[:3])

        live_tracks = [tr for tr in self.tracks.values() if tr.status in ("tentative", "confirmed")]
        track_features = np.asarray(
            [self._track_features(tr, frame_idx) for tr in live_tracks],
            dtype=np.float32,
        ) if live_tracks else np.empty((0, 10), dtype=np.float32)
        _, confirm_prob, death_prob, _ = self._score(track_features=track_features)
        for i, tr in enumerate(live_tracks):
            if death_prob is not None and death_prob[i] >= self.death_threshold:
                tr.status = "dead"
                continue
            if tr.miss_streak > self.max_missed:
                tr.status = "dead"
                continue
            if tr.status != "tentative" or self.next_slot >= self.max_targets:
                continue
            has_window = all(t in tr.history for t in range(frame_idx - self.tau + 1, frame_idx + 1))
            enough_hits = tr.total_hits >= self.min_confirm_hits
            kf_like_confirm = tr.hit_streak >= self.tau and tr.total_hits >= self.tau
            learned_confirm = confirm_prob is not None and confirm_prob[i] >= self.confirm_threshold
            if has_window and enough_hits and (learned_confirm or kf_like_confirm):
                tr.status = "confirmed"
                tr.confirmed_frame = int(frame_idx)
                tr.slot = self.next_slot
                self.next_slot += 1

        unmatched = [i for i in range(len(measurements)) if i not in assigned_meas]
        if unmatched:
            meas_features = np.asarray(
                [self._meas_features(measurements[i]) for i in unmatched],
                dtype=np.float32,
            )
            _, _, _, birth_prob = self._score(meas_features=meas_features)
            for local_i, mi in enumerate(unmatched):
                learned_birth = birth_prob is not None and birth_prob[local_i] >= self.birth_threshold
                if self.proposal_birth or learned_birth:
                    self._new_track(frame_idx, measurements[mi])

    def confirmed_tracks(self):
        return [
            tr for tr in self.tracks.values()
            if tr.status == "confirmed" and tr.slot is not None
        ]

    def build_past_states(self, frame_idx, dt=1.0, total_time=30.0):
        past = []
        num_past = []
        by_slot = {tr.slot: tr for tr in self.confirmed_tracks()}
        for t in range(frame_idx - self.tau, frame_idx):
            frame_states = [np.zeros(5, dtype=np.float32) for _ in range(self.max_targets)]
            for slot, tr in by_slot.items():
                xyz = tr.history.get(t)
                if xyz is None:
                    continue
                frame_states[slot] = np.array(
                    [slot + 1, xyz[0], xyz[1], xyz[2], (t * dt) / total_time],
                    dtype=np.float32,
                )
            num_past.append(sum(1 for state in frame_states if state[0] > 0))
            past.extend(frame_states)
        return np.array(past, dtype=np.float32), np.array(num_past, dtype=np.int64)

    def apply_bait_outputs(self, frame_idx, filtered_states, existence_probs=None, exist_threshold=0.3):
        for tr in self.confirmed_tracks():
            if tr.slot is None or tr.slot >= len(filtered_states):
                continue
            if existence_probs is not None and existence_probs[tr.slot] < exist_threshold:
                tr.miss_streak += 1
                if tr.miss_streak > self.max_missed:
                    tr.status = "dead"
                continue
            xyz = np.asarray(filtered_states[tr.slot], dtype=np.float64)
            tr.x[:3] = xyz
            tr.add_state(frame_idx, xyz)
            tr.miss_streak = 0
