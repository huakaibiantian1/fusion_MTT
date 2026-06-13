from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from data_generation import COORD_SCALE


@dataclass
class KFBAITTrack:
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

    def add_state(self, frame_idx, xyz):
        self.history[int(frame_idx)] = np.asarray(xyz, dtype=np.float32).copy()
        self.last_frame = int(frame_idx)


class KFBAITTracker:
    def __init__(
        self,
        tau=4,
        max_targets=20,
        max_missed=2,
        confirm_hits=None,
        gate_m=500.0,
        sigma_q_m=50.0,
        sigma_r_m=80.0,
        dt=1.0,
    ):
        self.tau = int(tau)
        self.max_targets = int(max_targets)
        self.max_missed = int(max_missed)
        self.confirm_hits = int(confirm_hits if confirm_hits is not None else tau)
        self.gate = float(gate_m) / COORD_SCALE
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
        tr = KFBAITTrack(
            track_id=self.next_track_id,
            x=x,
            P=P,
            birth_frame=int(frame_idx),
            last_frame=int(frame_idx),
            hit_streak=1,
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

    def step(self, frame_idx, measurements_norm):
        measurements = [np.asarray(z, dtype=np.float64) for z in measurements_norm]
        live_tracks = [tr for tr in self.tracks.values() if tr.status in ("tentative", "confirmed")]

        for tr in live_tracks:
            if tr.last_frame is not None and tr.last_frame < frame_idx:
                self._predict_track(tr)

        assigned_tracks = set()
        assigned_meas = set()
        if live_tracks and measurements:
            pred = np.array([tr.x[:3] for tr in live_tracks])
            meas = np.array(measurements)
            cost = np.linalg.norm(pred[:, None, :] - meas[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] > self.gate:
                    continue
                tr = live_tracks[r]
                self._update_track(tr, measurements[c])
                tr.add_state(frame_idx, tr.x[:3])
                tr.miss_streak = 0
                tr.hit_streak += 1
                assigned_tracks.add(tr.track_id)
                assigned_meas.add(c)

        for tr in live_tracks:
            if tr.track_id in assigned_tracks:
                continue
            tr.miss_streak += 1
            tr.hit_streak = 0
            if tr.status == "confirmed":
                tr.add_state(frame_idx, tr.x[:3])
            if tr.miss_streak > self.max_missed:
                tr.status = "dead"

        for i, z in enumerate(measurements):
            if i not in assigned_meas:
                self._new_track(frame_idx, z)

        self._confirm_ready_tracks(frame_idx)

    def _confirm_ready_tracks(self, frame_idx):
        tentative = sorted(
            [tr for tr in self.tracks.values() if tr.status == "tentative"],
            key=lambda tr: (tr.birth_frame, tr.track_id),
        )
        for tr in tentative:
            if self.next_slot >= self.max_targets:
                break
            if tr.hit_streak < self.confirm_hits:
                continue
            needed = range(frame_idx - self.tau + 1, frame_idx + 1)
            if not all(t in tr.history for t in needed):
                continue
            tr.status = "confirmed"
            tr.confirmed_frame = int(frame_idx)
            tr.slot = self.next_slot
            self.next_slot += 1

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

