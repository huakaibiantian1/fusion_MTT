from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from data_generation import COORD_SCALE
from phd_filter import GMPHDFilter


@dataclass
class PHDBAITTrack:
    track_id: int
    history: dict = field(default_factory=dict)
    status: str = "tentative"
    slot: int | None = None
    birth_frame: int | None = None
    confirmed_frame: int | None = None
    last_frame: int | None = None
    miss_streak: int = 0

    def add_state(self, frame_idx, xyz):
        self.history[int(frame_idx)] = np.asarray(xyz, dtype=np.float32).copy()
        self.last_frame = int(frame_idx)
        self.miss_streak = 0


class PHDBAITTracker:
    def __init__(
        self,
        tau=4,
        max_targets=20,
        max_missed=2,
        assign_gate_m=300.0,
        phd_kwargs=None,
    ):
        self.tau = int(tau)
        self.max_targets = int(max_targets)
        self.max_missed = int(max_missed)
        self.assign_gate = float(assign_gate_m) / COORD_SCALE
        self.phd = GMPHDFilter(**(phd_kwargs or {}))
        self.tracks = {}
        self.next_track_id = 1
        self.next_slot = 0

    def step_phd(self, measurements_norm):
        self.phd.predict()
        z_list = [np.asarray(z, dtype=np.float64) for z in measurements_norm]
        self.phd.update(z_list)
        self.phd.prune_and_merge()
        return self.phd.extract_states()

    def update_tracks_from_phd(self, frame_idx, estimates):
        estimates = [np.asarray(e, dtype=np.float32) for e in estimates]
        live_tracks = [
            tr for tr in self.tracks.values()
            if tr.status in ("tentative", "confirmed")
            and tr.last_frame is not None
        ]

        assigned_tracks = set()
        assigned_estimates = set()

        if live_tracks and estimates:
            prev = np.array([tr.history[tr.last_frame] for tr in live_tracks])
            est = np.array(estimates)
            cost = np.linalg.norm(prev[:, None, :] - est[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] > self.assign_gate:
                    continue
                tr = live_tracks[r]
                tr.add_state(frame_idx, estimates[c])
                assigned_tracks.add(tr.track_id)
                assigned_estimates.add(c)

        for tr in live_tracks:
            if tr.track_id not in assigned_tracks:
                tr.miss_streak += 1
                if tr.miss_streak > self.max_missed:
                    tr.status = "dead"

        for i, xyz in enumerate(estimates):
            if i in assigned_estimates:
                continue
            tr = PHDBAITTrack(
                track_id=self.next_track_id,
                birth_frame=int(frame_idx),
            )
            self.next_track_id += 1
            tr.add_state(frame_idx, xyz)
            self.tracks[tr.track_id] = tr

        self._confirm_ready_tracks(frame_idx)

    def _confirm_ready_tracks(self, frame_idx):
        tentative = sorted(
            [tr for tr in self.tracks.values() if tr.status == "tentative"],
            key=lambda tr: (tr.birth_frame, tr.track_id),
        )
        for tr in tentative:
            if self.next_slot >= self.max_targets:
                break
            if tr.birth_frame is None:
                continue
            if frame_idx - tr.birth_frame + 1 < self.tau:
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
        confirmed = self.confirmed_tracks()
        by_slot = {tr.slot: tr for tr in confirmed}

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
            tr.add_state(frame_idx, filtered_states[tr.slot])

