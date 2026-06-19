from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from data_generation import COORD_SCALE


@dataclass
class NTMBaitTrack:
    track_id: int
    state: np.ndarray
    history: dict = field(default_factory=dict)
    status: str = "tentative"
    slot: int | None = None
    birth_frame: int | None = None
    confirmed_frame: int | None = None
    last_frame: int | None = None
    miss_streak: int = 0
    death_streak: int = 0
    hit_streak: int = 0
    total_hits: int = 0
    confidence: float = 0.5

    def add_state(self, frame_idx, xyz):
        self.history[int(frame_idx)] = np.asarray(xyz, dtype=np.float32).copy()
        self.last_frame = int(frame_idx)

    def age(self, frame_idx):
        if self.birth_frame is None:
            return 0
        return max(0, int(frame_idx) - int(self.birth_frame) + 1)


class NTMBaitTracker:
    def __init__(
        self,
        ntm_model,
        device,
        tau=4,
        max_targets=20,
        max_missed=4,
        gate_m=900.0,
        sigma_q_m=None,
        sigma_r_m=None,
        assoc_threshold=0.35,
        birth_threshold=0.35,
        confirm_threshold=0.35,
        death_threshold=0.75,
        min_confirm_hits=4,
        dt=1.0,
        proposal_birth=True,
        max_candidate_tracks=40,
        birth_nms_m=250.0,
        max_births_per_frame=8,
        bait_update_gate_m=1000.0,
        confirm_nms_m=350.0,
        min_confirm_hit_streak=4,
        confirmed_gate_scale=1.5,
        confirmed_min_max_missed=6,
        confirmed_death_streak=2,
        confirmed_protect_frames=3,
    ):
        self.model = ntm_model
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
        self.dt = float(dt)
        self.proposal_birth = bool(proposal_birth)
        self.max_candidate_tracks = int(max_candidate_tracks)
        self.birth_nms = float(birth_nms_m) / COORD_SCALE
        self.max_births_per_frame = int(max_births_per_frame)
        self.bait_update_gate = float(bait_update_gate_m) / COORD_SCALE
        self.confirm_nms = float(confirm_nms_m) / COORD_SCALE
        self.min_confirm_hit_streak = int(min_confirm_hit_streak)
        self.confirmed_gate_scale = float(confirmed_gate_scale)
        self.confirmed_max_missed = max(int(max_missed), int(confirmed_min_max_missed))
        self.confirmed_death_streak = int(confirmed_death_streak)
        self.confirmed_protect_frames = int(confirmed_protect_frames)

        self.tracks = {}
        self.next_track_id = 1
        self.next_slot = 0

    def _new_track(self, frame_idx, z, birth_prob=0.5):
        live_count = sum(
            1 for tr in self.tracks.values()
            if tr.status in ("tentative", "confirmed")
        )
        if live_count >= self.max_candidate_tracks:
            return
        state = np.zeros(6, dtype=np.float64)
        state[:3] = z
        tr = NTMBaitTrack(
            track_id=self.next_track_id,
            state=state,
            birth_frame=int(frame_idx),
            last_frame=int(frame_idx),
            hit_streak=1,
            total_hits=1,
            confidence=float(birth_prob),
        )
        self.next_track_id += 1
        tr.add_state(frame_idx, state[:3])
        self.tracks[tr.track_id] = tr

    def _track_features(self, tr, frame_idx):
        age = min(tr.age(frame_idx), 80) / 80.0
        hit = min(tr.hit_streak, 10) / 10.0
        miss = min(tr.miss_streak, 10) / 10.0
        confirmed = 1.0 if tr.status == "confirmed" else 0.0
        return np.array(
            [
                tr.state[0], tr.state[1], tr.state[2],
                tr.state[3], tr.state[4], tr.state[5],
                age, hit, miss, confirmed,
            ],
            dtype=np.float32,
        )

    def _meas_features(self, z):
        r = float(np.linalg.norm(z))
        return np.array([z[0], z[1], z[2], r, 1.0], dtype=np.float32)

    def _pair_features(self, tr, z, frame_idx):
        dz = z - tr.state[:3]
        dist = float(np.linalg.norm(dz))
        tf = self._track_features(tr, frame_idx)
        mf = self._meas_features(z)
        return np.array(
            [
                dz[0], dz[1], dz[2],
                abs(dz[0]), abs(dz[1]), abs(dz[2]),
                dist, dist / max(self.gate, 1e-6),
                tf[0], tf[1], tf[2],
                tf[3], tf[4], tf[5],
                tf[6], tf[7], tf[8], tf[9],
                mf[3], 1.0,
            ],
            dtype=np.float32,
        )

    def _score(self, track_features=None, pair_features=None, meas_features=None):
        with torch.no_grad():
            motion = confirm = death = pair = update = birth = None
            if track_features is not None and len(track_features) > 0:
                x = torch.as_tensor(track_features, dtype=torch.float32, device=self.device)
                motion = self.model.predict_motion(x).cpu().numpy()
                c, d = self.model.score_lifecycle(x)
                confirm = torch.sigmoid(c).cpu().numpy()
                death = torch.sigmoid(d).cpu().numpy()
            if pair_features is not None and len(pair_features) > 0:
                x = torch.as_tensor(pair_features, dtype=torch.float32, device=self.device)
                pair = torch.sigmoid(self.model.score_pairs(x)).cpu().numpy()
                update = self.model.update_state_delta(x).cpu().numpy()
            if meas_features is not None and len(meas_features) > 0:
                x = torch.as_tensor(meas_features, dtype=torch.float32, device=self.device)
                birth = torch.sigmoid(self.model.score_birth(x)).cpu().numpy()
        return motion, confirm, death, pair, update, birth

    def step(self, frame_idx, measurements_norm):
        measurements = [np.asarray(z, dtype=np.float64) for z in measurements_norm]
        live_tracks = [tr for tr in self.tracks.values() if tr.status in ("tentative", "confirmed")]

        if live_tracks:
            track_features = np.asarray(
                [self._track_features(tr, frame_idx) for tr in live_tracks],
                dtype=np.float32,
            )
            motion, _, _, _, _, _ = self._score(track_features=track_features)
            for i, tr in enumerate(live_tracks):
                if tr.last_frame is not None and tr.last_frame < frame_idx:
                    delta = motion[i] if motion is not None else np.zeros(6)
                    tr.state = tr.state + delta.astype(np.float64)
                    tr.add_state(frame_idx, tr.state[:3])

        assigned_tracks = set()
        assigned_meas = set()
        if live_tracks and measurements:
            pair_rows = []
            pair_index = []
            for ti, tr in enumerate(live_tracks):
                for mi, z in enumerate(measurements):
                    pair_rows.append(self._pair_features(tr, z, frame_idx))
                    pair_index.append((ti, mi))
            _, _, _, pair_prob, update_delta, _ = self._score(
                pair_features=np.asarray(pair_rows, dtype=np.float32)
            )
            score = np.full((len(live_tracks), len(measurements)), -1e6, dtype=np.float64)
            dists = np.zeros_like(score)
            for k, (ti, mi) in enumerate(pair_index):
                dist = np.linalg.norm(measurements[mi] - live_tracks[ti].state[:3])
                dists[ti, mi] = dist
                gate = self.gate * (self.confirmed_gate_scale if live_tracks[ti].status == "confirmed" else 1.0)
                if dist <= gate:
                    score[ti, mi] = float(pair_prob[k])
            rows, cols = linear_sum_assignment(-score)
            for r, c in zip(rows, cols):
                tr = live_tracks[r]
                gate = self.gate * (self.confirmed_gate_scale if tr.status == "confirmed" else 1.0)
                if score[r, c] < self.assoc_threshold or dists[r, c] > gate:
                    continue
                k = pair_index.index((r, c))
                delta = update_delta[k] if update_delta is not None else np.zeros(6)
                tr.state = tr.state + delta.astype(np.float64)
                # Keep the matched measurement as a strong position anchor.
                prev_xyz = tr.history.get(frame_idx - 1, tr.state[:3])
                tr.state[3:6] = measurements[c] - prev_xyz
                tr.state[:3] = 0.5 * tr.state[:3] + 0.5 * measurements[c]
                tr.add_state(frame_idx, tr.state[:3])
                tr.miss_streak = 0
                tr.death_streak = 0
                tr.hit_streak += 1
                tr.total_hits += 1
                tr.confidence = max(tr.confidence, float(score[r, c]))
                assigned_tracks.add(tr.track_id)
                assigned_meas.add(c)

        for tr in live_tracks:
            if tr.track_id in assigned_tracks:
                continue
            tr.miss_streak += 1
            tr.hit_streak = 0
            if tr.status == "tentative" and tr.miss_streak > 1:
                tr.status = "dead"

        live_tracks = [tr for tr in self.tracks.values() if tr.status in ("tentative", "confirmed")]
        track_features = np.asarray(
            [self._track_features(tr, frame_idx) for tr in live_tracks],
            dtype=np.float32,
        ) if live_tracks else np.empty((0, 10), dtype=np.float32)
        _, confirm_prob, death_prob, _, _, _ = self._score(track_features=track_features)

        for i, tr in enumerate(live_tracks):
            if tr.status == "confirmed":
                protect_until = int(tr.confirmed_frame or frame_idx) + self.confirmed_protect_frames
                in_protect = frame_idx <= protect_until
                if tr.miss_streak == 0:
                    tr.death_streak = 0
                elif death_prob is not None and death_prob[i] >= self.death_threshold and not in_protect:
                    tr.death_streak += 1
                else:
                    tr.death_streak = max(0, tr.death_streak - 1)
                if not in_protect and (
                    tr.miss_streak > self.confirmed_max_missed
                    or tr.death_streak >= self.confirmed_death_streak
                ):
                    tr.status = "dead"
                continue
            if death_prob is not None and death_prob[i] >= self.death_threshold:
                tr.status = "dead"
                continue
            if tr.miss_streak > self.max_missed:
                tr.status = "dead"
                continue
            if tr.status != "tentative" or self.next_slot >= self.max_targets:
                continue
            has_window = all(t in tr.history for t in range(frame_idx - self.tau + 1, frame_idx + 1))
            enough_hits = (
                tr.total_hits >= self.min_confirm_hits
                and tr.hit_streak >= self.min_confirm_hit_streak
            )
            neural_confirm = confirm_prob is not None and confirm_prob[i] >= self.confirm_threshold
            fallback_confirm = tr.hit_streak >= self.tau and tr.total_hits >= self.tau + 1
            duplicate_confirmed = any(
                other.track_id != tr.track_id
                and other.status == "confirmed"
                and np.linalg.norm(other.state[:3] - tr.state[:3]) < self.confirm_nms
                for other in self.tracks.values()
            )
            if duplicate_confirmed:
                continue
            if has_window and enough_hits and (neural_confirm or fallback_confirm):
                tr.status = "confirmed"
                tr.confirmed_frame = int(frame_idx)
                tr.slot = self.next_slot
                tr.miss_streak = 0
                tr.death_streak = 0
                self.next_slot += 1

        unmatched = [i for i in range(len(measurements)) if i not in assigned_meas]
        if unmatched:
            meas_features = np.asarray(
                [self._meas_features(measurements[i]) for i in unmatched],
                dtype=np.float32,
            )
            _, _, _, _, _, birth_prob = self._score(meas_features=meas_features)
            birth_order = sorted(
                range(len(unmatched)),
                key=lambda j: float(birth_prob[j]) if birth_prob is not None else 0.5,
                reverse=True,
            )
            births = 0
            live_xyz = [
                tr.state[:3] for tr in self.tracks.values()
                if tr.status in ("tentative", "confirmed")
            ]
            for local_i in birth_order:
                if births >= self.max_births_per_frame:
                    break
                mi = unmatched[local_i]
                prob = float(birth_prob[local_i]) if birth_prob is not None else 0.5
                too_close = any(
                    np.linalg.norm(measurements[mi] - xyz) < self.birth_nms
                    for xyz in live_xyz
                )
                if too_close:
                    continue
                if self.proposal_birth or prob >= self.birth_threshold:
                    self._new_track(frame_idx, measurements[mi], prob)
                    live_xyz.append(measurements[mi])
                    births += 1

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
                protect_until = int(tr.confirmed_frame or frame_idx) + self.confirmed_protect_frames
                if frame_idx > protect_until:
                    tr.death_streak += 1
                    if (
                        tr.miss_streak > self.confirmed_max_missed
                        and tr.death_streak >= self.confirmed_death_streak
                    ):
                        tr.status = "dead"
                continue
            xyz = np.asarray(filtered_states[tr.slot], dtype=np.float64)
            if np.linalg.norm(xyz - tr.state[:3]) > self.bait_update_gate:
                continue
            prev_xyz = tr.history.get(frame_idx - 1, tr.state[:3])
            tr.state[3:6] = xyz - prev_xyz
            tr.state[:3] = xyz
            tr.add_state(frame_idx, xyz)
            tr.miss_streak = 0
            tr.death_streak = 0
            tr.hit_streak = max(tr.hit_streak, 1)
