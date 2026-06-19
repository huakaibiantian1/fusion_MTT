import json
import os

import numpy as np


def make_star_case(
    seed=617,
    out_path="example_cases/outer_like_star_30f.json",
    n_targets=5,
    num_frames=30,
    radius_range=(450.0, 850.0),
    z_range=(-300.0, 300.0),
    pd=0.9,
    lambda_c=10,
    meas_noise_m=10.0,
    motion_noise_m=0.1,
):
    rng = np.random.default_rng(seed)
    trajectories = []
    center = np.zeros(3)
    for i in range(n_targets):
        az = 2 * np.pi * i / n_targets + rng.normal(0.0, 0.2)
        r = rng.uniform(*radius_range)
        start = np.array([
            r * np.cos(az),
            rng.uniform(-150.0, 150.0),
            rng.uniform(*z_range),
        ])
        if i % 2 == 0:
            end = center + rng.normal(0.0, 15.0, size=3)
        else:
            end = -0.9 * start + rng.normal(0.0, 30.0, size=3)
        states = np.zeros((num_frames, 6), dtype=float)
        for t in range(num_frames):
            a = t / max(num_frames - 1, 1)
            pos = (1 - a) * start + a * end
            pos += rng.normal(0.0, motion_noise_m, size=3)
            states[t, :3] = pos
            if t > 0:
                states[t - 1, 3:6] = states[t, :3] - states[t - 1, :3]
        states[-1, 3:6] = states[-2, 3:6]
        trajectories.append({
            "label": i + 1,
            "birth_frame": 0,
            "death_frame": num_frames - 1,
            "states": np.round(states[:, :3], 3).tolist(),
        })

    measurements = []
    associations = []
    clutter_low = np.array([-850.0, -800.0, -750.0])
    clutter_high = np.array([850.0, 800.0, 750.0])
    for t in range(num_frames):
        frame_meas = []
        frame_assoc = []
        for tr in trajectories:
            if rng.random() <= pd:
                xyz = np.asarray(tr["states"][t], dtype=float)
                frame_meas.append(xyz + rng.normal(0.0, meas_noise_m, size=3))
                frame_assoc.append(int(tr["label"]))
        n_clutter = rng.poisson(lambda_c)
        for _ in range(n_clutter):
            frame_meas.append(rng.uniform(clutter_low, clutter_high))
            frame_assoc.append(0)
        order = rng.permutation(len(frame_meas)) if frame_meas else []
        measurements.append(np.round(np.asarray(frame_meas)[order], 3).tolist() if frame_meas else [])
        associations.append([int(np.asarray(frame_assoc)[j]) for j in order] if frame_meas else [])

    data = {
        "scenario_type": "outer_like_star",
        "seed": seed,
        "num_frames": num_frames,
        "pd": pd,
        "lambda_c": lambda_c,
        "meas_noise_m": meas_noise_m,
        "motion_noise_m": motion_noise_m,
        "trajectories": trajectories,
        "measurements": measurements,
        "gt_associations": associations,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


def make_train_like_motion_case(
    seed=618,
    out_path="example_cases/outer_like_train_motion_80f.json",
    n_targets=5,
    num_frames=80,
    speed_range=(100.0, 500.0),
    pd=0.9,
    lambda_c=10,
    meas_noise_m=50.0,
    motion_noise_m=0.1,
):
    rng = np.random.default_rng(seed)
    trajectories = []
    center = np.zeros(3)
    for i in range(n_targets):
        direction = rng.normal(0.0, 1.0, size=3)
        direction /= max(np.linalg.norm(direction), 1e-8)
        speed = rng.uniform(*speed_range)
        t_cross = rng.integers(num_frames // 3, 2 * num_frames // 3)
        start = center - direction * speed * t_cross
        states = np.zeros((num_frames, 6), dtype=float)
        for t in range(num_frames):
            pos = start + direction * speed * t
            pos += rng.normal(0.0, motion_noise_m, size=3)
            states[t, :3] = pos
            states[t, 3:6] = direction * speed
        trajectories.append({
            "label": i + 1,
            "birth_frame": 0,
            "death_frame": num_frames - 1,
            "states": np.round(states[:, :3], 3).tolist(),
        })

    all_xyz = np.concatenate([
        np.asarray(tr["states"], dtype=float) for tr in trajectories
    ], axis=0)
    low = all_xyz.min(axis=0) - 1000.0
    high = all_xyz.max(axis=0) + 1000.0

    measurements = []
    associations = []
    for t in range(num_frames):
        frame_meas = []
        frame_assoc = []
        for tr in trajectories:
            if rng.random() <= pd:
                xyz = np.asarray(tr["states"][t], dtype=float)
                frame_meas.append(xyz + rng.normal(0.0, meas_noise_m, size=3))
                frame_assoc.append(int(tr["label"]))
        for _ in range(rng.poisson(lambda_c)):
            frame_meas.append(rng.uniform(low, high))
            frame_assoc.append(0)
        order = rng.permutation(len(frame_meas)) if frame_meas else []
        measurements.append(np.round(np.asarray(frame_meas)[order], 3).tolist() if frame_meas else [])
        associations.append([int(np.asarray(frame_assoc)[j]) for j in order] if frame_meas else [])

    data = {
        "scenario_type": "outer_like_train_motion",
        "seed": seed,
        "num_frames": num_frames,
        "pd": pd,
        "lambda_c": lambda_c,
        "speed_range_m_per_frame": list(speed_range),
        "meas_noise_m": meas_noise_m,
        "motion_noise_m": motion_noise_m,
        "trajectories": trajectories,
        "measurements": measurements,
        "gt_associations": associations,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


def make_scaled_copy(src_path, scale, out_path):
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for tr in data["trajectories"]:
        tr["states"] = (np.asarray(tr["states"], dtype=float) * scale).round(3).tolist()
    data["measurements"] = [
        (np.asarray(fm, dtype=float) * scale).round(3).tolist() if len(fm) else []
        for fm in data["measurements"]
    ]
    data["scenario_type"] = data.get("scenario_type", "case") + f"_scaled_{scale:g}"
    data["coordinate_scale_factor"] = scale
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


if __name__ == "__main__":
    raw = make_star_case()
    scaled = make_scaled_copy(raw, 50.0, "example_cases/outer_like_star_30f_scaled50.json")
    train_motion = make_train_like_motion_case()
    print(raw)
    print(scaled)
    print(train_motion)
