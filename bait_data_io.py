import json
import os
import pickle

import numpy as np


def load_scenario_file(path, association_gate_m=300.0):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
    elif ext == ".npz":
        obj = dict(np.load(path, allow_pickle=True))
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    else:
        raise ValueError(f"不支持的数据文件格式: {ext}")

    scenario = _object_to_scenario(obj)
    trajectories, measurements, gt_associations = scenario
    trajectories = _normalize_trajectories(trajectories)
    measurements = _normalize_measurements(measurements)

    if gt_associations is None:
        gt_associations = infer_gt_associations(
            trajectories, measurements, gate_m=association_gate_m
        )
    else:
        gt_associations = _normalize_associations(gt_associations, len(measurements))

    return trajectories, measurements, gt_associations


def _object_to_scenario(obj):
    if isinstance(obj, list):
        if len(obj) == 0:
            raise ValueError("数据文件为空")
        if _looks_like_scenario(obj):
            return _tuple_to_scenario(obj)
        return _object_to_scenario(obj[0])

    if isinstance(obj, tuple):
        return _tuple_to_scenario(obj)

    if isinstance(obj, dict):
        if "scenario" in obj:
            return _object_to_scenario(obj["scenario"])
        trajectories = _first_present(obj, ("trajectories", "tracks", "truth", "states"))
        measurements = _first_present(obj, ("measurements", "detections", "meas"))
        gt_associations = _first_present(
            obj, ("gt_associations", "associations", "measurement_labels", "meas_labels")
        )
        if gt_associations is None and "labels" in obj and "track_labels" not in obj:
            gt_associations = obj["labels"]
        if trajectories is not None and not _is_list_of_track_dicts(trajectories):
            births = _first_present(obj, ("birth_frames", "births", "birth"))
            deaths = _first_present(obj, ("death_frames", "deaths", "death"))
            track_labels = _first_present(obj, ("track_labels", "trajectory_labels", "ids"))
            if births is not None or deaths is not None or track_labels is not None:
                trajectories = _array_with_metadata_to_trajectories(
                    trajectories, births, deaths, track_labels
                )
        if trajectories is None or measurements is None:
            raise ValueError("数据文件需要包含 trajectories/truth/states 和 measurements/detections")
        return trajectories, measurements, gt_associations

    raise ValueError(f"无法识别的数据结构: {type(obj)}")


def _first_present(obj, keys):
    for key in keys:
        if key in obj:
            return obj[key]
    return None


def _looks_like_scenario(obj):
    return len(obj) >= 2 and isinstance(obj[0], (list, tuple)) and isinstance(obj[1], (list, tuple, np.ndarray))


def _is_list_of_track_dicts(value):
    value = _unwrap_npz_value(value)
    if isinstance(value, np.ndarray) and value.dtype == object:
        value = value.tolist()
    return isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict)


def _array_with_metadata_to_trajectories(raw, births, deaths, labels):
    arr = np.asarray(_unwrap_npz_value(raw), dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("带元数据的 truth/states 需要是 [N,T,D] 或 [T,N,D]")
    if arr.shape[0] > arr.shape[1]:
        arr = np.transpose(arr, (1, 0, 2))
    n_tracks = arr.shape[0]
    births = np.asarray(_unwrap_npz_value(births), dtype=np.int64).reshape(-1) if births is not None else None
    deaths = np.asarray(_unwrap_npz_value(deaths), dtype=np.int64).reshape(-1) if deaths is not None else None
    labels = np.asarray(_unwrap_npz_value(labels), dtype=np.int64).reshape(-1) if labels is not None else None
    trajectories = []
    for i in range(n_tracks):
        states = arr[i]
        birth = int(births[i]) if births is not None and i < len(births) else _first_valid_frame(states)
        death = int(deaths[i]) if deaths is not None and i < len(deaths) else _last_valid_frame(states)
        label = int(labels[i]) if labels is not None and i < len(labels) else i + 1
        states = _fill_invalid_states(states, birth, death)
        trajectories.append({
            "label": label,
            "states": states.astype(np.float64),
            "birth_frame": birth,
            "death_frame": death,
        })
    return trajectories


def _tuple_to_scenario(obj):
    if len(obj) < 2:
        raise ValueError("scenario 至少需要包含 trajectories 和 measurements")
    trajectories = obj[0]
    measurements = obj[1]
    gt_associations = obj[2] if len(obj) >= 3 else None
    return trajectories, measurements, gt_associations


def _unwrap_npz_value(value):
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        return value.item()
    return value


def _normalize_trajectories(raw):
    raw = _unwrap_npz_value(raw)
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        raw = raw.tolist()

    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        trajectories = []
        for i, tr in enumerate(raw):
            states_raw = _first_present(tr, ("states", "state", "xyz"))
            states = np.asarray(states_raw, dtype=np.float64)
            if states.ndim != 2 or states.shape[1] < 3:
                raise ValueError("每条 trajectory 的 states 需要是 [T, >=3]")
            birth = int(tr.get("birth_frame", tr.get("birth", _first_valid_frame(states))))
            death = int(tr.get("death_frame", tr.get("death", _last_valid_frame(states))))
            label = int(tr.get("label", tr.get("id", i + 1)))
            states = _fill_invalid_states(states, birth, death)
            trajectories.append({
                "label": label,
                "states": states.astype(np.float64),
                "birth_frame": birth,
                "death_frame": death,
            })
        return trajectories

    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("truth/states 数组需要是 [N,T,D] 或 [T,N,D]")

    if arr.shape[0] > arr.shape[1]:
        arr = np.transpose(arr, (1, 0, 2))

    trajectories = []
    for i in range(arr.shape[0]):
        states = arr[i]
        birth = _first_valid_frame(states)
        death = _last_valid_frame(states)
        states = _fill_invalid_states(states, birth, death)
        trajectories.append({
            "label": i + 1,
            "states": states.astype(np.float64),
            "birth_frame": birth,
            "death_frame": death,
        })
    return trajectories


def _first_valid_frame(states):
    valid = _valid_rows(states)
    idx = np.where(valid)[0]
    return int(idx[0]) if len(idx) else 0


def _last_valid_frame(states):
    valid = _valid_rows(states)
    idx = np.where(valid)[0]
    return int(idx[-1]) if len(idx) else max(0, len(states) - 1)


def _valid_rows(states):
    xyz = np.asarray(states)[:, :3]
    return np.all(np.isfinite(xyz), axis=1) & (np.linalg.norm(xyz, axis=1) > 0)


def _fill_invalid_states(states, birth, death):
    states = np.asarray(states, dtype=np.float64).copy()
    if states.shape[1] < 6:
        pad = np.zeros((states.shape[0], 6 - states.shape[1]), dtype=np.float64)
        states = np.hstack([states, pad])
    valid = np.all(np.isfinite(states[:, :3]), axis=1)
    last = None
    for t in range(states.shape[0]):
        if valid[t]:
            last = states[t, :3].copy()
        elif last is not None:
            states[t, :3] = last
    for t in range(states.shape[0] - 1, -1, -1):
        if valid[t]:
            last = states[t, :3].copy()
        elif last is not None:
            states[t, :3] = last
    states[:birth, :3] = states[birth, :3]
    states[death + 1:, :3] = states[death, :3]
    return states


def _normalize_measurements(raw):
    raw = _unwrap_npz_value(raw)
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        raw = raw.tolist()
    if isinstance(raw, list):
        return [np.asarray(frame, dtype=np.float64).reshape(-1, 3) for frame in raw]

    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("measurements 数组需要是 [T,M,3]，空/填充值可用 NaN")
    frames = []
    for t in range(arr.shape[0]):
        fm = arr[t, :, :3]
        valid = np.all(np.isfinite(fm), axis=1)
        frames.append(fm[valid].astype(np.float64))
    return frames


def _normalize_associations(raw, num_frames):
    raw = _unwrap_npz_value(raw)
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        raw = raw.tolist()
    if isinstance(raw, list):
        return [np.asarray(frame, dtype=np.int64).reshape(-1) for frame in raw]
    arr = np.asarray(raw)
    if arr.ndim == 1 and num_frames == 1:
        return [arr.astype(np.int64)]
    if arr.ndim != 2:
        raise ValueError("gt_associations 需要是 list[array] 或 [T,M]")
    frames = []
    for t in range(arr.shape[0]):
        fm = arr[t]
        if np.issubdtype(fm.dtype, np.floating):
            fm = fm[np.isfinite(fm)]
        frames.append(fm.astype(np.int64))
    return frames


def infer_gt_associations(trajectories, measurements, gate_m=300.0):
    associations = []
    for t, meas in enumerate(measurements):
        labels = np.zeros(len(meas), dtype=np.int64)
        active = [
            tr for tr in trajectories
            if int(tr["birth_frame"]) <= t <= int(tr["death_frame"])
        ]
        if len(active) == 0 or len(meas) == 0:
            associations.append(labels)
            continue
        true_xyz = np.asarray([tr["states"][t, :3] for tr in active], dtype=np.float64)
        dists = np.linalg.norm(np.asarray(meas)[:, None, :3] - true_xyz[None, :, :], axis=2)
        nearest = np.argmin(dists, axis=1)
        nearest_dist = dists[np.arange(len(meas)), nearest]
        for i, (j, dist) in enumerate(zip(nearest, nearest_dist)):
            if dist <= gate_m:
                labels[i] = int(active[int(j)]["label"])
        associations.append(labels)
    return associations
