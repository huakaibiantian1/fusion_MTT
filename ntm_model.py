import torch
import torch.nn as nn


TRACK_FEATURE_DIM = 10
MEAS_FEATURE_DIM = 5
PAIR_FEATURE_DIM = 20


class NTMModel(nn.Module):
    """Neural track manager.

    The model replaces KF motion/update logic with learned heads:
    - motion_head predicts per-frame [dx, dy, dz, dvx, dvy, dvz]
    - pair_head scores track-measurement association
    - update_head predicts state correction from a matched measurement
    - birth_head scores unmatched measurements
    - lifecycle_head scores confirm/death decisions
    """

    def __init__(self, hidden_dim=128, dropout=0.1):
        super().__init__()

        def mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )

        self.motion_head = mlp(TRACK_FEATURE_DIM, 6)
        self.pair_head = mlp(PAIR_FEATURE_DIM, 1)
        self.update_head = mlp(PAIR_FEATURE_DIM, 6)
        self.birth_head = mlp(MEAS_FEATURE_DIM, 1)
        self.lifecycle_head = mlp(TRACK_FEATURE_DIM, 2)

    def predict_motion(self, track_features):
        return self.motion_head(track_features)

    def score_pairs(self, pair_features):
        return self.pair_head(pair_features).squeeze(-1)

    def update_state_delta(self, pair_features):
        return self.update_head(pair_features)

    def score_birth(self, meas_features):
        return self.birth_head(meas_features).squeeze(-1)

    def score_lifecycle(self, track_features):
        out = self.lifecycle_head(track_features)
        return out[..., 0], out[..., 1]


def load_ntm_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_kwargs", {})
    model = NTMModel(**cfg).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt
