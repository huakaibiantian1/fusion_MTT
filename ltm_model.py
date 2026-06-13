import torch
import torch.nn as nn


TRACK_FEATURE_DIM = 10
MEAS_FEATURE_DIM = 5
PAIR_FEATURE_DIM = 18


class LTMScorer(nn.Module):
    """Small learned track manager scorer.

    It scores three decisions:
    - track/measurement affinity
    - unmatched measurement birth probability
    - track confirm/death probabilities
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

        self.pair_head = mlp(PAIR_FEATURE_DIM, 1)
        self.birth_head = mlp(MEAS_FEATURE_DIM, 1)
        self.track_head = mlp(TRACK_FEATURE_DIM, 2)

    def score_pairs(self, pair_features):
        return self.pair_head(pair_features).squeeze(-1)

    def score_birth(self, meas_features):
        return self.birth_head(meas_features).squeeze(-1)

    def score_track(self, track_features):
        out = self.track_head(track_features)
        return out[..., 0], out[..., 1]


def load_ltm_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_kwargs", {})
    model = LTMScorer(**cfg).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt
