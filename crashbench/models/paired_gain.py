"""Exact-state paired utility targets and a small deployable gain predictor."""
from __future__ import annotations
from collections import defaultdict
import numpy as np

OPTION_IDS = ("base_continue", "observation_refresh", "safe_stop")
GAIN_OPTIONS = OPTION_IDS[1:]


def pair_decisions(data):
    """Pair by block identity, never by incidental row ordering."""
    required = ("row_ids", "features", "options", "actual_u0", "outcomes", "costs", "sources", "roles")
    n = len(data["row_ids"])
    if not n or any(len(data[k]) != n for k in required):
        raise ValueError("paired data lengths must agree and be nonempty")
    groups = defaultdict(dict)
    for i, (row_id, option) in enumerate(zip(data["row_ids"], data["options"])):
        block, suffix = str(row_id).rsplit(":", 1)
        if suffix != option or option not in OPTION_IDS or option in groups[block]:
            raise ValueError("duplicate or inconsistent block/option identity")
        groups[block][str(option)] = i
    blocks, indices = [], []
    for block, by_option in sorted(groups.items()):
        if set(by_option) != set(OPTION_IDS):
            raise ValueError("paired targets require a complete option catalog")
        idx = [by_option[o] for o in OPTION_IDS]
        if len(set(map(str, data["sources"][idx]))) != 1 or len(set(map(str, data["roles"][idx]))) != 1:
            raise ValueError("paired branches must share source and split role")
        if not all(np.array_equal(data["features"][idx[0]], data["features"][i]) for i in idx):
            raise ValueError("paired branches must share the exact anchor features")
        blocks.append(block)
        indices.append(idx)
    indices = np.asarray(indices)
    utility = np.asarray(data["actual_u0"][indices], dtype=np.float32)
    features = np.asarray(data["features"][indices[:, 0]], dtype=np.float32)
    if not np.isfinite(utility).all() or not np.isfinite(features).all():
        raise ValueError("paired targets and features must be finite")
    roles, sources = data["roles"][indices[:, 0]], data["sources"][indices[:, 0]]
    for source in set(sources):
        if len(set(roles[sources == source])) != 1:
            raise ValueError("physical source crosses roles")
    return dict(blocks=np.asarray(blocks), indices=indices, features=features,
                utility=utility, gains=utility[:, 1:] - utility[:, :1],
                outcomes=data["outcomes"][indices], costs=data["costs"][indices],
                sources=sources, roles=roles)


def train_standardization(features, roles, *, floor=1e-6):
    """Feature normalization fitted to declared train rows only."""
    x, roles = np.asarray(features), np.asarray(roles)
    if x.ndim != 2 or len(x) != len(roles) or set(roles) != {"train"} or not np.isfinite(x).all():
        raise ValueError("normalization fitting requires finite train-only features")
    return x.mean(0).astype(np.float32), np.maximum(x.std(0), floor).astype(np.float32)


def choose_from_gains(gains, blocks, *, allow_stop=True):
    values = np.asarray(gains)
    if values.shape != (len(blocks), 2) or not np.isfinite(values).all():
        raise ValueError("expected finite Refresh/Stop gains for each block")
    catalog = OPTION_IDS if allow_stop else OPTION_IDS[:2]
    scores = np.column_stack([np.zeros(len(values)), values])[:, :len(catalog)]
    # Exact ties preserve Base. No development-tuned threshold.
    selected = scores.argmax(1)
    return {str(b): catalog[i] for b, i in zip(blocks, selected)}


def tiny_train_sources(paired, count=4):
    train = paired["roles"] == "train"
    rescue = (paired["outcomes"][:, 1] == 0) & (paired["outcomes"][:, 0] != 0)
    supported = []
    for source in sorted(set(paired["sources"][train])):
        mask = train & (paired["sources"] == source)
        if np.any(rescue & mask) and np.any((paired["gains"][:, 0] <= 0) & mask):
            supported.append(str(source))
    return supported[:count]


try:
    import torch
    from torch import nn
except ImportError:  # no training in lightweight installs
    torch = None

if torch is not None:
    class PairedGainModel(nn.Module):
        def __init__(self, input_dim, *, normalization="layernorm", mean=None, scale=None, hidden_dim=128):
            super().__init__()
            if normalization not in {"layernorm", "train_standardized"}:
                raise ValueError("unknown gain feature normalization")
            self.normalization = normalization
            if normalization == "train_standardized":
                if mean is None or scale is None or len(mean) != input_dim or len(scale) != input_dim:
                    raise ValueError("standardized model requires train normalization statistics")
                if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(np.asarray(scale) <= 0):
                    raise ValueError("invalid normalization statistics")
            else:
                mean, scale = np.zeros(input_dim), np.ones(input_dim)
            self.register_buffer("feature_mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("feature_scale", torch.tensor(scale, dtype=torch.float32))
            self.trunk = nn.Sequential(
                nn.LayerNorm(input_dim) if normalization == "layernorm" else nn.Identity(),
                nn.Linear(input_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            )
            self.gain_head = nn.Linear(hidden_dim, 2)

        def forward(self, features):
            return self.gain_head(self.trunk((features - self.feature_mean) / self.feature_scale))

        @staticmethod
        def loss(predictions, targets, source_weights, *, positive_weight=1.0):
            if positive_weight < 1 or not np.isfinite(positive_weight):
                raise ValueError("positive weight must be finite and at least one")
            weights = source_weights[:, None] * torch.where(targets > 0, positive_weight, 1.0)
            return (((predictions - targets) ** 2 * weights).sum(0) / weights.sum(0)).mean()
else:
    class PairedGainModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("paired gain training requires PyTorch")
