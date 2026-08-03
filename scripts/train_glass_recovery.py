#!/usr/bin/env python3
"""Train the frozen-feature glass catastrophe critic + recovery head."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, OrderedDict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crashbench.glass_recovery_data import (
    HAZARD_TYPES,
    RISK_HORIZONS,
    PairedTrajectoryRecord,
    read_trajectory_manifest,
    validate_episode_arrays,
)
from crashbench.glass_recovery_model import (
    GlassLossWeights,
    GlassRecoveryConfig,
    GlassRecoveryNetwork,
    glass_recovery_loss,
)


class PairedFrameDataset(Dataset):
    """Lazy frame view over compressed per-trajectory arrays."""

    def __init__(self, manifest: str | Path, cache_size: int = 2):
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.records = read_trajectory_manifest(self.manifest)
        self.index: list[tuple[int, int]] = []
        self.kinds: list[str] = []
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        for record_index, record in enumerate(self.records):
            with np.load(self.root / record.arrays_path) as arrays:
                validate_episode_arrays(arrays, record.n_steps)
            self.index.extend((record_index, frame) for frame in range(record.n_steps))
            self.kinds.extend([record.trajectory_kind] * record.n_steps)
        if not self.index:
            raise ValueError("training dataset has no frames")

    def __len__(self) -> int:
        return len(self.index)

    def _arrays(self, record_index: int) -> dict[str, np.ndarray]:
        if record_index in self._cache:
            value = self._cache.pop(record_index)
            self._cache[record_index] = value
            return value
        record = self.records[record_index]
        with np.load(self.root / record.arrays_path) as loaded:
            value = {key: loaded[key] for key in loaded.files if key != "images"}
        self._cache[record_index] = value
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        record_index, frame = self.index[item]
        arrays = self._arrays(record_index)
        float_keys = (
            "hidden", "robot_state", "nominal_action", "target_action",
            "risk_targets", "risk_mask", "severity_force", "severity_mask",
            "abort_target", "recovery_mask", "invariance_mask", "sensitivity_mask",
        )
        out = {
            key: torch.as_tensor(np.asarray(arrays[key][frame], dtype=np.float32))
            for key in float_keys
        }
        out["hazard_type"] = torch.tensor(int(arrays["hazard_type"][frame]), dtype=torch.long)
        out["trajectory_kind"] = torch.tensor(
            ("nominal_catastrophe", "oracle_recovery", "off_path_control", "blocked_safe_abort")
            .index(self.records[record_index].trajectory_kind),
            dtype=torch.long,
        )
        return out

    def balanced_sampler(self, seed: int) -> WeightedRandomSampler:
        counts = Counter(self.kinds)
        weights = torch.as_tensor([1.0 / counts[kind] for kind in self.kinds], dtype=torch.double)
        return WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )


def _move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    positive, negative = scores[labels], scores[~labels]
    if not len(positive) or not len(negative):
        return float("nan")
    # Pairwise definition handles ties exactly and is fine for validation scale.
    return float(np.mean((positive[:, None] > negative[None, :])
                         + 0.5 * (positive[:, None] == negative[None, :])))


def _select_risk_threshold(scores: np.ndarray, labels: np.ndarray, max_fpr: float) -> dict:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    if not labels.any() or labels.all():
        raise ValueError("risk calibration needs positive and negative validation frames")
    candidates = np.unique(np.concatenate(([1.0], scores, [0.0])))
    best = None
    for threshold in candidates:
        predicted = scores >= threshold
        fpr = float(predicted[~labels].mean())
        tpr = float(predicted[labels].mean())
        if fpr <= max_fpr + 1e-12:
            key = (tpr, -fpr, float(threshold))
            if best is None or key > best[0]:
                best = (key, threshold, fpr, tpr)
    assert best is not None
    return {"threshold": float(best[1]), "fpr": best[2], "tpr": best[3],
            "max_fpr": float(max_fpr)}


@torch.inference_mode()
def evaluate(model: GlassRecoveryNetwork, loader: DataLoader, device: torch.device) -> tuple[float, dict]:
    model.eval()
    loss_values = []
    risk_scores, risk_targets, risk_masks = [], [], []
    hazard_scores, hazard_targets = [], []
    severity_pred, severity_target, severity_mask = [], [], []
    abort_scores, abort_targets = [], []
    residuals, invariance_masks, sensitivity_masks = [], [], []
    for batch in loader:
        batch = _move(batch, device)
        outputs = model(batch["hidden"], batch["robot_state"], batch["nominal_action"])
        loss, _ = glass_recovery_loss(outputs, batch)
        loss_values.append(float(loss.cpu()))
        risk_scores.append(torch.sigmoid(outputs["risk_logits"]).cpu().numpy())
        risk_targets.append(batch["risk_targets"].cpu().numpy())
        risk_masks.append(batch["risk_mask"].cpu().numpy())
        hazard_scores.append(outputs["hazard_logits"].argmax(-1).cpu().numpy())
        hazard_targets.append(batch["hazard_type"].cpu().numpy())
        severity_pred.append(torch.expm1(outputs["severity_log"]).clamp_min(0).cpu().numpy())
        severity_target.append(batch["severity_force"].cpu().numpy())
        severity_mask.append(batch["severity_mask"].cpu().numpy())
        abort_scores.append(torch.sigmoid(outputs["abort_logit"]).cpu().numpy())
        abort_targets.append(batch["abort_target"].cpu().numpy())
        residuals.append(torch.linalg.vector_norm(outputs["action_residual"], dim=-1).cpu().numpy())
        invariance_masks.append(batch["invariance_mask"].cpu().numpy())
        sensitivity_masks.append(batch["sensitivity_mask"].cpu().numpy())
    risk_scores = np.concatenate(risk_scores)
    risk_targets = np.concatenate(risk_targets)
    risk_masks = np.concatenate(risk_masks).astype(bool)
    hazard_scores = np.concatenate(hazard_scores)
    hazard_targets = np.concatenate(hazard_targets)
    severity_pred = np.concatenate(severity_pred)
    severity_target = np.concatenate(severity_target)
    severity_mask = np.concatenate(severity_mask).astype(bool)
    abort_scores = np.concatenate(abort_scores)
    abort_targets = np.concatenate(abort_targets).astype(bool)
    residuals = np.concatenate(residuals)
    invariance_masks = np.concatenate(invariance_masks).astype(bool)
    sensitivity_masks = np.concatenate(sensitivity_masks).astype(bool)
    metrics = {
        "risk_auc_by_horizon": {
            str(horizon): _auc(risk_scores[:, index][risk_masks[:, index]],
                               risk_targets[:, index][risk_masks[:, index]])
            for index, horizon in enumerate(RISK_HORIZONS)
        },
        "hazard_accuracy": float((hazard_scores == hazard_targets).mean()),
        "severity_force_mae_n": (
            float(np.abs(severity_pred[severity_mask] - severity_target[severity_mask]).mean())
            if severity_mask.any() else None
        ),
        "abort_auc": _auc(abort_scores, abort_targets),
        "control_residual_l2_mean": (
            float(residuals[invariance_masks].mean()) if invariance_masks.any() else None
        ),
        "hazard_residual_l2_mean": (
            float(residuals[sensitivity_masks].mean()) if sensitivity_masks.any() else None
        ),
    }
    return float(np.mean(loss_values)), metrics | {
        "_risk_scores": risk_scores,
        "_risk_targets": risk_targets,
        "_risk_masks": risk_masks,
        "_abort_scores": abort_scores,
        "_abort_targets": abort_targets,
    }


def train(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite non-empty {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_data = PairedFrameDataset(args.train_manifest, cache_size=args.cache_size)
    validation_data = PairedFrameDataset(args.validation_manifest, cache_size=args.cache_size)
    first = train_data[0]
    config = GlassRecoveryConfig(
        hidden_dim=int(first["hidden"].numel()),
        width=args.width,
        depth=args.depth,
        dropout=args.dropout,
        max_action_residual=args.max_action_residual,
    )
    model = GlassRecoveryNetwork(config).to(device)
    weights = GlassLossWeights(
        recovery_bc=args.lambda_recovery,
        risk=args.lambda_risk,
        hazard=args.lambda_hazard,
        severity=args.lambda_severity,
        abort=args.lambda_abort,
        control_invariance=args.lambda_invariance,
        hazard_sensitivity=args.lambda_sensitivity,
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=train_data.balanced_sampler(args.seed),
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=False,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=False,
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.max_steps))
    iterator = iter(train_loader)
    best_loss = float("inf")
    trace = []
    model.train()
    for step in range(1, args.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["hidden"], batch["robot_state"], batch["nominal_action"])
        loss, terms = glass_recovery_loss(
            outputs, batch, weights=weights, sensitivity_margin=args.sensitivity_margin
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
            "learning_rate": scheduler.get_last_lr()[0],
            **{name: float(value.detach().cpu()) for name, value in terms.items()},
        }
        trace.append(row)
        if step == 1 or step % args.log_every == 0:
            print(json.dumps(row), flush=True)
        if step % args.eval_every == 0 or step == args.max_steps:
            validation_loss, validation_metrics = evaluate(model, validation_loader, device)
            public_metrics = {key: value for key, value in validation_metrics.items()
                              if not key.startswith("_")}
            print(json.dumps({"step": step, "validation_loss": validation_loss,
                              **public_metrics}), flush=True)
            if validation_loss < best_loss:
                best_loss = validation_loss
                model.save_checkpoint(output / "best.pt", metadata={"step": step})
            model.train()

    validation_loss, validation_metrics = evaluate(model, validation_loader, device)
    calibration_index = RISK_HORIZONS.index(args.gating_horizon)
    mask = validation_metrics["_risk_masks"][:, calibration_index]
    calibration = _select_risk_threshold(
        validation_metrics["_risk_scores"][:, calibration_index][mask],
        validation_metrics["_risk_targets"][:, calibration_index][mask],
        args.max_control_fpr,
    )
    calibration.update({
        "horizon": args.gating_horizon,
        "exit_threshold": max(0.0, calibration["threshold"] * args.exit_threshold_ratio),
        "abort_threshold": args.abort_threshold,
    })
    public_metrics = {key: value for key, value in validation_metrics.items()
                      if not key.startswith("_")}
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": os.environ.get("CB_CODE_COMMIT"),
        "method": "frozen OpenVLA hidden + robot-state joint critic/recovery action head",
        "train_manifest": str(args.train_manifest),
        "validation_manifest": str(args.validation_manifest),
        "heldout_used_for_training_or_calibration": False,
        "loss_weights": asdict(weights),
        "calibration": calibration,
        "validation_loss": validation_loss,
        "validation_metrics": public_metrics,
        "train_frames": len(train_data),
        "validation_frames": len(validation_data),
        "train_kind_counts": dict(Counter(train_data.kinds)),
    }
    model.save_checkpoint(output / "last.pt", metadata=metadata)
    # Promote the final calibrated state for inference; ``best.pt`` remains an
    # uncalibrated diagnostic snapshot.
    model.save_checkpoint(output / "glass_recovery.pt", metadata=metadata)
    (output / "training_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "train_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trace)
    )
    print(f"saved {output / 'glass_recovery.pt'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", default="results/glass_recovery_v1/dataset/train.jsonl")
    parser.add_argument("--validation-manifest", default="results/glass_recovery_v1/dataset/validation.jsonl")
    parser.add_argument("--output", default="results/glass_recovery_v1/checkpoint")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--max-action-residual", type=float, default=1.0)
    parser.add_argument("--sensitivity-margin", type=float, default=0.20)
    parser.add_argument("--lambda-recovery", type=float, default=1.0)
    parser.add_argument("--lambda-risk", type=float, default=1.0)
    parser.add_argument("--lambda-hazard", type=float, default=0.25)
    parser.add_argument("--lambda-severity", type=float, default=0.25)
    parser.add_argument("--lambda-abort", type=float, default=0.5)
    parser.add_argument("--lambda-invariance", type=float, default=0.5)
    parser.add_argument("--lambda-sensitivity", type=float, default=0.25)
    parser.add_argument("--gating-horizon", type=int, choices=RISK_HORIZONS, default=10)
    parser.add_argument("--max-control-fpr", type=float, default=0.05)
    parser.add_argument("--exit-threshold-ratio", type=float, default=0.5)
    parser.add_argument("--abort-threshold", type=float, default=0.60)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_steps < 1 or args.batch_size < 1:
        raise SystemExit("max-steps and batch-size must be positive")
    train(args)


if __name__ == "__main__":
    main()
