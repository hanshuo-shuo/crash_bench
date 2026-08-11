#!/usr/bin/env python3
"""Train the frozen-feature glass catastrophe critic + recovery head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, OrderedDict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crashbench.glass_recovery_data import (
    HAZARD_TYPES,
    PRIMARY_TRAJECTORY_KINDS,
    RISK_HORIZONS,
    SCHEMA_VERSION,
    PairedTrajectoryRecord,
    canonical_sha256,
    read_trajectory_manifest,
    validate_episode_arrays,
)
from crashbench.glass_recovery_model import (
    GlassLossWeights,
    GlassRecoveryConfig,
    GlassRecoveryNetwork,
    HIDDEN_HOOK_IDENTITY,
    PRIMARY_CHECKPOINT_KIND,
    PRIMARY_DISABLED_AUXILIARY_HEADS,
    PRIMARY_DISABLED_AUXILIARY_TERMS,
    TTE_DEFINITION,
    glass_recovery_loss,
    state_dict_sha256,
)


PHASES = (
    "nominal_far_safe",
    "nominal_imminent",
    "exact_trigger",
    "oracle_first_k",
    "oracle_continuation",
    "off_path_control",
)
# Eighths make the paper's 25/25/12.5/12.5/25 pilot target exact.  The
# imminent/certified-trigger quarter is split evenly so the exact certified
# trigger cannot be diluted by the rest of a nominal suffix.
PHASE_BATCH_UNITS = OrderedDict((
    ("nominal_far_safe", 2),
    ("nominal_imminent", 1),
    ("exact_trigger", 1),
    ("oracle_first_k", 1),
    ("oracle_continuation", 1),
    ("off_path_control", 2),
))


class PairedFrameDataset(Dataset):
    """Lazy frame view over compressed per-trajectory arrays."""

    def __init__(self, manifest: str | Path, cache_size: int = 2, first_k: int = 5):
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.records = read_trajectory_manifest(self.manifest)
        if int(first_k) < 1:
            raise ValueError("first_k must be positive")
        self.first_k = int(first_k)
        bad_versions = sorted({
            record.schema_version for record in self.records
            if record.schema_version != SCHEMA_VERSION
        })
        if bad_versions:
            raise ValueError(
                "main training accepts only schema-v2 primary manifests; "
                f"found schema versions {bad_versions}"
            )
        bad_kinds = sorted({
            record.trajectory_kind for record in self.records
            if record.trajectory_kind not in PRIMARY_TRAJECTORY_KINDS
        })
        if bad_kinds:
            raise ValueError(
                "main training excludes auxiliary trajectories; "
                f"found {bad_kinds}"
            )
        self.index: list[tuple[int, int]] = []
        self.kinds: list[str] = []
        self.item_phases: list[str] = []
        self.pair_ids = sorted({record.pair_id for record in self.records})
        self.pair_to_index = {pair_id: index for index, pair_id in enumerate(self.pair_ids)}
        self.phase_indices: dict[str, dict[str, list[int]]] = {
            phase: {pair_id: [] for pair_id in self.pair_ids} for phase in PHASES
        }
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        for record_index, record in enumerate(self.records):
            with np.load(self.root / record.arrays_path) as arrays:
                validate_episode_arrays(
                    arrays, record.n_steps, schema_version=record.schema_version
                )
                phases = self._record_phases(record, arrays)
            for frame, phase in enumerate(phases):
                item_index = len(self.index)
                self.index.append((record_index, frame))
                self.kinds.append(record.trajectory_kind)
                self.item_phases.append(phase)
                self.phase_indices[phase][record.pair_id].append(item_index)
        if not self.index:
            raise ValueError("training dataset has no frames")

        missing = {
            pair_id: [
                phase for phase in PHASES
                if not self.phase_indices[phase][pair_id]
            ]
            for pair_id in self.pair_ids
        }
        missing = {pair_id: phases for pair_id, phases in missing.items() if phases}
        if missing:
            raise ValueError(
                "every accepted pair must support every main sampling phase; "
                f"missing {missing}"
            )

    def _record_phases(
        self,
        record: PairedTrajectoryRecord,
        arrays: Mapping[str, np.ndarray],
    ) -> list[str]:
        if record.trajectory_kind == "off_path_control":
            return ["off_path_control"] * record.n_steps
        if record.trajectory_kind == "oracle_recovery":
            return [
                "oracle_first_k" if frame < self.first_k else "oracle_continuation"
                for frame in range(record.n_steps)
            ]
        if record.trajectory_kind != "nominal_catastrophe":
            raise ValueError(
                f"unsupported main trajectory kind {record.trajectory_kind!r}"
            )

        horizon = int(record.trigger_horizon_actions or 0)
        if horizon < 2:
            raise ValueError(f"{record.pair_id} needs H >= 2 for phase-balanced training")
        times = np.asarray(arrays["time_to_catastrophe_actions"], dtype=int)
        certified = np.asarray(arrays["runtime_trigger_eligible"], dtype=bool)
        if int(certified.sum()) != 1:
            raise ValueError(
                f"{record.pair_id} nominal branch needs exactly one certified trigger frame"
            )
        midpoint = max(1, horizon // 2)
        phases = []
        for remaining, is_certified in zip(times, certified):
            if is_certified:
                phases.append("exact_trigger")
            elif int(remaining) > midpoint:
                phases.append("nominal_far_safe")
            else:
                phases.append("nominal_imminent")
        return phases

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
            "time_to_catastrophe_actions", "time_to_catastrophe_mask",
            "runtime_trigger_eligible",
        )
        out = {
            key: torch.as_tensor(np.asarray(arrays[key][frame], dtype=np.float32))
            for key in float_keys
        }
        out["hazard_type"] = torch.tensor(int(arrays["hazard_type"][frame]), dtype=torch.long)
        out["trajectory_kind"] = torch.tensor(
            ("nominal_catastrophe", "oracle_recovery", "off_path_control")
            .index(self.records[record_index].trajectory_kind),
            dtype=torch.long,
        )
        out["episode_id"] = torch.tensor(record_index, dtype=torch.long)
        out["pair_id"] = torch.tensor(
            self.pair_to_index[self.records[record_index].pair_id], dtype=torch.long
        )
        out["frame_index"] = torch.tensor(frame, dtype=torch.long)
        out["sampling_phase"] = torch.tensor(
            PHASES.index(self.item_phases[item]), dtype=torch.long
        )
        return out

    def phase_counts(self) -> dict[str, int]:
        return dict(Counter(self.item_phases))


class PairPhaseBatchSampler(Sampler[list[int]]):
    """Fixed-quota batches with uniform pair selection inside every phase."""

    def __init__(self, dataset: PairedFrameDataset, batch_size: int, seed: int):
        if int(batch_size) < 8 or int(batch_size) % 8:
            raise ValueError("phase-balanced batch_size must be a positive multiple of 8")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.quotas = {
            phase: units * (self.batch_size // 8)
            for phase, units in PHASE_BATCH_UNITS.items()
        }
        self.num_batches = max(1, math.ceil(len(dataset) / self.batch_size))

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        pair_orders: dict[str, list[str]] = {}
        pair_cursors = {phase: 0 for phase in PHASES}

        def next_pair(phase: str) -> str:
            order = pair_orders.get(phase, [])
            cursor = pair_cursors[phase]
            if cursor >= len(order):
                order = list(self.dataset.pair_ids)
                rng.shuffle(order)
                pair_orders[phase] = order
                cursor = 0
            pair_cursors[phase] = cursor + 1
            return order[cursor]

        for _ in range(self.num_batches):
            batch: list[int] = []
            for phase, quota in self.quotas.items():
                for _ in range(quota):
                    pair_id = next_pair(phase)
                    pool = self.dataset.phase_indices[phase][pair_id]
                    batch.append(pool[rng.randrange(len(pool))])
            rng.shuffle(batch)
            yield batch


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_collection_contract(dataset: PairedFrameDataset) -> dict:
    """Resolve and cross-check the P0-B collection contract for one split."""

    summary_path = dataset.manifest.parent / "collection_summary.json"
    if not summary_path.is_file():
        raise ValueError(
            f"{dataset.manifest} needs sibling collection_summary.json for provenance"
        )
    summary = json.loads(summary_path.read_text())
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{summary_path} is not schema v{SCHEMA_VERSION}")
    metadata = summary.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{summary_path} lacks collection metadata")
    protocol = metadata.get("primary_protocol")
    protocol_sha256 = metadata.get("primary_protocol_sha256")
    if not isinstance(protocol, Mapping) or not protocol_sha256:
        raise ValueError(f"{summary_path} lacks the primary protocol and SHA")
    if canonical_sha256(protocol) != protocol_sha256:
        raise ValueError(f"{summary_path} primary protocol SHA is invalid")
    checkpoint_identity = metadata.get("checkpoint_identity")
    if not isinstance(checkpoint_identity, Mapping):
        raise ValueError(f"{summary_path} lacks Base checkpoint identity")
    resolved_revision = checkpoint_identity.get("resolved_revision")
    if not resolved_revision:
        raise ValueError(f"{summary_path} lacks Base resolved revision")
    policy_protocol = protocol.get("policy")
    if not isinstance(policy_protocol, Mapping) or not policy_protocol.get("unnorm_key"):
        raise ValueError(f"{summary_path} lacks the Base unnorm key")

    versions = {record.schema_version for record in dataset.records}
    horizons = {record.trigger_horizon_actions for record in dataset.records}
    if versions != {SCHEMA_VERSION}:
        raise ValueError(f"{dataset.manifest} is not a pure schema-v2 cohort")
    if len(horizons) != 1 or None in horizons:
        raise ValueError(f"{dataset.manifest} does not declare one fixed H")
    horizon = int(next(iter(horizons)))
    if int(protocol.get("precrash_horizon_actions", -1)) != horizon:
        raise ValueError(f"{dataset.manifest} H disagrees with its collection protocol")
    return {
        "base_resolved_revision": str(resolved_revision),
        "unnorm_key": str(policy_protocol["unnorm_key"]),
        "protocol_sha256": str(protocol_sha256),
        "trajectory_schema_version": SCHEMA_VERSION,
        "trigger_horizon_actions": horizon,
        "collection_summary": str(summary_path),
    }


def _training_provenance(
    train_data: PairedFrameDataset,
    validation_data: PairedFrameDataset,
    args: argparse.Namespace,
) -> dict:
    train_contract = _manifest_collection_contract(train_data)
    validation_contract = _manifest_collection_contract(validation_data)
    agreement_fields = (
        "base_resolved_revision",
        "unnorm_key",
        "protocol_sha256",
        "trajectory_schema_version",
        "trigger_horizon_actions",
    )
    mismatches = {
        field: (train_contract[field], validation_contract[field])
        for field in agreement_fields
        if train_contract[field] != validation_contract[field]
    }
    if mismatches:
        raise ValueError(f"train/validation collection contracts disagree: {mismatches}")

    explicit = {
        "base_resolved_revision": getattr(args, "base_resolved_revision", None),
        "unnorm_key": getattr(args, "unnorm_key", None),
        "protocol_sha256": getattr(args, "protocol_sha256", None),
        "trigger_horizon_actions": getattr(args, "trigger_horizon", None),
    }
    for field, expected in explicit.items():
        if expected is not None and str(expected) != str(train_contract[field]):
            raise ValueError(
                f"configured {field}={expected!r} disagrees with manifest "
                f"value {train_contract[field]!r}"
            )
    if train_contract["trigger_horizon_actions"] not in RISK_HORIZONS:
        raise ValueError(
            "fixed H must select an available risk head; got "
            f"H={train_contract['trigger_horizon_actions']} and heads={RISK_HORIZONS}"
        )
    return {
        **{field: train_contract[field] for field in agreement_fields},
        "hidden_hook_identity": str(
            getattr(args, "hidden_hook_identity", HIDDEN_HOOK_IDENTITY)
        ),
        "train_manifest": str(train_data.manifest),
        "train_manifest_sha256": _file_sha256(train_data.manifest),
        "validation_manifest": str(validation_data.manifest),
        "validation_manifest_sha256": _file_sha256(validation_data.manifest),
        "tte_definition": TTE_DEFINITION,
        "seed": int(args.seed),
    }


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


def _episode_max_scores(
    scores: np.ndarray,
    valid: np.ndarray,
    episode_ids: np.ndarray,
    trajectory_kinds: np.ndarray,
    selected_kind: int,
) -> np.ndarray:
    """Reduce valid frame risks to one maximum for each selected trajectory."""

    scores = np.asarray(scores, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    episode_ids = np.asarray(episode_ids, dtype=int)
    trajectory_kinds = np.asarray(trajectory_kinds, dtype=int)
    maxima = []
    for episode_id in np.unique(episode_ids[trajectory_kinds == selected_kind]):
        episode = episode_ids == episode_id
        kinds = np.unique(trajectory_kinds[episode])
        if len(kinds) != 1 or int(kinds[0]) != selected_kind:
            raise ValueError(f"episode {episode_id} mixes trajectory kinds")
        usable = episode & valid
        if not usable.any():
            raise ValueError(f"episode {episode_id} has no valid risk frames")
        maxima.append(float(scores[usable].max()))
    if not maxima:
        raise ValueError("risk calibration has no episodes for the requested trajectory kind")
    return np.asarray(maxima, dtype=float)


def _episode_qualified_scores(
    scores: np.ndarray,
    valid: np.ndarray,
    qualified: np.ndarray,
    episode_ids: np.ndarray,
    trajectory_kinds: np.ndarray,
    selected_kind: int,
) -> np.ndarray:
    """Return the best score only over certified timely-trigger frames."""

    return _episode_max_scores(
        scores,
        np.asarray(valid, dtype=bool) & np.asarray(qualified, dtype=bool),
        episode_ids,
        trajectory_kinds,
        selected_kind,
    )


def _select_timely_risk_threshold(
    control_episode_scores: np.ndarray,
    timely_trigger_episode_scores: np.ndarray,
    max_control_episode_fpr: float,
) -> dict:
    """Maximize timely triggers subject to clean-control episode FPR."""

    controls = np.asarray(control_episode_scores, dtype=float)
    timely = np.asarray(timely_trigger_episode_scores, dtype=float)
    if not len(controls) or not len(timely):
        raise ValueError("risk calibration needs control and timely-trigger validation episodes")
    if not 0.0 <= float(max_control_episode_fpr) <= 1.0:
        raise ValueError("max_control_episode_fpr must lie in [0, 1]")
    candidates = np.unique(np.concatenate(([1.0], controls, timely, [0.0])))
    best = None
    for threshold in candidates:
        fpr = float((controls >= threshold).mean())
        timely_rate = float((timely >= threshold).mean())
        if fpr <= max_control_episode_fpr + 1e-12:
            key = (timely_rate, -fpr, float(threshold))
            if best is None or key > best[0]:
                best = (key, threshold, fpr, timely_rate)
    assert best is not None
    return {
        "threshold": float(best[1]),
        "control_episode_fpr": best[2],
        "timely_trigger_rate": best[3],
        "max_control_episode_fpr": float(max_control_episode_fpr),
        "control_episodes": int(len(controls)),
        "certified_trigger_episodes": int(len(timely)),
        "calibration_unit": "control_episode_max_vs_certified_trigger_frame",
        "objective": "maximize_timely_trigger_rate_subject_to_clean_control_episode_fpr",
    }


@torch.inference_mode()
def evaluate(
    model: GlassRecoveryNetwork,
    loader: DataLoader,
    device: torch.device,
    *,
    weights: GlassLossWeights,
    sensitivity_margin: float,
) -> tuple[float, dict]:
    model.eval()
    loss_values = []
    risk_scores, risk_targets, risk_masks = [], [], []
    hazard_scores, hazard_targets = [], []
    severity_pred, severity_target, severity_mask = [], [], []
    abort_scores, abort_targets = [], []
    action_deltas, invariance_masks, sensitivity_masks = [], [], []
    recovery_actions, target_actions, recovery_masks = [], [], []
    episode_ids, trajectory_kinds, trigger_eligible = [], [], []
    for batch in loader:
        batch = _move(batch, device)
        outputs = model(batch["hidden"], batch["robot_state"], batch["nominal_action"])
        loss, _ = glass_recovery_loss(
            outputs,
            batch,
            weights=weights,
            sensitivity_margin=sensitivity_margin,
        )
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
        action_deltas.append(torch.linalg.vector_norm(outputs["action_delta"], dim=-1).cpu().numpy())
        invariance_masks.append(batch["invariance_mask"].cpu().numpy())
        sensitivity_masks.append(batch["sensitivity_mask"].cpu().numpy())
        recovery_actions.append(outputs["recovery_action"].cpu().numpy())
        target_actions.append(batch["target_action"].cpu().numpy())
        recovery_masks.append(batch["recovery_mask"].cpu().numpy())
        episode_ids.append(batch["episode_id"].cpu().numpy())
        trajectory_kinds.append(batch["trajectory_kind"].cpu().numpy())
        trigger_eligible.append(batch["runtime_trigger_eligible"].cpu().numpy())
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
    action_deltas = np.concatenate(action_deltas)
    invariance_masks = np.concatenate(invariance_masks).astype(bool)
    sensitivity_masks = np.concatenate(sensitivity_masks).astype(bool)
    recovery_actions = np.concatenate(recovery_actions)
    target_actions = np.concatenate(target_actions)
    recovery_masks = np.concatenate(recovery_masks).astype(bool)
    episode_ids = np.concatenate(episode_ids)
    trajectory_kinds = np.concatenate(trajectory_kinds)
    trigger_eligible = np.concatenate(trigger_eligible).astype(bool)
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
        "control_action_delta_l2_mean": (
            float(action_deltas[invariance_masks].mean()) if invariance_masks.any() else None
        ),
        "hazard_action_delta_l2_mean": (
            float(action_deltas[sensitivity_masks].mean()) if sensitivity_masks.any() else None
        ),
        "gripper_sign_accuracy": (
            float((
                (recovery_actions[recovery_masks, -1] >= 0.0)
                == (target_actions[recovery_masks, -1] >= 0.0)
            ).mean())
            if recovery_masks.any() else None
        ),
    }
    return float(np.mean(loss_values)), metrics | {
        "_risk_scores": risk_scores,
        "_risk_targets": risk_targets,
        "_risk_masks": risk_masks,
        "_abort_scores": abort_scores,
        "_abort_targets": abort_targets,
        "_episode_ids": episode_ids,
        "_trajectory_kinds": trajectory_kinds,
        "_runtime_trigger_eligible": trigger_eligible,
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

    first_k = int(getattr(args, "first_k", 5))
    train_data = PairedFrameDataset(
        args.train_manifest, cache_size=args.cache_size, first_k=first_k
    )
    validation_data = PairedFrameDataset(
        args.validation_manifest, cache_size=args.cache_size, first_k=first_k
    )
    provenance = _training_provenance(train_data, validation_data, args)
    gating_horizon = getattr(args, "gating_horizon", None)
    if gating_horizon is None:
        gating_horizon = provenance["trigger_horizon_actions"]
    if int(gating_horizon) != int(provenance["trigger_horizon_actions"]):
        raise ValueError(
            "primary gating horizon must equal certified H; "
            f"got gating={gating_horizon}, H={provenance['trigger_horizon_actions']}"
        )
    gating_horizon = int(gating_horizon)
    first = train_data[0]
    config = GlassRecoveryConfig(
        hidden_dim=int(first["hidden"].numel()),
        width=args.width,
        depth=args.depth,
        dropout=args.dropout,
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
    weights.validate_primary()
    disabled_auxiliary_heads = list(PRIMARY_DISABLED_AUXILIARY_HEADS)
    sampler = PairPhaseBatchSampler(train_data, args.batch_size, args.seed)
    train_loader = DataLoader(
        train_data,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=False,
    )
    train_eval_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=False,
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.max_steps))
    iterator = iter(train_loader)
    best_loss = float("inf")
    best_step: int | None = None
    trace = []
    checkpoint_contract = {
        "checkpoint_kind": PRIMARY_CHECKPOINT_KIND,
        **provenance,
        "loss_weights": asdict(weights),
        "disabled_auxiliary_heads": disabled_auxiliary_heads,
        "disabled_auxiliary_loss_terms": list(PRIMARY_DISABLED_AUXILIARY_TERMS),
        "enabled_objectives": ["risk", "recovery_bc", "control_invariance"],
        "sampler": {
            "name": "pair_uniform_phase_balanced",
            "batch_size": int(args.batch_size),
            "first_k": first_k,
            "fixed_batch_quota": dict(sampler.quotas),
            "phase_counts": train_data.phase_counts(),
            "phase_definition": {
                "exact_trigger": "runtime_trigger_eligible == 1",
                "nominal_far_safe": "non-trigger nominal TTE > floor(H / 2)",
                "nominal_imminent": "non-trigger nominal TTE <= floor(H / 2)",
                "oracle_first_k": "oracle frame_index < K",
                "oracle_continuation": "oracle frame_index >= K",
                "off_path_control": "all matched off-path task-success frames",
            },
        },
        "validation_criterion": "configured_primary_validation_loss",
        "validation_loss_weights": asdict(weights),
    }
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
            validation_loss, validation_metrics = evaluate(
                model,
                validation_loader,
                device,
                weights=weights,
                sensitivity_margin=args.sensitivity_margin,
            )
            public_metrics = {key: value for key, value in validation_metrics.items()
                              if not key.startswith("_")}
            print(json.dumps({"step": step, "validation_loss": validation_loss,
                              **public_metrics}), flush=True)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_step = step
                model_digest = state_dict_sha256(model)
                model.save_checkpoint(output / "best.pt", metadata={
                    **checkpoint_contract,
                    "selected_step": step,
                    "selected_validation_loss": best_loss,
                    "model_state_sha256": model_digest,
                })
            model.train()

    if best_step is None or not (output / "best.pt").is_file():
        raise RuntimeError("training did not produce a selected best checkpoint")

    # Keep the terminal optimization state for diagnosis, but do not calibrate
    # or publish it.  The selected checkpoint is reloaded below.
    last_validation_loss, last_validation_metrics = evaluate(
        model,
        validation_loader,
        device,
        weights=weights,
        sensitivity_margin=args.sensitivity_margin,
    )
    last_digest = state_dict_sha256(model)
    model.save_checkpoint(output / "last.pt", metadata={
        **checkpoint_contract,
        "last_step": int(args.max_steps),
        "last_validation_loss": last_validation_loss,
        "model_state_sha256": last_digest,
        "selected_best_step": best_step,
    })

    best_model, best_metadata = GlassRecoveryNetwork.load_checkpoint(
        output / "best.pt", map_location=device
    )
    best_model.to(device).eval()
    best_digest = state_dict_sha256(best_model)
    if best_metadata.get("model_state_sha256") != best_digest:
        raise RuntimeError("reloaded best checkpoint state digest is inconsistent")
    if best_metadata.get("checkpoint_kind") != PRIMARY_CHECKPOINT_KIND:
        raise RuntimeError("reloaded best checkpoint lost its primary contract")
    validation_loss, validation_metrics = evaluate(
        best_model,
        validation_loader,
        device,
        weights=weights,
        sensitivity_margin=args.sensitivity_margin,
    )
    if not math.isclose(validation_loss, best_loss, rel_tol=1e-7, abs_tol=1e-9):
        raise RuntimeError(
            "reloaded best checkpoint does not reproduce its selection loss: "
            f"selected={best_loss}, reloaded={validation_loss}"
        )
    train_loss, train_metrics = evaluate(
        best_model,
        train_eval_loader,
        device,
        weights=weights,
        sensitivity_margin=args.sensitivity_margin,
    )

    calibration_index = RISK_HORIZONS.index(gating_horizon)
    mask = validation_metrics["_risk_masks"][:, calibration_index]
    scores = validation_metrics["_risk_scores"][:, calibration_index]
    control_episode_scores = _episode_max_scores(
        scores,
        mask,
        validation_metrics["_episode_ids"],
        validation_metrics["_trajectory_kinds"],
        selected_kind=2,
    )
    timely_trigger_episode_scores = _episode_qualified_scores(
        scores,
        mask,
        validation_metrics["_runtime_trigger_eligible"],
        validation_metrics["_episode_ids"],
        validation_metrics["_trajectory_kinds"],
        selected_kind=0,
    )
    calibration = _select_timely_risk_threshold(
        control_episode_scores,
        timely_trigger_episode_scores,
        args.max_control_episode_fpr,
    )
    calibration.update({
        "horizon": gating_horizon,
        "certified_horizon_actions": provenance["trigger_horizon_actions"],
        "ownership": "recovery_latched_until_terminal_or_reset",
        "abort_enabled": False,
    })
    public_metrics = {key: value for key, value in validation_metrics.items()
                      if not key.startswith("_")}
    public_train_metrics = {key: value for key, value in train_metrics.items()
                            if not key.startswith("_")}
    metadata = {
        **checkpoint_contract,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": os.environ.get("CB_CODE_COMMIT"),
        "method": (
            "frozen OpenVLA hidden + robot state + nominal action joint critic; "
            "direct recovery action head"
        ),
        "heldout_used_for_training_or_calibration": False,
        "calibration": calibration,
        "validation_loss": validation_loss,
        "validation_metrics": public_metrics,
        "train_loss": train_loss,
        "train_metrics": public_train_metrics,
        "selected_best_step": best_step,
        "selected_best_validation_loss": best_loss,
        "selected_best_model_state_sha256": best_digest,
        "published_model_state_sha256": best_digest,
        "published_from": "best.pt",
        "last_validation_loss": last_validation_loss,
        "last_model_state_sha256": last_digest,
        "train_frames": len(train_data),
        "validation_frames": len(validation_data),
        "train_kind_counts": dict(Counter(train_data.kinds)),
    }
    best_model.save_checkpoint(output / "glass_recovery.pt", metadata=metadata)
    published_model, published_metadata = GlassRecoveryNetwork.load_checkpoint(
        output / "glass_recovery.pt", map_location="cpu"
    )
    published_digest = state_dict_sha256(published_model)
    if published_digest != best_digest:
        raise RuntimeError("published checkpoint weights differ from selected best checkpoint")
    if published_metadata.get("published_from") != "best.pt":
        raise RuntimeError("published checkpoint does not identify its selected source")
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
    parser.add_argument("--sensitivity-margin", type=float, default=0.20)
    parser.add_argument("--lambda-recovery", type=float, default=1.0)
    parser.add_argument("--lambda-risk", type=float, default=1.0)
    # Retain the legacy flags so old commands fail with a precise primary-loss
    # error instead of an argparse error.  Main checkpoints require all four
    # auxiliary values to remain zero.
    parser.add_argument("--lambda-hazard", type=float, default=0.0)
    parser.add_argument("--lambda-severity", type=float, default=0.0)
    parser.add_argument("--lambda-abort", type=float, default=0.0)
    parser.add_argument("--lambda-invariance", type=float, default=0.5)
    parser.add_argument("--lambda-sensitivity", type=float, default=0.0)
    parser.add_argument("--gating-horizon", type=int, choices=RISK_HORIZONS, default=None)
    parser.add_argument("--first-k", type=int, default=5)
    parser.add_argument("--max-control-episode-fpr", type=float, default=0.05)
    parser.add_argument("--base-resolved-revision", default=None)
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--protocol-sha256", default=None)
    parser.add_argument("--trigger-horizon", type=int, default=None)
    # Deprecated primary-wrapper knobs are accepted for command compatibility;
    # latched ownership and disabled abort make them intentionally inert.
    parser.add_argument("--exit-threshold-ratio", type=float, default=0.5)
    parser.add_argument("--abort-threshold", type=float, default=0.60)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_steps < 1 or args.batch_size < 1 or args.first_k < 1:
        raise SystemExit("max-steps, batch-size, and first-k must be positive")
    train(args)


if __name__ == "__main__":
    main()
