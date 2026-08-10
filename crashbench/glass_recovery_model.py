"""Frozen-feature catastrophe critic and glass recovery action head.

The 7B OpenVLA backbone is not modified in phase one.  Its final hidden state is
concatenated with the 8-D LIBERO robot state and nominal 7-D action, then passed
through this small joint network.  The outputs are:

* collision logits for 1/3/5/10/20 steps;
* a direct bounded 7-D recovery action.

The hazard, severity, and abort heads remain in the module and checkpoint
format so historical E14 checkpoints stay readable.  P0-C primary checkpoints
train only risk, recovery behavior cloning, and clean-control invariance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from crashbench.glass_recovery_data import HAZARD_TYPES, RISK_HORIZONS


PRIMARY_CHECKPOINT_KIND = "glass_recovery_primary"
TTE_DEFINITION = (
    "steps_until_event(event_action_index=c, observation_index=i) = c - i + 1 "
    "actions from pre-action observation s_i"
)
HIDDEN_HOOK_IDENTITY = (
    "OpenVLAPolicy.language_model.model.norm:prefill_last_prompt_token"
)
PRIMARY_DISABLED_AUXILIARY_HEADS = (
    "hazard",
    "severity",
    "abort",
)
PRIMARY_DISABLED_AUXILIARY_TERMS = PRIMARY_DISABLED_AUXILIARY_HEADS + (
    "hazard_sensitivity",
)


@dataclass(frozen=True)
class GlassRecoveryConfig:
    hidden_dim: int = 4096
    robot_state_dim: int = 8
    action_dim: int = 7
    width: int = 512
    depth: int = 2
    dropout: float = 0.10


@dataclass(frozen=True)
class GlassLossWeights:
    recovery_bc: float = 1.0
    risk: float = 1.0
    hazard: float = 0.0
    severity: float = 0.0
    abort: float = 0.0
    control_invariance: float = 0.5
    hazard_sensitivity: float = 0.0

    def validate_primary(self) -> None:
        """Fail closed if an auxiliary objective is enabled for a main run."""

        nonzero = {
            name: float(getattr(self, name))
            for name in PRIMARY_DISABLED_AUXILIARY_TERMS
            if float(getattr(self, name)) != 0.0
        }
        if nonzero:
            raise ValueError(
                "primary checkpoints require zero auxiliary loss weights; "
                f"got {nonzero}"
            )
        enabled = {
            "risk": float(self.risk),
            "recovery_bc": float(self.recovery_bc),
            "control_invariance": float(self.control_invariance),
        }
        if any(value <= 0.0 for value in enabled.values()):
            raise ValueError(
                "primary risk, recovery_bc, and control_invariance weights "
                f"must be positive; got {enabled}"
            )


class GlassRecoveryNetwork(nn.Module):
    def __init__(self, config: GlassRecoveryConfig | None = None):
        super().__init__()
        self.config = config or GlassRecoveryConfig()
        cfg = self.config
        self.hidden_norm = nn.LayerNorm(cfg.hidden_dim)
        self.state_norm = nn.LayerNorm(cfg.robot_state_dim)
        self.action_norm = nn.LayerNorm(cfg.action_dim)
        layers: list[nn.Module] = [
            nn.Linear(cfg.hidden_dim + cfg.robot_state_dim + cfg.action_dim, cfg.width),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        ]
        for _ in range(max(0, cfg.depth - 1)):
            layers.extend([
                nn.Linear(cfg.width, cfg.width),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ])
        self.trunk = nn.Sequential(*layers)
        self.risk_head = nn.Linear(cfg.width, len(RISK_HORIZONS))
        self.hazard_head = nn.Linear(cfg.width, len(HAZARD_TYPES))
        self.severity_head = nn.Linear(cfg.width, 1)
        self.abort_head = nn.Linear(cfg.width, 1)
        self.action_head = nn.Linear(cfg.width, cfg.action_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        robot_state: torch.Tensor,
        nominal_action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if hidden.shape[-1] != self.config.hidden_dim:
            raise ValueError(
                f"hidden dim {hidden.shape[-1]} != configured {self.config.hidden_dim}"
            )
        if robot_state.shape[-1] != self.config.robot_state_dim:
            raise ValueError("robot-state dimension mismatch")
        if nominal_action.shape[-1] != self.config.action_dim:
            raise ValueError("nominal-action dimension mismatch")
        fused = torch.cat((
            self.hidden_norm(hidden),
            self.state_norm(robot_state),
            self.action_norm(nominal_action),
        ), dim=-1)
        features = self.trunk(fused)
        recovery = torch.tanh(self.action_head(features))
        action_delta = recovery - nominal_action
        return {
            "risk_logits": self.risk_head(features),
            "hazard_logits": self.hazard_head(features),
            "severity_log": F.softplus(self.severity_head(features).squeeze(-1)),
            "abort_logit": self.abort_head(features).squeeze(-1),
            "action_delta": action_delta,
            "recovery_action": recovery,
        }

    def checkpoint_payload(self, metadata: Mapping | None = None) -> dict:
        return {
            "schema_version": 2,
            "model_config": asdict(self.config),
            "risk_horizons": list(RISK_HORIZONS),
            "hazard_types": list(HAZARD_TYPES),
            "state_dict": self.state_dict(),
            "metadata": dict(metadata or {}),
        }

    def save_checkpoint(self, path: str | Path, metadata: Mapping | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(metadata), target)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["GlassRecoveryNetwork", dict]:
        # Checkpoints are produced locally by this pipeline and contain config +
        # metadata in addition to tensors, so the tensor-only loader is not used.
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if payload.get("schema_version") != 2:
            raise ValueError(f"unsupported recovery checkpoint {payload.get('schema_version')!r}")
        if tuple(payload.get("risk_horizons", ())) != RISK_HORIZONS:
            raise ValueError("checkpoint risk horizons do not match runtime")
        if tuple(payload.get("hazard_types", ())) != HAZARD_TYPES:
            raise ValueError("checkpoint hazard vocabulary does not match runtime")
        model = cls(GlassRecoveryConfig(**payload["model_config"]))
        model.load_state_dict(payload["state_dict"])
        return model, dict(payload.get("metadata", {}))


def state_dict_sha256(model: nn.Module) -> str:
    """Stable digest of parameter names, dtypes, shapes, and tensor bytes."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def glass_recovery_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    weights: GlassLossWeights | None = None,
    sensitivity_margin: float = 0.20,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Joint supervised objective with explicit invariance/sensitivity terms."""

    weights = weights or GlassLossWeights()
    risk_elementwise = F.binary_cross_entropy_with_logits(
        outputs["risk_logits"], batch["risk_targets"], reduction="none"
    )
    risk = _masked_mean(risk_elementwise, batch["risk_mask"])
    hazard = F.cross_entropy(outputs["hazard_logits"], batch["hazard_type"].long())
    severity_elementwise = F.smooth_l1_loss(
        outputs["severity_log"], torch.log1p(batch["severity_force"]), reduction="none"
    )
    severity = _masked_mean(severity_elementwise, batch["severity_mask"])
    abort = F.binary_cross_entropy_with_logits(outputs["abort_logit"], batch["abort_target"])

    action_error = (outputs["recovery_action"] - batch["target_action"]).pow(2).mean(dim=-1)
    recovery_mask = batch["recovery_mask"]
    if "trajectory_kind" in batch:
        # Index 3 is the historical/appendix blocked_safe_abort branch.  Main
        # schema-v2 datasets reject it entirely, and this second guard prevents
        # a legacy or manually assembled batch from turning abort demonstrations
        # into recovery action targets.
        recovery_mask = recovery_mask * (batch["trajectory_kind"] != 3)
    recovery_bc = _masked_mean(action_error, recovery_mask)
    invariance_error = outputs["action_delta"].pow(2).mean(dim=-1)
    invariance = _masked_mean(invariance_error, batch["invariance_mask"])
    action_change = torch.linalg.vector_norm(outputs["action_delta"], dim=-1)
    sensitivity_error = F.relu(float(sensitivity_margin) - action_change).pow(2)
    sensitivity = _masked_mean(sensitivity_error, batch["sensitivity_mask"])

    terms = {
        "recovery_bc": recovery_bc,
        "risk": risk,
        "hazard": hazard,
        "severity": severity,
        "abort": abort,
        "control_invariance": invariance,
        "hazard_sensitivity": sensitivity,
    }
    # Exclude zero-weight terms from the autograd graph entirely.  Multiplying
    # an auxiliary term by zero would still create zero gradients, which in
    # turn lets decoupled optimizer weight decay mutate a supposedly disabled
    # head in a main run.
    enabled_terms = [
        float(getattr(weights, name)) * value
        for name, value in terms.items()
        if float(getattr(weights, name)) != 0.0
    ]
    if not enabled_terms:
        raise ValueError("glass recovery loss has no enabled terms")
    total = sum(enabled_terms)
    return total, terms
