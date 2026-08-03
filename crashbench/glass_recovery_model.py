"""Frozen-feature catastrophe critic and glass recovery action head.

The 7B OpenVLA backbone is not modified in phase one.  Its final hidden state is
concatenated with the 8-D LIBERO robot state and passed through this small joint
network.  The outputs are:

* collision logits for 1/3/5/10/20 steps;
* causal hazard type;
* log1p future contact-force severity;
* blocked/safe-abort logit;
* a bounded residual over the nominal 7-D action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from crashbench.glass_recovery_data import HAZARD_TYPES, RISK_HORIZONS


@dataclass(frozen=True)
class GlassRecoveryConfig:
    hidden_dim: int = 4096
    robot_state_dim: int = 8
    action_dim: int = 7
    width: int = 512
    depth: int = 2
    dropout: float = 0.10
    max_action_residual: float = 1.0


@dataclass(frozen=True)
class GlassLossWeights:
    recovery_bc: float = 1.0
    risk: float = 1.0
    hazard: float = 0.25
    severity: float = 0.25
    abort: float = 0.5
    control_invariance: float = 0.5
    hazard_sensitivity: float = 0.25


class GlassRecoveryNetwork(nn.Module):
    def __init__(self, config: GlassRecoveryConfig | None = None):
        super().__init__()
        self.config = config or GlassRecoveryConfig()
        cfg = self.config
        self.hidden_norm = nn.LayerNorm(cfg.hidden_dim)
        self.state_norm = nn.LayerNorm(cfg.robot_state_dim)
        layers: list[nn.Module] = [
            nn.Linear(cfg.hidden_dim + cfg.robot_state_dim, cfg.width),
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
        fused = torch.cat((self.hidden_norm(hidden), self.state_norm(robot_state)), dim=-1)
        features = self.trunk(fused)
        residual = torch.tanh(self.action_head(features)) * self.config.max_action_residual
        recovery = torch.clamp(nominal_action + residual, -1.0, 1.0)
        return {
            "risk_logits": self.risk_head(features),
            "hazard_logits": self.hazard_head(features),
            "severity_log": F.softplus(self.severity_head(features).squeeze(-1)),
            "abort_logit": self.abort_head(features).squeeze(-1),
            "action_residual": residual,
            "recovery_action": recovery,
        }

    def checkpoint_payload(self, metadata: Mapping | None = None) -> dict:
        return {
            "schema_version": 1,
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
        if payload.get("schema_version") != 1:
            raise ValueError(f"unsupported recovery checkpoint {payload.get('schema_version')!r}")
        if tuple(payload.get("risk_horizons", ())) != RISK_HORIZONS:
            raise ValueError("checkpoint risk horizons do not match runtime")
        if tuple(payload.get("hazard_types", ())) != HAZARD_TYPES:
            raise ValueError("checkpoint hazard vocabulary does not match runtime")
        model = cls(GlassRecoveryConfig(**payload["model_config"]))
        model.load_state_dict(payload["state_dict"])
        return model, dict(payload.get("metadata", {}))


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
    recovery_bc = _masked_mean(action_error, batch["recovery_mask"])
    invariance_error = outputs["action_residual"].pow(2).mean(dim=-1)
    invariance = _masked_mean(invariance_error, batch["invariance_mask"])
    action_change = torch.linalg.vector_norm(outputs["action_residual"], dim=-1)
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
    total = sum(getattr(weights, name) * value for name, value in terms.items())
    return total, terms
