"""Latched runtime ownership for the frozen OpenVLA + recovery head."""

from __future__ import annotations

import numpy as np

from crashbench.glass_recovery_data import RISK_HORIZONS, SCHEMA_VERSION
from crashbench.glass_recovery_model import (
    HIDDEN_HOOK_IDENTITY,
    PRIMARY_CHECKPOINT_KIND,
    PRIMARY_DISABLED_AUXILIARY_HEADS,
    PRIMARY_DISABLED_AUXILIARY_TERMS,
    TTE_DEFINITION,
    GlassLossWeights,
    GlassRecoveryNetwork,
)


def _is_sha256(value) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _is_immutable_revision(value) -> bool:
    text = str(value or "").lower()
    return 40 <= len(text) <= 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _validate_primary_checkpoint_metadata(metadata: dict) -> None:
    required = {
        "checkpoint_kind",
        "base_resolved_revision",
        "unnorm_key",
        "hidden_hook_identity",
        "train_manifest_sha256",
        "validation_manifest_sha256",
        "trajectory_schema_version",
        "protocol_sha256",
        "trigger_horizon_actions",
        "tte_definition",
        "seed",
        "disabled_auxiliary_heads",
        "disabled_auxiliary_loss_terms",
        "loss_weights",
        "calibration",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"primary recovery checkpoint metadata missing {missing}")
    if metadata["checkpoint_kind"] != PRIMARY_CHECKPOINT_KIND:
        raise ValueError("checkpoint is not a P0-C primary recovery checkpoint")
    if not _is_immutable_revision(metadata["base_resolved_revision"]):
        raise ValueError("checkpoint Base resolved revision is not immutable")
    if not metadata["unnorm_key"]:
        raise ValueError("checkpoint Base unnorm key is empty")
    if metadata["trajectory_schema_version"] != SCHEMA_VERSION:
        raise ValueError("checkpoint trajectory schema does not match runtime")
    for name in (
        "train_manifest_sha256",
        "validation_manifest_sha256",
        "protocol_sha256",
    ):
        if not _is_sha256(metadata[name]):
            raise ValueError(f"checkpoint {name} is not a SHA-256 digest")
    if metadata["tte_definition"] != TTE_DEFINITION:
        raise ValueError("checkpoint TTE definition does not match runtime")
    if metadata["hidden_hook_identity"] != HIDDEN_HOOK_IDENTITY:
        raise ValueError("checkpoint hidden-hook identity does not match runtime")
    horizon = metadata["trigger_horizon_actions"]
    if not isinstance(horizon, int) or horizon not in RISK_HORIZONS:
        raise ValueError("checkpoint H does not select a runtime risk horizon")
    disabled = set(metadata["disabled_auxiliary_heads"])
    if disabled != set(PRIMARY_DISABLED_AUXILIARY_HEADS):
        raise ValueError("checkpoint disabled auxiliary heads are inconsistent")
    disabled_terms = set(metadata["disabled_auxiliary_loss_terms"])
    if disabled_terms != set(PRIMARY_DISABLED_AUXILIARY_TERMS):
        raise ValueError("checkpoint disabled auxiliary loss terms are inconsistent")
    weights = metadata["loss_weights"]
    try:
        configured_weights = GlassLossWeights(**weights)
        configured_weights.validate_primary()
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint primary loss ownership is inconsistent") from exc
    calibration = metadata["calibration"]
    if (
        not isinstance(calibration, dict)
        or calibration.get("horizon") != horizon
        or calibration.get("objective")
        != "maximize_timely_trigger_rate_subject_to_clean_control_episode_fpr"
        or calibration.get("ownership") != "recovery_latched_until_terminal_or_reset"
        or calibration.get("abort_enabled") is not False
    ):
        raise ValueError("checkpoint calibration/ownership contract is inconsistent")


def _base_resolved_revision(base) -> str | None:
    identity = getattr(base, "checkpoint_identity", None)
    if not isinstance(identity, dict):
        return None
    revision = identity.get("resolved_revision") or identity.get("model_config_commit_hash")
    return str(revision) if revision else None


class GlassRecoveryPolicy:
    """Pass nominal actions until first risk crossing, then own the episode.

    The auxiliary abort head remains checkpoint-readable but cannot influence
    this primary wrapper.  Once recovery takes over, post-intervention risk is
    logged only; it never hands control back to the nominal policy.
    """

    def __init__(
        self,
        base,
        checkpoint: str,
        *,
        risk_horizon: int = 10,
        risk_enter_threshold: float = 0.50,
        # Deprecated P0-B arguments retained for caller compatibility.  They
        # are intentionally inert under latched primary ownership.
        risk_exit_threshold: float = 0.25,
        abort_threshold: float = 0.60,
        min_recovery_steps: int = 3,
        device: str | None = None,
        abort_controller=None,
    ):
        import torch

        if risk_horizon not in RISK_HORIZONS:
            raise ValueError(f"risk_horizon must be one of {RISK_HORIZONS}")
        if not 0 <= risk_enter_threshold <= 1:
            raise ValueError("risk_enter_threshold must lie in [0, 1]")
        if not getattr(base, "capture_hidden", False):
            raise ValueError("base policy must be created with capture_hidden=True")
        self.base = base
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.checkpoint_metadata = GlassRecoveryNetwork.load_checkpoint(
            checkpoint, map_location=self.device
        )
        _validate_primary_checkpoint_metadata(self.checkpoint_metadata)
        self.model.to(self.device).eval()
        checkpoint_horizon = int(self.checkpoint_metadata["trigger_horizon_actions"])
        if int(risk_horizon) != checkpoint_horizon:
            raise ValueError(
                f"runtime H={risk_horizon} does not match checkpoint H={checkpoint_horizon}"
            )
        expected_revision = str(self.checkpoint_metadata["base_resolved_revision"])
        actual_revision = _base_resolved_revision(base)
        if actual_revision != expected_revision:
            raise ValueError(
                "runtime Base resolved revision does not match checkpoint: "
                f"runtime={actual_revision!r}, checkpoint={expected_revision!r}"
            )
        actual_unnorm_key = getattr(getattr(base, "cfg", None), "unnorm_key", None)
        expected_unnorm_key = str(self.checkpoint_metadata["unnorm_key"])
        if actual_unnorm_key != expected_unnorm_key:
            raise ValueError(
                "runtime Base unnorm key does not match checkpoint: "
                f"runtime={actual_unnorm_key!r}, checkpoint={expected_unnorm_key!r}"
            )
        self.risk_index = RISK_HORIZONS.index(risk_horizon)
        self.risk_horizon = risk_horizon
        self.risk_enter_threshold = float(risk_enter_threshold)
        # Public attributes make the deprecation observable without letting
        # these values affect execution.
        self.risk_exit_threshold = float(risk_exit_threshold)
        self.abort_threshold = float(abort_threshold)
        self.min_recovery_steps = int(min_recovery_steps)
        self.abort_enabled = False
        if abort_controller is not None:
            raise ValueError("abort is disabled in the primary glass recovery wrapper")
        self.reset()

    @property
    def resize_size(self) -> int:
        return self.base.resize_size

    def reset(self) -> None:
        if hasattr(self.base, "reset"):
            self.base.reset()
        self.mode = "nominal"
        self.ownership_age: int | None = None
        self.intervention_count = 0
        self.first_intervention_step = None
        self.previous_risk_probability: float | None = None
        self.step_index = -1
        self.decisions: list[dict] = []
        self.last_decision: dict | None = None

    def mark_terminal(self) -> None:
        """Mark the current episode terminal without implicitly resetting Base."""

        self.mode = "terminal"

    def _predict(self, hidden: np.ndarray, state: np.ndarray, nominal: np.ndarray) -> dict:
        import torch

        with torch.inference_mode():
            outputs = self.model(
                torch.as_tensor(hidden, dtype=torch.float32, device=self.device).reshape(1, -1),
                torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1),
                torch.as_tensor(nominal, dtype=torch.float32, device=self.device).reshape(1, -1),
            )
            risk = torch.sigmoid(outputs["risk_logits"])[0].float().cpu().numpy()
            recovery = outputs["recovery_action"][0].float().cpu().numpy()
        return {
            "risk": risk,
            "recovery_action": recovery,
        }

    def act(self, observation: dict, instruction: str) -> np.ndarray:
        if self.mode == "terminal":
            raise RuntimeError("cannot act after terminal; call reset() for a new episode")
        self.step_index += 1
        nominal = np.asarray(self.base.act(observation, instruction), dtype=np.float32)
        hidden = self.base.last_hidden
        if hidden is None:
            raise RuntimeError("base policy did not expose a hidden state")
        prediction = self._predict(
            np.asarray(hidden, dtype=np.float32),
            np.asarray(observation["state"], dtype=np.float32),
            nominal,
        )
        risk_probability = float(prediction["risk"][self.risk_index])
        qualified_crossing = bool(
            self.mode == "nominal"
            and risk_probability >= self.risk_enter_threshold
            and (
                self.previous_risk_probability is None
                or self.previous_risk_probability < self.risk_enter_threshold
            )
        )
        if qualified_crossing:
            self.mode = "recovery_latched"
            self.ownership_age = 0
            self.first_intervention_step = self.step_index
        elif self.mode == "recovery_latched":
            assert self.ownership_age is not None
            self.ownership_age += 1

        recovery = np.asarray(prediction["recovery_action"], dtype=np.float32)
        if self.mode == "recovery_latched":
            action = recovery
        else:
            action = nominal

        intervened = self.mode == "recovery_latched"
        if intervened:
            self.intervention_count += 1
        executed = np.clip(action, -1.0, 1.0).astype(np.float32)
        decision = {
            "step": self.step_index,
            "mode": self.mode,
            "intervened": intervened,
            "first_trigger": qualified_crossing,
            "first_trigger_step": self.first_intervention_step,
            "ownership_age": self.ownership_age,
            "risk_horizon": self.risk_horizon,
            "risk_probability": risk_probability,
            "risk_vector": [float(value) for value in prediction["risk"]],
            "nominal_action": nominal.astype(float).tolist(),
            "recovery_action": recovery.astype(float).tolist(),
            "executed_action": executed.astype(float).tolist(),
            "abort_enabled": False,
        }
        self.decisions.append(decision)
        self.last_decision = decision
        self.previous_risk_probability = risk_probability
        return executed
