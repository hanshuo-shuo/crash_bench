"""Runtime risk gating for the frozen OpenVLA + glass recovery head."""

from __future__ import annotations

import numpy as np

from crashbench.glass_recovery_data import HAZARD_TYPES, RISK_HORIZONS
from crashbench.glass_recovery_model import GlassRecoveryNetwork
from crashbench.recovery import RetreatHold


class GlassRecoveryPolicy:
    """Pass nominal actions at low risk; recover or abort at high risk.

    Recovery uses hysteresis and is re-evaluated each step.  Abort is latched for
    the rest of the episode and uses the already validated closed-loop
    ``RetreatHold`` controller.  The wrapper exposes decision metadata so clean
    control false interventions can be measured directly.
    """

    def __init__(
        self,
        base,
        checkpoint: str,
        *,
        risk_horizon: int = 10,
        risk_enter_threshold: float = 0.50,
        risk_exit_threshold: float = 0.25,
        abort_threshold: float = 0.60,
        min_recovery_steps: int = 3,
        device: str | None = None,
        abort_controller=None,
    ):
        import torch

        if risk_horizon not in RISK_HORIZONS:
            raise ValueError(f"risk_horizon must be one of {RISK_HORIZONS}")
        if not 0 <= risk_exit_threshold < risk_enter_threshold <= 1:
            raise ValueError("risk thresholds must satisfy 0 <= exit < enter <= 1")
        if not 0 <= abort_threshold <= 1:
            raise ValueError("abort_threshold must lie in [0, 1]")
        if not getattr(base, "capture_hidden", False):
            raise ValueError("base policy must be created with capture_hidden=True")
        self.base = base
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.checkpoint_metadata = GlassRecoveryNetwork.load_checkpoint(
            checkpoint, map_location=self.device
        )
        self.model.to(self.device).eval()
        self.risk_index = RISK_HORIZONS.index(risk_horizon)
        self.risk_horizon = risk_horizon
        self.risk_enter_threshold = float(risk_enter_threshold)
        self.risk_exit_threshold = float(risk_exit_threshold)
        self.abort_threshold = float(abort_threshold)
        self.min_recovery_steps = int(min_recovery_steps)
        self.abort_controller = abort_controller if abort_controller is not None else RetreatHold()
        self.reset()

    @property
    def resize_size(self) -> int:
        return self.base.resize_size

    def reset(self) -> None:
        if hasattr(self.base, "reset"):
            self.base.reset()
        if hasattr(self.abort_controller, "target"):
            self.abort_controller.target = None
        self.mode = "nominal"
        self.recovery_steps = 0
        self.intervention_count = 0
        self.first_intervention_step = None
        self.step_index = -1
        self.decisions: list[dict] = []
        self.last_decision: dict | None = None

    def _predict(self, hidden: np.ndarray, state: np.ndarray, nominal: np.ndarray) -> dict:
        import torch

        with torch.inference_mode():
            outputs = self.model(
                torch.as_tensor(hidden, dtype=torch.float32, device=self.device).reshape(1, -1),
                torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1),
                torch.as_tensor(nominal, dtype=torch.float32, device=self.device).reshape(1, -1),
            )
            risk = torch.sigmoid(outputs["risk_logits"])[0].float().cpu().numpy()
            hazard = torch.softmax(outputs["hazard_logits"], dim=-1)[0].float().cpu().numpy()
            abort = float(torch.sigmoid(outputs["abort_logit"])[0].cpu())
            severity = float(torch.expm1(outputs["severity_log"])[0].clamp_min(0).cpu())
            recovery = outputs["recovery_action"][0].float().cpu().numpy()
        return {
            "risk": risk,
            "hazard": hazard,
            "abort_probability": abort,
            "severity_force_n": severity,
            "recovery_action": recovery,
        }

    def act(self, observation: dict, instruction: str) -> np.ndarray:
        self.step_index += 1
        if self.mode == "abort":
            action = self.abort_controller.step(observation)
            decision = {
                "step": self.step_index, "mode": "abort", "intervened": True,
                "latched": True,
            }
            self.decisions.append(decision)
            self.last_decision = decision
            self.intervention_count += 1
            return np.asarray(action, dtype=np.float32)

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
        abort_probability = float(prediction["abort_probability"])

        if risk_probability >= self.risk_enter_threshold and abort_probability >= self.abort_threshold:
            self.mode = "abort"
            self.abort_controller.engage(observation)
            action = np.asarray(self.abort_controller.step(observation), dtype=np.float32)
        elif risk_probability >= self.risk_enter_threshold:
            self.mode = "recovery"
            self.recovery_steps += 1
            action = np.asarray(prediction["recovery_action"], dtype=np.float32)
        elif self.mode == "recovery" and (
            self.recovery_steps < self.min_recovery_steps
            or risk_probability > self.risk_exit_threshold
        ):
            self.recovery_steps += 1
            action = np.asarray(prediction["recovery_action"], dtype=np.float32)
        else:
            self.mode = "nominal"
            self.recovery_steps = 0
            action = nominal

        intervened = self.mode != "nominal"
        if intervened:
            self.intervention_count += 1
            if self.first_intervention_step is None:
                self.first_intervention_step = self.step_index
        decision = {
            "step": self.step_index,
            "mode": self.mode,
            "intervened": intervened,
            "risk_horizon": self.risk_horizon,
            "risk_probability": risk_probability,
            "risk_probabilities": {
                str(horizon): float(probability)
                for horizon, probability in zip(RISK_HORIZONS, prediction["risk"])
            },
            "abort_probability": abort_probability,
            "severity_force_n": float(prediction["severity_force_n"]),
            "hazard_probabilities": {
                name: float(probability)
                for name, probability in zip(HAZARD_TYPES, prediction["hazard"])
            },
        }
        self.decisions.append(decision)
        self.last_decision = decision
        return np.clip(action, -1.0, 1.0).astype(np.float32)
