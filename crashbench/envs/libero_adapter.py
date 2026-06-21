"""LIBERO/robosuite substrate adapter (PLAN.md §1, §5).

Wraps the exact API used by OpenVLA's `run_libero_eval.py` so CrashBench reuses the
proven observation/action bridge instead of rebuilding it:

    benchmark.get_benchmark_dict()[suite]()  -> task_suite
    task_suite.get_task(task_id)             -> task
    get_libero_env(task, model_family, 256)  -> (env, task_description)
    env.reset(); env.set_init_state(state)   -> obs        # our pre-crash state goes here
    env.step(action) -> obs, reward, done, info            # done == LIBERO task success
    get_libero_image(obs, resize_size)       -> img (for the policy)

`LiberoSimView` exposes the read-only quantities the predicates need. We read them
from the robosuite *observation dict* (which already contains `<object>_pos`,
`robot0_eef_pos`, `robot0_gripper_qpos`, ...) rather than poking the low-level
mujoco `sim.data`, which is both simpler and more stable across robosuite versions.

Object names in predicates use the obs convention WITHOUT the `_main` body suffix,
e.g. `akita_black_bowl_1` (obs key `akita_black_bowl_1_pos`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

DEFAULT_OPENVLA_ROOT = os.path.expanduser("~/crash_bench/third_party/openvla")


def add_openvla_to_path(openvla_root: str = DEFAULT_OPENVLA_ROOT) -> None:
    """Put the OpenVLA repo on sys.path so `experiments.robot.*` imports resolve."""
    root = str(Path(openvla_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


class LiberoSimView:
    """Read-only view backing the predicates, sourced from the robosuite obs dict.

    Implements crashbench.predicates.SimView. Updated every step by LiberoEnv.
    """

    def __init__(self, env):
        self._env = env
        self._obs: dict = {}
        self._last_done = False
        self.peak_force = 0.0  # not measured in the pilot (no contact-force predicate)

    def update(self, obs: dict, done: bool) -> None:
        self._obs = obs or {}
        self._last_done = bool(done)

    @property
    def libero_done(self) -> bool:
        return self._last_done

    def object_z(self, object_name: str) -> float:
        """World z (m) of an object, from obs `<object_name>_pos`."""
        key = f"{object_name}_pos"
        if key not in self._obs:
            raise KeyError(f"{key} not in obs; available object keys: "
                           f"{[k for k in self._obs if k.endswith('_pos')]}")
        return float(np.asarray(self._obs[key])[2])

    def is_grasped(self, object_name: str) -> bool:
        """Heuristic grasp check: object near the eef AND gripper not fully open.

        TODO(verify): replace with robosuite `env._check_grasp` for grasp_dropped
        scenarios. The pilot (unsafe-terminal / object_fell) does not use this.
        """
        rel = self._obs.get(f"{object_name}_to_robot0_eef_pos")
        grip = self._obs.get("robot0_gripper_qpos")
        if rel is None or grip is None:
            return False
        near = float(np.linalg.norm(rel)) < 0.06
        closed = float(np.sum(np.abs(grip))) < 0.06  # TODO(verify) gripper-closed threshold
        return near and closed

    def max_contact_force(self, bodies: list[str]) -> float:
        """Contact force is not read in the pilot (no contact-force predicate).

        TODO(Phase 2): wire mujoco contact forces (cfrc_ext) for env_collision scenarios.
        """
        return 0.0

    def _robot_bodies(self) -> list[str]:
        return []


class LiberoEnv:
    """Thin wrapper over a LIBERO task env using OpenVLA's verified helpers."""

    def __init__(self, task_suite: str, task_id: int, model_family: str = "openvla",
                 resolution: int = 256, openvla_root: str = DEFAULT_OPENVLA_ROOT):
        add_openvla_to_path(openvla_root)
        from libero.libero import benchmark
        from experiments.robot.libero.libero_utils import get_libero_env

        suite = benchmark.get_benchmark_dict()[task_suite]()
        self.task = suite.get_task(task_id)
        self.task_suite = task_suite
        self.task_id = task_id
        self.env, self.task_description = get_libero_env(self.task, model_family, resolution=resolution)
        self.sim_view = LiberoSimView(self.env)
        self._model_family = model_family

    def default_init_states(self) -> np.ndarray:
        from libero.libero import benchmark
        suite = benchmark.get_benchmark_dict()[self.task_suite]()
        return suite.get_task_init_states(self.task_id)

    # ---- rollout API (mirrors run_libero_eval.py) --------------------------
    def reset_to(self, init_state: np.ndarray):
        self.env.reset()
        obs = self.env.set_init_state(init_state)
        self.sim_view.update(obs, done=False)
        return obs

    def dummy_action(self):
        from experiments.robot.libero.libero_utils import get_libero_dummy_action
        return get_libero_dummy_action(self._model_family)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.sim_view.update(obs, done=done)
        return obs, reward, done, info

    def render(self, obs, resize_size):
        from experiments.robot.libero.libero_utils import get_libero_image
        return get_libero_image(obs, resize_size)

    def policy_observation(self, obs, resize_size):
        """Build the dict OpenVLA's get_action expects (image + proprio state)."""
        from experiments.robot.libero.libero_utils import get_libero_image, quat2axisangle
        img = get_libero_image(obs, resize_size)
        return {
            "full_image": img,
            "state": np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ),
        }
