#!/usr/bin/env python
"""Diagnostic: is the steering write-hook actually perturbing OpenVLA's forward pass, and why is
the alpha sweep null? Runs one fixed observation through the policy at increasing alpha (including
an extreme alpha as a no-op bug-detector) and reports:
  - ||action(alpha) - action(0)||  : does the emitted action move at all?
  - ||W_action @ d||                : does the probe direction project onto the action-token logits?
                                      (if ~0 => true orthogonality negative; if large but action
                                      unchanged => the hook isn't firing => bug)
"""
from __future__ import annotations
import os
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import glob
from crashbench.scenario import Scenario
from crashbench.envs import LiberoEnv
from crashbench.policies import OpenVLAPolicy
from crashbench.probe import Probe

probe = Probe.load()
d = probe.steer_vector()
policy = OpenVLAPolicy(pretrained_checkpoint="openvla/openvla-7b-finetuned-libero-spatial",
                       unnorm_key="libero_spatial", center_crop=True,
                       enable_steering=True, capture_hidden=True)
env = LiberoEnv("libero_spatial", 0)
sc = Scenario.load(os.path.dirname(sorted(glob.glob("scenarios/*/scenario.json"))[0]))
obs = env.reset_to(sc.init_state, obstacles=(sc.obstacles or None))
for _ in range(10):
    obs, _, _, _ = env.step(env.dummy_action())
observation = env.policy_observation(obs, policy.resize_size)

# direct readout test: does d project onto the action-token rows of lm_head?
import torch
W = policy.model.language_model.lm_head.weight.detach()        # [vocab, hidden]
dt = torch.as_tensor(d, dtype=W.dtype, device=W.device)
n_act = policy.model.get_action_dim("libero_spatial")
vocab = W.shape[0]
# OpenVLA action tokens map to the last 256 vocab ids (bins). Look at that slice.
act_rows = W[vocab - 256:]                                     # [256, hidden]
proj_all = float(torch.linalg.norm(W @ dt))
proj_act = float(torch.linalg.norm(act_rows @ dt))
print(f"||W_full @ d|| = {proj_all:.3f}   ||W_action(256) @ d|| = {proj_act:.3f}   "
      f"(d unit; logit shift at alpha is alpha*these)")

print("\nalpha   action(7-dof)                                            ||da|| vs alpha=0")
a0 = None
for alpha in [0.0, 20.0, 80.0, 200.0, 1000.0]:
    policy.set_steering(d, alpha)
    a = np.asarray(policy.act(observation, sc.instruction), dtype=np.float32)
    if a0 is None:
        a0 = a.copy()
    da = float(np.linalg.norm(a - a0))
    print(f"{alpha:7.0f} {np.array2string(a, precision=3, suppress_small=True):52s} {da:.4f}")
