"""CrashBench: controlled diagnosis and routing repair for VLA collisions.

See README.md and docs/CURRENT.md for the current design. The core loop:

    reset env to scenario.initial_state
    for t in range(max_steps):
        obs    = env.render()
        action = policy(obs, scenario.instruction)
        env.step(action)
        if crash_predicate(sim):   -> CRASH
        if success_predicate(sim): -> RECOVERY_SUCCESS
    else:                          -> SAFE_ABORT or TIMEOUT
"""

from crashbench.scenario import Scenario, PredicateSpec
from crashbench.eval import Outcome, EpisodeResult, run_episode

__all__ = ["Scenario", "PredicateSpec", "Outcome", "EpisodeResult", "run_episode"]
