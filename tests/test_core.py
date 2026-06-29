"""Pure-logic tests for the CrashBench core (no GPU / no LIBERO needed).

Run on any node:  python -m pytest tests/ -q   (or just python tests/test_core.py)
"""

from __future__ import annotations

import tempfile

import numpy as np

from crashbench.scenario import Scenario, PredicateSpec
from crashbench.predicates import build_predicate, build_any
from crashbench.eval import EpisodeResult, Outcome
from crashbench import metrics


class FakeSim:
    """Stand-in for the env's SimView so predicates can be tested without robosuite."""
    def __init__(self, done=False, force=0.0, z=1.0, grasped=True, xy=(0.0, 0.0)):
        self._done, self._force, self._z, self._grasped = done, force, z, grasped
        self._xy = xy
    @property
    def libero_done(self): return self._done
    def max_contact_force(self, bodies, against=None): return self._force
    def object_z(self, name): return self._z
    def object_xy(self, name): return self._xy
    def is_grasped(self, name): return self._grasped


def test_scenario_roundtrip():
    sc = Scenario(
        id="env_collision__T5__libero_spatial_t0_000",
        category="env_collision", horizon="T-5",
        task_suite="libero_spatial", task_id=0,
        instruction="pick up the black bowl",
        init_state=np.arange(10, dtype=np.float64),
        crash_predicates=[PredicateSpec("contact_force", {"bodies": ["robot0_link6"], "threshold": 20.0})],
        success_predicate=PredicateSpec("libero_task_success", {}),
    )
    with tempfile.TemporaryDirectory() as d:
        sc.save(d)
        sc2 = Scenario.load(f"{d}/{sc.id}")
    assert sc2.id == sc.id and sc2.category == sc.category
    assert np.allclose(sc2.init_state, sc.init_state)
    assert sc2.crash_predicates[0].params["threshold"] == 20.0


def test_predicates():
    cf = build_predicate(PredicateSpec("contact_force", {"bodies": ["b"], "threshold": 20.0}))
    assert cf(FakeSim(force=25.0)) is True
    assert cf(FakeSim(force=5.0)) is False

    fell = build_predicate(PredicateSpec("object_fell", {"object_name": "bowl", "table_z": 0.41}))
    assert fell(FakeSim(z=0.30)) is True
    assert fell(FakeSim(z=0.45)) is False

    # object_displaced: baseline captured on first call (post-settle), then horizontal move
    disp = build_predicate(PredicateSpec("object_displaced", {"object_name": "box", "max_disp": 0.06}))
    assert disp(FakeSim(xy=(0.0, 0.0))) is False        # first call sets baseline
    assert disp(FakeSim(xy=(0.03, 0.0))) is False        # within tolerance
    assert disp(FakeSim(xy=(0.10, 0.0))) is True         # swept > 6 cm

    drop = build_predicate(PredicateSpec("grasp_dropped", {"object_name": "bowl", "init_z": 0.95}))
    assert drop(FakeSim(grasped=False, z=0.70)) is True   # released and fell
    assert drop(FakeSim(grasped=True, z=0.70)) is False   # still held

    succ = build_predicate(PredicateSpec("libero_task_success", {}))
    assert succ(FakeSim(done=True)) is True

    any_crash = build_any([PredicateSpec("contact_force", {"bodies": ["b"], "threshold": 20.0})])
    assert any_crash(FakeSim(force=99.0)) is True


def test_metrics():
    rs = [
        EpisodeResult("a", "env_collision", "T-5", Outcome.CRASH, 3, 50.0, True, False),
        EpisodeResult("b", "env_collision", "T-5", Outcome.RECOVERY_SUCCESS, 40, 0.5, False, True),
        EpisodeResult("c", "grasp_instability", "T-5", Outcome.SAFE_ABORT, 220, 0.2, False, False),
        EpisodeResult("d", "grasp_instability", "T-1", Outcome.CRASH, 1, 80.0, True, False),
    ]
    s = metrics.summarize(rs)
    assert s.n == 4
    assert abs(s.crash_rate - 0.5) < 1e-9
    assert abs(s.impact_severity - 65.0) < 1e-9     # mean of {50, 80}
    # headline = crash rate @ T-5 averaged across categories:
    #   env_collision T-5 crash = 1/2 ; grasp_instability T-5 crash = 0/1 -> mean = 0.25
    assert abs(metrics.headline_crash_rate(rs) - 0.25) < 1e-9
    print(metrics.report(rs))


if __name__ == "__main__":
    test_scenario_roundtrip()
    test_predicates()
    test_metrics()
    print("\nall core tests passed ✓")
