from __future__ import annotations

from types import SimpleNamespace

from crashbench.envs.libero_adapter import LiberoEnv


def test_reset_fresh_seeds_updates_model_identity_and_view():
    calls = []
    env = LiberoEnv.__new__(LiberoEnv)
    env.seed = lambda seed: calls.append(("seed", seed))
    env.env = SimpleNamespace(
        reset=lambda: {"obs": [1]},
        sim=SimpleNamespace(model=SimpleNamespace(get_xml=lambda: "<xml/>")),
    )
    env.sim_view = SimpleNamespace(
        peak_force=9,
        update=lambda obs, done: calls.append(("update", obs, done)),
    )
    obs = env.reset_fresh(17)
    assert obs == {"obs": [1]}
    assert env._continuation_model_xml == "<xml/>"
    assert env.sim_view.peak_force == 0
    assert calls == [("seed", 17), ("update", {"obs": [1]}, False)]
