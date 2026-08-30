from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from crashbench.envs.libero_adapter import LiberoEnv


def test_default_init_states_explicitly_disables_weights_only(monkeypatch):
    calls = []

    class Suite:
        def get_task(self, task_id):
            return SimpleNamespace(problem_folder="folder", init_states_file="states.pt")

    fake_benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"suite": Suite})
    fake_libero = SimpleNamespace(
        benchmark=fake_benchmark,
        get_libero_path=lambda key: "/trusted/init_states",
    )

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return [[1, 2], [3, 4]]

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=load))
    monkeypatch.setitem(sys.modules, "libero.libero", fake_libero)
    env = LiberoEnv.__new__(LiberoEnv)
    env.task_suite = "suite"
    env.task_id = 2
    values = env.default_init_states()
    np.testing.assert_array_equal(values, [[1, 2], [3, 4]])
    assert calls == [("/trusted/init_states/folder/states.pt", {"weights_only": False})]
