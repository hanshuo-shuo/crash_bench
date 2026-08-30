from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from crashbench.envs.libero_adapter import LiberoSimView


def fake_view(monkeypatch):
    names = {
        ("body", 0): "world",
        ("body", 1): "table",
        ("body", 2): "movable_object",
        ("body", 3): "robot0_link7",
        ("geom", 0): "floor",
        ("geom", 1): "table_collision",
        ("geom", 2): "object_collision",
        ("geom", 3): "robot_collision",
    }
    fake_mujoco = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_BODY="body", mjOBJ_GEOM="geom"),
        mjtGeom=SimpleNamespace(mjGEOM_BOX=1, mjGEOM_PLANE=2),
        mj_id2name=lambda model, kind, index: names.get((kind, index)),
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    model = SimpleNamespace(
        ngeom=4,
        geom_bodyid=np.array([0, 1, 2, 3]),
        body_jntnum=np.array([0, 0, 1, 0]),
        geom_type=np.array([2, 1, 1, 1]),
        geom_size=np.array(
            [[1, 1, 0], [0.5, 0.5, 0.05], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
            dtype=float,
        ),
    )
    identity = np.eye(3).reshape(-1)
    data = SimpleNamespace(
        geom_xpos=np.array([[0, 0, 0], [0, 0, 0.7], [0, 0, 0.9], [0, 0, 1.0]], dtype=float),
        geom_xmat=np.stack([identity] * 4),
    )
    view = LiberoSimView.__new__(LiberoSimView)
    view._live_mj = lambda: (model, data)
    return view


def test_resolver_selects_static_table_not_dynamic_object_or_robot(monkeypatch):
    result = fake_view(monkeypatch).static_support_surface_at((0, 0), below_z=1.2)
    assert result["geom_name"] == "table_collision"
    assert result["top_z"] == pytest.approx(0.75)


def test_resolver_falls_back_to_plane_outside_table_footprint(monkeypatch):
    result = fake_view(monkeypatch).static_support_surface_at((0.8, 0.8), below_z=1.2)
    assert result["geom_name"] == "floor"
    assert result["top_z"] == 0


def test_resolver_fails_closed_when_no_support_is_below_query(monkeypatch):
    with pytest.raises(RuntimeError, match="no static horizontal support"):
        fake_view(monkeypatch).static_support_surface_at((2, 2), below_z=-1)
