from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from crashbench.glass_detector import (
    DETECTOR_HORIZON_ACTIONS,
    GlassRiskDetector,
    episode_operating_metrics,
    fit_glass_detector,
    risk_frame_targets,
    select_operating_point,
    validate_capture_arrays,
    validate_source_split,
)
from scripts.prepare_glass_detector_split import _allocation, prepare
from scripts.fit_glass_detector import fit as fit_d0
import scripts.capture_glass_detector_placements as placement_capture
from scripts.capture_glass_detector_placements import (
    _episode_exclusion,
    build_capture_plan,
)


def _episode_rows(
    episode: str,
    source: str,
    condition: str,
    *,
    crashed: bool,
    steps: int = 25,
) -> list[dict]:
    return [
        {
            "episode_id": episode,
            "source_state_sha256": source,
            "condition": condition,
            "t": index,
            "crashed_episode": crashed,
            "time_to_catastrophe_actions": steps - index if crashed else None,
        }
        for index in range(steps)
    ]


def test_operating_point_uses_exact_t20_anchor_and_full_control_maximum():
    rows = (
        _episode_rows("treatment", "source_t", "glass", crashed=True)
        + _episode_rows("offpath", "source_o", "offpath", crashed=False)
        + _episode_rows("noglass", "source_n", "noglass", crashed=False)
    )
    scores = np.full(len(rows), -1.0)
    treatment = rows[:25]
    anchor = next(
        index for index, row in enumerate(treatment)
        if row["time_to_catastrophe_actions"] == DETECTOR_HORIZON_ACTIONS
    )
    scores[anchor] = 0.8
    scores[25 + 7] = 0.6
    scores[50 + 9] = 0.5

    metrics = episode_operating_metrics(
        scores,
        rows,
        np.ones(len(rows), dtype=bool),
        max_control_episode_fpr=0.10,
    )

    assert metrics["threshold"] == pytest.approx(0.8)
    assert metrics["timely_trigger_rate"] == 1.0
    assert metrics["control_episode_fpr"] == 0.0
    assert metrics["calibration_unit"].startswith("exact_T20")
    assert metrics["control_episode_fpr_by_condition"] == {
        "offpath": 0.0,
        "noglass": 0.0,
    }


def test_t5_only_signal_is_not_mislabeled_as_t20_deadline_evidence():
    rows = (
        _episode_rows("treatment", "source_t", "glass", crashed=True)
        + _episode_rows("offpath", "source_o", "offpath", crashed=False)
    )
    scores = np.zeros(len(rows))
    for index, row in enumerate(rows[:25]):
        if row["time_to_catastrophe_actions"] <= 5:
            scores[index] = 0.9
    scores[25 + 3] = 0.4

    metrics = episode_operating_metrics(
        scores,
        rows,
        np.ones(len(rows), dtype=bool),
        max_control_episode_fpr=0.0,
    )

    assert metrics["timely_trigger_rate"] == 0.0
    assert metrics["has_positive_operating_point"] is False
    assert metrics["threshold"] > 0.4
    assert metrics["threshold"] < 0.9


def test_threshold_is_score_space_value_not_probability_sentinel():
    selected = select_operating_point(
        [0.8, 0.9], [0.4, 0.5], max_control_episode_fpr=0.0
    )
    assert selected["threshold"] == pytest.approx(0.8)
    assert selected["timely_trigger_rate"] == 1.0
    assert selected["control_episode_fpr"] == 0.0


@pytest.mark.parametrize(
    "variant", ("hidden_robot_action", "hidden_only", "robot_action_only")
)
def test_detector_round_trip_for_all_feature_contracts(tmp_path: Path, variant: str):
    rng = np.random.default_rng(4)
    hidden = rng.normal(size=(80, 6)).astype(np.float32)
    robot = rng.normal(size=(80, 8)).astype(np.float32)
    action = rng.normal(size=(80, 7)).astype(np.float32)
    if variant == "hidden_only":
        labels = (hidden[:, 0] > 0).astype(np.float32)
    elif variant == "robot_action_only":
        labels = (robot[:, 0] + action[:, 0] > 0).astype(np.float32)
    else:
        labels = (hidden[:, 0] + robot[:, 0] + action[:, 0] > 0).astype(np.float32)
    detector = fit_glass_detector(
        hidden,
        robot,
        action,
        labels,
        variant=variant,
        pca_components=4,
        iterations=100,
        seed=7,
        metadata={"horizon_actions": 20},
    ).with_threshold(0.25, {"split": "calibration"})
    path = tmp_path / f"{variant}.npz"
    detector.save(path)
    loaded = GlassRiskDetector.load(path)

    np.testing.assert_allclose(
        detector.score(hidden, robot, action),
        loaded.score(hidden, robot, action),
        rtol=1e-6,
        atol=1e-6,
    )
    assert loaded.variant == variant
    assert loaded.threshold == pytest.approx(0.25)
    assert loaded.metadata["calibration"] == {"split": "calibration"}


def test_capture_and_split_contract_reject_leakage_prerequisites():
    rows = (
        _episode_rows("train", "source_train", "glass", crashed=True)
        + _episode_rows("cal", "source_cal", "offpath", crashed=False)
        + _episode_rows("dev", "source_dev", "noglass", crashed=False)
    )
    n_rows = len(rows)
    validate_capture_arrays(
        np.zeros((n_rows, 6)),
        np.zeros((n_rows, 8)),
        np.zeros((n_rows, 7)),
        rows,
    )
    by_split = validate_source_split(rows, {
        "source_train": "train",
        "source_cal": "calibration",
        "source_dev": "development",
    })
    assert by_split == {
        "train": ["source_train"],
        "calibration": ["source_cal"],
        "development": ["source_dev"],
    }
    with pytest.raises(ValueError, match="missing"):
        validate_source_split(rows, {
            "source_train": "train",
            "source_cal": "calibration",
        })


def test_right_censored_onpath_frames_are_not_negative_controls():
    rows = (
        _episode_rows("crash", "source_a", "glass", crashed=True)
        + _episode_rows("censored", "source_b", "glass", crashed=False)
        + _episode_rows("control", "source_c", "noglass", crashed=False)
    )
    targets, usable = risk_frame_targets(rows)
    assert usable[:25].all()
    assert not usable[25:50].any()
    assert usable[50:].all()
    assert targets[:5].sum() == 0
    assert targets[5:25].sum() == DETECTOR_HORIZON_ACTIONS


def test_source_split_is_condition_stratified_and_outcome_blind(tmp_path: Path):
    rows = []
    for condition in ("glass", "offpath", "noglass"):
        for source_index in range(6):
            rows.append({
                "source_state_sha256": f"{condition}_{source_index}",
                "condition": condition,
                # The split preparer must not read either field.
                "crashed_episode": bool(source_index % 2),
                "score": 1000.0 * source_index,
            })
    metadata = tmp_path / "meta.json"
    metadata.write_text(json.dumps(rows))
    output = tmp_path / "split.json"
    payload = prepare(SimpleNamespace(
        metadata=str(metadata),
        output=str(output),
        overwrite=False,
        horizon_actions=20,
        train_fraction=0.5,
        calibration_fraction=0.25,
        seed=7,
    ))

    assert payload["assignment"]["uses_outcomes_or_model_scores"] is False
    assert len(payload["source_splits"]) == 18
    for signature in payload["assignment"]["strata"]:
        assert all(signature["assigned"][split] > 0 for split in (
            "train", "calibration", "development"
        ))
    assert _allocation(6, (0.5, 0.25, 0.25)) == [3, 2, 1]


def test_d0_fit_writes_primary_and_both_baselines(tmp_path: Path):
    rng = np.random.default_rng(11)
    rows = []
    hidden, robot, action = [], [], []
    source_splits = {}
    for split in ("train", "calibration", "development"):
        for condition in ("glass", "offpath", "noglass"):
            source = f"{split}_{condition}"
            source_splits[source] = split
            episode_rows = _episode_rows(
                f"{split}_{condition}_episode",
                source,
                condition,
                crashed=condition == "glass",
            )
            for row in episode_rows:
                risk = float(
                    condition == "glass"
                    and row["time_to_catastrophe_actions"] <= 20
                )
                rows.append(row)
                hidden.append(rng.normal(size=6) + risk)
                robot.append(rng.normal(size=8))
                action.append(rng.normal(size=7))
    capture = tmp_path / "hidden.npz"
    np.savez_compressed(
        capture,
        hidden=np.asarray(hidden, dtype=np.float32),
        robot_state=np.asarray(robot, dtype=np.float32),
        nominal_action=np.asarray(action, dtype=np.float32),
    )
    metadata = tmp_path / "meta.json"
    metadata.write_text(json.dumps(rows))
    split_manifest = tmp_path / "source_split.json"
    split_manifest.write_text(json.dumps({
        "kind": "glass_detector_source_split",
        "horizon_actions": 20,
        "source_splits": source_splits,
    }))
    output = tmp_path / "fit"
    summary = fit_d0(SimpleNamespace(
        output=str(output),
        overwrite=False,
        horizon_actions=20,
        capture=str(capture),
        metadata=str(metadata),
        split_manifest=str(split_manifest),
        pca_components=4,
        l2=2.0,
        iterations=30,
        learning_rate=0.2,
        seed=3,
        max_control_episode_fpr=0.10,
        min_timely_trigger_rate=0.80,
        min_calibration_source_states=5,
    ))

    assert summary["primary_variant"] == "hidden_robot_action"
    assert set(summary["variants"]) == {
        "hidden_robot_action", "hidden_only", "robot_action_only"
    }
    assert (output / "glass_detector.npz").is_file()
    assert (output / "baseline_hidden_only.npz").is_file()
    assert (output / "baseline_robot_action_only.npz").is_file()
    assert (output / "d0_summary.json").is_file()


def test_placement_capture_freezes_exposed_and_fresh_source_roles(
    monkeypatch: pytest.MonkeyPatch,
):
    exposed = [
        SimpleNamespace(
            placement_id=f"exposed_{split}", split=split,
            source_state_sha256=f"source_{split}",
        )
        for split in ("train", "validation", "heldout")
    ]
    fresh = [
        SimpleNamespace(
            placement_id=f"fresh_{split}", split=split,
            source_state_sha256=f"fresh_source_{split}",
        )
        for split in ("train", "validation", "heldout")
    ]

    def fake_read(path):
        return (fresh if "fresh" in str(path) else exposed), {}

    monkeypatch.setattr(placement_capture, "read_placement_manifest", fake_read)
    plan, source_splits = build_capture_plan("exposed.json", "fresh.json")

    assert len(plan) == 6
    assert source_splits["source_train"] == "train"
    assert source_splits["source_validation"] == "calibration"
    assert source_splits["source_heldout"] == "calibration"
    assert all(
        source_splits[f"fresh_source_{split}"] == "development"
        for split in ("train", "validation", "heldout")
    )


def test_placement_capture_excludes_invalid_control_and_missing_t20_anchor():
    assert _episode_exclusion("offpath", {
        "crashed": True, "collision_step": None,
        "initial_predicate_error": "already true",
    }) == "offpath_predicate_true_before_first_action"
    assert _episode_exclusion("offpath", {
        "crashed": True, "collision_step": 30,
    }) == "offpath_not_a_clean_control"
    assert _episode_exclusion("glass", {
        "crashed": False, "collision_step": None,
    }) == "onpath_no_catastrophe"
    assert _episode_exclusion("glass", {
        "crashed": True, "collision_step": 18,
    }) == "onpath_collision_before_T20_anchor"
    assert _episode_exclusion("glass", {
        "crashed": True, "collision_step": 19,
    }) is None
    assert _episode_exclusion("noglass", {
        "crashed": False, "collision_step": None,
    }) is None


def test_placement_capture_turns_initial_predicate_failure_into_exclusion(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeEnv:
        sim_view = object()

        def reset_to(self, source_state, movable_objects=None):
            return {}

    def reject(_crash, _sim):
        raise placement_capture.CandidateRejected(
            "invalid_initial_state", "crash predicate is already true"
        )

    monkeypatch.setattr(placement_capture, "build_any", lambda _specs: object())
    monkeypatch.setattr(placement_capture, "_glass_predicate_specs", lambda _glass: [])
    monkeypatch.setattr(placement_capture, "_prime_glass_predicates", reject)
    monkeypatch.setattr(placement_capture, "_glass_force", lambda *_args: 3.5)
    placement = SimpleNamespace(on_path_glass={"name": "glass"})
    result = placement_capture._rollout(
        FakeEnv(),
        object(),
        np.zeros(2),
        placement,
        "glass",
        settle_steps=0,
        max_steps=20,
    )

    assert result["rows"] == []
    assert result["crashed"] is True
    assert result["collision_step"] is None
    assert result["peak_glass_force_n"] == 3.5
    assert _episode_exclusion("glass", result) == (
        "glass_predicate_true_before_first_action"
    )
