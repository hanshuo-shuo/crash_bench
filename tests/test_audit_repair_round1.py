"""Regression counterexamples for scientific semantics in the September audit."""
import json
from pathlib import Path

import numpy as np
import pytest

from crashbench.data.support import source_label_support, aggregate_source_support
from crashbench.models.selective import PairwiseSourceConformalSelector
from scripts.expansion.analyze_benchmark_test import support_counts
from scripts.expansion.run_round1_repair import prediction_choices, policy_metrics, paired_interval
from scripts.expansion.train_option_value import OPTION_IDS, build_training_arrays
from crashbench.data.utility import PhysicalBudgets


def test_mixed_source_counts_in_both_labels_and_empty_source_is_not_b0():
    rows = [source_label_support(x) for x in ([True, False], [False], [True], [])]
    counts = aggregate_source_support(rows)
    assert counts["contains_B0"] == counts["contains_B1"] == 2
    assert counts["contains_both"] == counts["entirely_B0"] == 1
    assert counts["benefit_zero_sources"] == 2
    assert not any(rows[-1].values())


def branches_fixture():
    anchors, branches = [], []
    for block, values in [("s:0", [1., 0., -0.3]), ("s:1", [-2., 1., -0.3])]:
        anchors.append({"block_id": block, "physical_source_id": "s"})
        for option, u in zip(OPTION_IDS, values):
            branches.append({"block_id": block, "physical_source_id": "s", "option_id": option,
                "u0": u, "outcome": {"task_success": int(u == 1), "catastrophe": int(u == -2)}})
    return anchors, branches


def test_actual_benchmark_analyzer_counts_mixed_support():
    anchors, branches = branches_fixture()
    counts = support_counts(anchors, branches)
    assert counts["benefit_zero_sources"] == counts["benefit_one_sources"] == 1
    assert counts["contains_both"] == 1 and counts["entirely_B0"] == 0


def calibration(q=0.):
    return {key: {"__global__": q, "observation_staleness_v1": q} for key in (
        "utility_pairwise_quantiles", "catastrophe_difference_quantiles", "catastrophe_absolute_quantiles")}


def test_ideal_predictions_expose_impossible_certificate_without_silently_relaxing_it():
    rows = ["a:" + option for option in OPTION_IDS]
    choices, diagnosis = prediction_choices(rows, OPTION_IDS,
        [[-2., 1., -0.3]] * 5, [[1., 0., 0.]] * 5, calibration(.97697))
    assert choices["point"]["a"] == choices["benefit_gate"]["a"] == "observation_refresh"
    assert choices["conformal"]["a"] == "safe_stop"
    assert diagnosis["status"] == "CALIBRATION_FAILURE"
    assert diagnosis["forced_safe_stop_for_all_legal_predictions"]
    assert not diagnosis["conditional_risk_guarantee"]


def test_gain_gate_and_base_tie_are_distinct_from_argmax():
    rows = ["a:" + option for option in OPTION_IDS]
    choices, _ = prediction_choices(rows, OPTION_IDS,
        [[.5, .52, -.3]] * 5, [[0., 0., 0.]] * 5, calibration())
    assert choices["point"]["a"] == "observation_refresh"
    assert choices["benefit_gate"]["a"] == "base_continue"
    choices, _ = prediction_choices(rows, OPTION_IDS,
        [[.5, .5, -.3]] * 5, [[0., 0., 0.]] * 5, calibration())
    assert choices["point"]["a"] == "base_continue"


def test_oracle_has_useful_recall_without_intervention_quota():
    _, branches = branches_fixture()
    oracle = policy_metrics(branches, {"s:0": "base_continue", "s:1": "observation_refresh"})
    base = policy_metrics(branches, {"s:0": "base_continue", "s:1": "base_continue"})
    assert oracle["B1_beneficial_intervention_recall"] == 1
    assert oracle["B0_unnecessary_intervention_rate"] == 0
    assert oracle["u0"] == 1
    assert paired_interval(oracle, base)["delta"] == 1.5


def test_cross_role_source_rejected_before_feature_access(tmp_path):
    anchors = [{"block_id": role, "physical_source_id": "same", "split_role": role}
               for role in ("train", "development")]
    with pytest.raises(ValueError, match="crosses split roles"):
        build_training_arrays(anchors, [], artifact_store=tmp_path,
            budgets=PhysicalBudgets(1, 1, 1, 1, "test"))


@pytest.mark.parametrize("architecture", ["interaction", "per_option"])
def test_repaired_model_learns_opposite_option_effects(architecture):
    torch = pytest.importorskip("torch")
    from crashbench.models.option_outcome import OptionOutcomeModel
    torch.set_num_threads(1)
    torch.manual_seed(17)
    # Same two states for Base and Refresh. Correct class flips in each state.
    x = torch.tensor([[1., -1.], [1., -1.], [-1., 1.], [-1., 1.]])
    idx = torch.tensor([0, 1, 0, 1])
    y = torch.tensor([0, 1, 1, 0])
    model = OptionOutcomeModel(input_dim=2, option_ids=("base_continue", "observation_refresh"),
        hidden_dim=16, option_embedding_dim=4, architecture=architecture)
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    for _ in range(150):
        optimizer.zero_grad()
        outputs = model(x, idx)
        loss = torch.nn.functional.cross_entropy(outputs["outcome_logits"], y)
        loss.backward()
        optimizer.step()
    logits = model(x, idx)["outcome_logits"].detach()
    assert torch.equal(logits.argmax(-1), y)
    log_odds = logits[:, 0] - logits[:, 1]
    assert log_odds[1] - log_odds[0] < -2
    assert log_odds[3] - log_odds[2] > 2


def test_legacy_additive_checkpoint_loads_as_additive(tmp_path):
    torch = pytest.importorskip("torch")
    from crashbench.models.option_outcome import OptionOutcomeModel
    from scripts.expansion.calibrate_selector import predict_seed
    model = OptionOutcomeModel(input_dim=2, option_ids=OPTION_IDS, architecture="additive")
    path = tmp_path / "old.pt"
    torch.save({"input_dim": 2, "option_ids": OPTION_IDS, "state_dict": model.state_dict()}, path)
    u, cat = predict_seed(path, {"features": np.array([[1., -1.]], dtype=np.float32), "options": np.array([OPTION_IDS[0]])})
    assert len(u) == len(cat) == 1 and np.isfinite(u).all()


def test_development_rejects_refit_and_wrong_calibration_identity(tmp_path):
    from scripts.expansion.analyze_scoped_gate_b import validate_development_model
    from scripts.expansion.calibrate_selector import sha256_file
    directory = tmp_path / "seed_0"
    directory.mkdir()
    model = directory / "model.pt"
    model.write_bytes(b"checkpoint")
    manifest_path = directory / "manifest.json"
    def save(role="train", sources=("train-source",)):
        manifest_path.write_text(json.dumps({"seed": 0, "fit_on": role,
            "fit_source_ids": list(sources), "calibration_rows_read": 0, "test_rows_read": 0}))
        return {"model_artifacts": [{"model_sha256": sha256_file(model),
                                     "manifest_sha256": sha256_file(manifest_path)}]}
    frozen = save()
    validate_development_model(directory, 0, frozen, ["dev-source"])
    frozen = save("train_development")
    with pytest.raises(ValueError, match="train-only"):
        validate_development_model(directory, 0, frozen, ["dev-source"])
    frozen = save(sources=("dev-source",))
    with pytest.raises(ValueError, match="used for model fitting"):
        validate_development_model(directory, 0, frozen, ["dev-source"])
    frozen = save()
    model.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="differ from the calibrated"):
        validate_development_model(directory, 0, frozen, ["dev-source"])


def test_retrospective_erratum_preserves_nonempty_exposed_test_support():
    root = Path(__file__).resolve().parents[1]
    erratum = json.loads((root / "docs/audits/20260906/source_support_erratum.json").read_text())
    support = erratum["D5_D8"]["D8_exposed_test"]["support"]
    assert support["source_count"] == support["contains_B0"] == 32
    assert support["contains_B1"] == support["contains_both"] == 31
    assert support["entirely_B0"] == 1
    assert not erratum["test_authorized"]
