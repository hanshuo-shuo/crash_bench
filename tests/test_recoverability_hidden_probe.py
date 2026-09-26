"""Synthetic checks of scientific leakage boundaries and metric semantics."""
import unittest
import json
from pathlib import Path
import tempfile
import numpy as np

from scripts.analysis.recoverability_hidden_probe import (
    binary_metrics, bootstrap_comparison, fit_logistic, fold_indices, predict,
    source_weights, fit_suite, summarize,
)
from scripts.train_minimal_counterfactual_router import fit_frame_pca


class RecoverabilityProbeTests(unittest.TestCase):
    def test_source_folds_keep_all_times_and_conditions_together(self):
        sources = np.repeat(["a", "b", "c", "d"], 6)
        conditions = np.tile(["glass", "noglass", "offpath"], 8)
        splits = np.repeat(["train", "calibration", "development", "development"], 6)
        for regime in ("pooled_loso", "glass_loso", "historical_train_to_development"):
            seen = []
            for _, train, test in fold_indices(sources, conditions, splits, regime):
                self.assertFalse(set(sources[train]) & set(sources[test]))
                seen.extend(test)
                if regime == "glass_loso":
                    self.assertTrue(np.all(conditions[np.r_[train, test]] == "glass"))
                if regime.startswith("historical"):
                    self.assertTrue(np.all(splits[train] == "train"))
            self.assertEqual(len(seen), len(set(seen)))

    def test_evaluation_features_cannot_change_training_pca_or_scaler(self):
        rng = np.random.default_rng(7)
        hidden = rng.normal(size=(12, 1, 20))
        train = np.arange(12) < 8
        a = fit_frame_pca(hidden, np.ones((12, 1)), train, 3)
        hidden[~train] += 1e6
        b = fit_frame_pca(hidden, np.ones((12, 1)), train, 3)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_metrics_perfect_tied_reversed_and_single_class(self):
        y = np.array([0, 1, 0, 1])
        self.assertEqual(binary_metrics(y, y)["auc"], 1)
        self.assertEqual(binary_metrics(y, 1 - y)["auc"], 0)
        tied = binary_metrics(y, np.full(4, .5))
        self.assertEqual(tied["auc"], .5)
        self.assertEqual(tied["ap"], .5)
        self.assertEqual(tied["brier"], .25)
        self.assertIsNone(binary_metrics(np.ones(4), y)["auc"])
        self.assertIsNone(binary_metrics(np.zeros(4), y)["ap"])

    def test_each_source_has_equal_loss_weight(self):
        sources = np.array(["a", "b", "b", "b"])
        w = source_weights(sources)
        self.assertAlmostEqual(w[0], w[1:].sum())
        model, info = fit_logistic(np.empty((4, 0)), np.array([1, 0, 0, 0]), sources)
        self.assertTrue(info["converged"])
        np.testing.assert_allclose(predict(model, np.empty((4, 0))), .5, atol=1e-8)

    def test_logistic_recovers_direction_and_handles_one_class(self):
        x = np.linspace(-3, 3, 30)[:, None]
        sources = np.repeat(["a", "b", "c"], 10)
        model, _ = fit_logistic(x, (x[:, 0] > 0).astype(int), sources)
        self.assertGreater(predict(model, x)[-1], .9)
        self.assertLess(predict(model, x)[0], .1)
        constant, info = fit_logistic(x, np.zeros(30), sources)
        self.assertTrue(info["single_class"])
        np.testing.assert_array_equal(predict(constant, x), np.zeros(30))

    def test_paired_bootstrap_preserves_identity_and_reports_null_draws(self):
        rows = [{"source": str(i), "decision_id": str(i), "y": i % 2, "score": float(i % 2)} for i in range(4)]
        result = bootstrap_comparison(rows, rows, 100, 7)
        self.assertEqual(result["difference_auc"]["estimate"], 0)
        self.assertEqual(result["difference_auc"]["ci95"], [0., 0.])
        self.assertLess(result["left_auc"]["valid_repeats"], 100)
        with self.assertRaises(ValueError):
            bootstrap_comparison(rows, rows[::-1], 2, 7)

    def test_synthetic_fit_artifacts_and_analysis_remain_joined(self):
        rng = np.random.default_rng(3)
        n = 12
        sources = np.repeat(["a", "b", "c"], 4)
        conditions = np.tile(["glass", "glass", "noglass", "offpath"], 3)
        splits = np.repeat(["train", "calibration", "development"], 4)
        metadata = [{"decision_id": str(i), "source_state_sha256": str(sources[i]),
                     "condition": str(conditions[i]), "split": str(splits[i]),
                     "horizon_actions": 5, "placement_key": str(sources[i])} for i in range(n)]
        outcomes = np.tile(["task_success", "catastrophe", "safe_noncompletion"], (n, 1))
        outcomes[::2, 0] = "catastrophe"
        outcomes[1::3, 1] = "task_success"
        data = {"sources": sources, "conditions": conditions, "splits": splits,
                "horizons": np.full(n, 5), "metadata": metadata, "outcomes": outcomes,
                "arrays": {"hidden": rng.normal(size=(n, 8, 20)),
                           "robot_state": rng.normal(size=(n, 8, 8)),
                           "nominal_action": rng.normal(size=(n, 8, 7))}}
        config = {"regimes": ["pooled_loso"], "feature_sets": ["hidden", "prior"],
                  "targets": ["base_crash", "detour_success", "base_success"],
                  "pca_components": 3, "seed": 7, "l2": .01}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            predictions = fit_suite(data, config, output)
            self.assertEqual(len(predictions), n * 7)
            inverse = [r for r in predictions if r["feature"] == "inverse_risk_hidden"]
            self.assertEqual(len(inverse), n)
            self.assertTrue(all(r["y"] == r["detour_success"] for r in inverse))
            metrics = summarize(predictions, output)
            self.assertTrue(metrics)
            self.assertTrue((output / "models.npz").is_file())
            for fold in json.loads((output / "folds.json").read_text()):
                self.assertFalse(set(fold["train_sources"]) & set(fold["test_sources"]))


if __name__ == "__main__":
    unittest.main()
