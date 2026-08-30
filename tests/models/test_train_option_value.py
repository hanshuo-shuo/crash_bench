from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/train_option_value.py"
SPEC = importlib.util.spec_from_file_location("train_option_value", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_expected_u0_penalizes_catastrophe_intervention_and_costs():
    probabilities = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    costs = np.zeros((2, 4), dtype=np.float32)
    options = np.array(["base_continue", "safe_stop"])
    values = MODULE.expected_u0(probabilities, costs, options)
    assert values[0] == 1.0
    assert values[1] == pytest.approx(-2.05)


def test_training_script_explicitly_excludes_calibration_and_test_roles():
    assert MODULE.ALLOWED_ROLES == {"train", "development"}


def test_two_stage_fit_modes_are_explicit_in_cli_source():
    source = SCRIPT.read_text()
    assert 'choices=("train", "train_development")' in source
    assert 'args.fit_on == "train"' in source


def test_feature_loader_revalidates_content_address_and_split_identity():
    source = SCRIPT.read_text()
    assert "store.validate(blob_ref)" in source
    assert "anchor/branch split role mismatch" in source
    assert "materialized feature cache lacks blob" in source
    assert "materialized feature cache SHA-256 mismatch" in source
