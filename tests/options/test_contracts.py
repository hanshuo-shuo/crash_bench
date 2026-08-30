from __future__ import annotations

import json
from pathlib import Path

import pytest

from crashbench.options.base import CatalogKind, InformationBoundaryViolation, OptionSpec
from crashbench.options.registry import OptionCatalog


def spec(option_id, kind=CatalogKind.DEPLOYABLE, fields=frozenset({"proprioception"})):
    return OptionSpec(
        option_id=option_id,
        version=1,
        catalog_kind=kind,
        mechanical_family="global",
        max_duration_steps=10,
        information_fields=fields,
        intervention_cost=0.05,
        allows_return_to_base=True,
        snapshot_contract_version=1,
    )


def test_catalogs_are_separate_and_deployable_loader_rejects_oracle():
    deployable = OptionCatalog(CatalogKind.DEPLOYABLE, [spec("safe_stop")])
    oracle = OptionCatalog(
        CatalogKind.DIAGNOSTIC_ORACLE,
        [spec("oracle_path", CatalogKind.DIAGNOSTIC_ORACLE, frozenset({"oracle_geometry"}))],
    )
    deployable.assert_disjoint(oracle)
    assert deployable.get("safe_stop", consumer_kind=CatalogKind.DEPLOYABLE).option_id == "safe_stop"
    with pytest.raises(PermissionError, match="cannot load"):
        oracle.get("oracle_path", consumer_kind=CatalogKind.DEPLOYABLE)


def test_same_option_id_cannot_exist_in_deployable_and_oracle_catalogs():
    deployable = OptionCatalog(CatalogKind.DEPLOYABLE, [spec("same")])
    oracle = OptionCatalog(
        CatalogKind.DIAGNOSTIC_ORACLE,
        [spec("same", CatalogKind.DIAGNOSTIC_ORACLE)],
    )
    with pytest.raises(ValueError, match="overlap"):
        deployable.assert_disjoint(oracle)


def test_deployable_spec_cannot_declare_oracle_information():
    with pytest.raises(InformationBoundaryViolation, match="oracle fields"):
        spec("bad", fields=frozenset({"proprioception", "future_outcome"}))


def test_repository_catalogs_parse_and_are_disjoint():
    root = Path(__file__).resolve().parents[2]
    deployable = OptionCatalog.from_payload(
        json.loads((root / "configs/expansion/deployable_options_v1.yaml").read_text())
    )
    oracle = OptionCatalog.from_payload(
        json.loads((root / "configs/expansion/diagnostic_oracle_options_v1.yaml").read_text())
    )
    deployable.assert_disjoint(oracle)
    assert "deployable:observation_refresh:v1" in deployable.canonical_ids()
    assert "diagnostic_oracle:oracle_inverse_drift:v1" in oracle.canonical_ids()
