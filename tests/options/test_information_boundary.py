from __future__ import annotations

import pytest

from crashbench.options.base import (
    CatalogKind,
    InformationBoundaryViolation,
    OptionSpec,
)


def test_information_view_fails_closed_on_unapproved_field():
    option = OptionSpec(
        option_id="safe_stop",
        version=1,
        catalog_kind=CatalogKind.DEPLOYABLE,
        mechanical_family="global",
        max_duration_steps=10,
        information_fields=frozenset({"proprioception"}),
        intervention_cost=0.05,
        allows_return_to_base=False,
        snapshot_contract_version=1,
    )
    view = option.view({"proprioception": [1, 2], "oracle_geometry": "secret"})
    assert view["proprioception"] == [1, 2]
    assert "oracle_geometry" not in view
    with pytest.raises(InformationBoundaryViolation):
        _ = view["oracle_geometry"]
