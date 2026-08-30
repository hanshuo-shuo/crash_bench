from __future__ import annotations

import pytest

from crashbench.runtime.option_router import OptionRouterRuntime, ReturnCertificate, RouterState


OPTIONS = ("base_continue", "observation_refresh", "safe_stop")


def certificate(**overrides):
    values = {
        "hazard_clear": True,
        "task_stage_valid": True,
        "low_force_velocity_consecutive_observations": 3,
        "option_state_serializable": True,
        "base_catastrophe_ucb": 0.05,
    }
    values.update(overrides)
    return ReturnCertificate(**values)


def test_option_can_return_only_after_certificate_and_base_sync():
    runtime = OptionRouterRuntime(OPTIONS)
    assert runtime.route("observation_refresh") == RouterState.OPTION_ACTIVE
    assert runtime.request_return(certificate()) == RouterState.RETURN_PENDING
    assert runtime.confirm_base_requery_reset(policy_state_synchronized=True) == RouterState.BASE
    assert [row["event"] for row in runtime.trace] == [
        "START_OPTION", "RETURN_CERTIFICATE_PASSED", "RETURN_TO_BASE"
    ]


def test_failed_certificate_or_shadow_sync_fails_closed():
    runtime = OptionRouterRuntime(OPTIONS)
    runtime.route("observation_refresh")
    assert runtime.request_return(certificate(hazard_clear=False)) == RouterState.SAFE_STOPPED
    runtime = OptionRouterRuntime(OPTIONS)
    runtime.route("observation_refresh")
    runtime.request_return(certificate())
    assert runtime.confirm_base_requery_reset(policy_state_synchronized=False) == RouterState.SAFE_STOPPED


def test_safe_stop_and_catalog_firewall():
    runtime = OptionRouterRuntime(OPTIONS)
    assert runtime.route("safe_stop") == RouterState.SAFE_STOPPED
    with pytest.raises(RuntimeError, match="Base is active"):
        runtime.route("base_continue")
    with pytest.raises(ValueError, match="outside"):
        OptionRouterRuntime(OPTIONS).route("oracle_refresh")


def test_return_certificate_requires_three_consecutive_low_force_observations():
    assert not certificate(low_force_velocity_consecutive_observations=2).passes()
    assert certificate(low_force_velocity_consecutive_observations=3).passes()
