from collections import deque
from types import SimpleNamespace
import numpy as np
from crashbench.policies.pi0_policy import Pi0Policy
from crashbench.mechanisms.observation_staleness import ObservationDelayQueue
from scripts.expansion.run_staleness_screen import mechanism_observation
from scripts.expansion.probe_neutral_refresh import trace_difference


def test_policy_episode_reset_does_not_reset_rng():
    policy=Pi0Policy.__new__(Pi0Policy)
    policy._queue=deque([np.ones(7)])
    policy._policy=SimpleNamespace(_rng=1234)
    policy.reset()
    assert not policy._queue
    assert policy._policy._rng==1234


def test_neutral_control_delivery_equals_refresh_at_every_step():
    for condition in ['fresh_control','matched_buffer_control']:
        delay=0 if condition=='fresh_control' else 3
        base=ObservationDelayQueue(delay);refresh=ObservationDelayQueue(delay)
        for i in range(20):
            obs={'state':np.array([float(i)]),'image':np.full((2,2,3),i,dtype=np.uint8)}
            a=mechanism_observation(base,obs,condition=condition)
            b=refresh.refresh(obs) if i==5 else mechanism_observation(refresh,obs,condition=condition)
            assert all(np.array_equal(a[k],b[k]) for k in a)


def test_trace_diagnostic_reports_first_divergence_and_length():
    a=[dict(observation_sha256='a',policy_continuation_sha256='b',action=[0,1],state=[1,2],force=0)]*3
    b=[dict(x) for x in a];b[1]['action']=[0,1.1]
    d=trace_difference(a,b)
    assert not d['identical'] and d['first_differing_step_0based']['action']==1
    assert d['first_differing_step_0based']['state'] is None
    assert not trace_difference(a,b[:2])['identical']
    assert trace_difference(a,a)['identical']
