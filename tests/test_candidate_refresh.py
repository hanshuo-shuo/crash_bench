import copy
import json
import numpy as np
import pytest
from test_selection_retest import FakeEnv, FakePolicy
from scripts.expansion.candidate_refresh import (rollout, TimedQueue, select_panel,
    SPLIT, freeze_horizons, at_horizon)
from scripts.expansion.analyze_candidate_refresh import summarize


class HorizonEnv(FakeEnv):
    def __init__(self, success_at=None, accident_at=None, terminate_at=None):
        super().__init__()
        self.success_at=success_at
        self.accident_at=accident_at
        self.terminate_at=terminate_at
        self.sim_view.max_contact_force=lambda _: 75 if self.t==self.accident_at else 0
    def episode_terminated(self):
        return self.t==self.terminate_at
    def step(self,action):
        obs,_,_,info=super().step(action)
        return obs,0,self.t==self.success_at,info


@pytest.mark.parametrize('success,accident,terminate,expected',[
    (99,None,None,(1,1)),(100,None,None,(1,1)),(101,None,None,(0,1)),
    (200,None,None,(0,1)),(None,None,None,(0,0)),
    (101,101,None,(0,0)),(None,None,150,(0,0))])
def test_nested_horizons_and_terminal_precedence(tmp_path,success,accident,terminate,expected):
    env=HorizonEnv(success,accident,terminate)
    r=rollout(env,FakePolicy(),TimedQueue(1),{'anchor_steps':5,'condition':'stale'},1,tmp_path/'trace.gz')
    assert tuple(r['horizons'][h]['task_success'] for h in ('100','200'))==expected
    assert r['horizons']['100']['steps']<=100
    assert r['horizons']['200']['steps']<=200
    if accident: assert r['catastrophe']==1
    if terminate: assert r['termination_reason']=='environment_termination'


def test_frozen_candidate_census():
    rows=select_panel(json.loads(SPLIT.read_text()))
    stale=[r for r in rows if r['condition']=='stale']
    assert len(stale)==9 and len({r['physical_source_id'] for r in stale})==8
    assert sum(r['historical_base']['catastrophe'] for r in stale)==1
    assert sorted(r['historical_refresh']['steps'] for r in stale)[-4:]==[96,96,99,99]
    for i in range(0,18,2):
        assert rows[i]['candidate_id']==rows[i+1]['candidate_id']
        assert rows[i]['physical_source_id']==rows[i+1]['physical_source_id']


def test_horizon_selection_uses_only_a_and_exposes_catchup():
    anchors=[dict(panel_id='b0',physical_source_id='s0',task_id='t0',condition='stale',age_steps=1,anchor_steps=5)]
    phases={}
    for phase,repeats in [('A',range(4)),('B',range(4,8))]:
        records=[]
        for repeat in repeats:
            for option in (0,1):
                horizons={str(h):dict(task_success=int(h==200 or option==1),catastrophe=0,
                    steps=90 if option else min(h,150),path_length_m=1.,inference_calls=20,
                    termination_reason='success' if h==200 or option else 'timeout') for h in (100,200)}
                records.append(dict(panel_id='b0',phase=phase,repeat=repeat,option=option,horizons=horizons))
        phases[phase]=records
    frozen=freeze_horizons(anchors,phases['A'])
    assert frozen['100']['reference']['b0']==1
    assert frozen['200']['reference']['b0']==0
    with pytest.raises(ValueError):freeze_horizons(anchors,phases['B'])
    result=summarize(anchors,phases['A'],phases['B'],frozen,bootstrap=10)
    assert result['anchors'][0]['paired_B']['base_catches_up_by_200']==4
    assert result['groups']['200']['stale']['table'][1]['success_delta']==0
