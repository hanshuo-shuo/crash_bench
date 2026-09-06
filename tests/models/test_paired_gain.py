import numpy as np
import pytest
from crashbench.models.paired_gain import (
    OPTION_IDS, pair_decisions, train_standardization, choose_from_gains, tiny_train_sources, PairedGainModel,
)
from scripts.expansion.run_round2_gain import gain_diagnostics, recovery_metrics


def fixture_data():
    return dict(
        row_ids=np.array(['a:safe_stop', 'b:observation_refresh', 'a:base_continue', 'b:base_continue', 'a:observation_refresh', 'b:safe_stop']),
        options=np.array(['safe_stop', 'observation_refresh', 'base_continue', 'base_continue', 'observation_refresh', 'safe_stop']),
        actual_u0=np.array([-.3, 1., -2., 1., 1., -.3], dtype=np.float32),
        outcomes=np.array([2,0,1,0,0,2]), costs=np.zeros((6,4), dtype=np.float32),
        features=np.array([[1.,2.],[3.,4.],[1.,2.],[3.,4.],[1.,2.],[3.,4.]], dtype=np.float32),
        sources=np.array(['s1','s2','s1','s2','s1','s2']), roles=np.array(['train']*6))


def test_pairing_uses_identity_and_base_difference_not_row_order():
    paired = pair_decisions(fixture_data())
    assert paired['blocks'].tolist() == ['a','b']
    np.testing.assert_allclose(paired['gains'], [[3.,1.7],[0.,-1.3]])
    assert choose_from_gains(paired['gains'], paired['blocks']) == {'a':'observation_refresh','b':'base_continue'}


@pytest.mark.parametrize('fault', ['missing','duplicate','features','role','source'])
def test_invalid_counterfactual_pairs_fail(fault):
    data = fixture_data()
    if fault == 'missing':
        data = {k:v[:-1] for k,v in data.items()}
    elif fault == 'duplicate':
        data['row_ids'][0] = 'a:base_continue'; data['options'][0] = 'base_continue'
    elif fault == 'features':
        data['features'][0,0] = 99
    elif fault == 'role':
        data['roles'] = data['roles'].astype('<U20'); data['roles'][0]='development'
    else:
        data['sources'][0]='s2'
    with pytest.raises(ValueError): pair_decisions(data)


def test_standardization_rejects_development_and_preserves_constant_features():
    mean, scale = train_standardization(np.array([[1.,2.],[3.,2.]]), ['train','train'])
    np.testing.assert_allclose(mean, [2.,2.])
    assert scale[1]>0
    with pytest.raises(ValueError, match='train-only'):
        train_standardization(np.array([[1.,2.],[1000.,2.]]), ['train','development'])


def test_safe_stop_cannot_count_as_refresh_rescue():
    paired = pair_decisions(fixture_data())
    mask = np.ones(2, dtype=bool)
    metrics = recovery_metrics(paired, mask, {'a':'safe_stop','b':'base_continue'})
    assert metrics['refresh_rescue_opportunities'] == 1
    assert metrics['refresh_rescues_recovered'] == 0 and metrics['stop_count'] == 1
    metrics = recovery_metrics(paired, mask, {'a':'observation_refresh','b':'safe_stop'})
    assert metrics['refresh_rescues_recovered'] == 1 and metrics['base_successes_lost'] == 1


def test_refresh_only_and_base_first_ties():
    gains = np.array([[.5,1.],[0.,0.]])
    assert choose_from_gains(gains,['a','b']) == {'a':'safe_stop','b':'base_continue'}
    assert choose_from_gains(gains,['a','b'],allow_stop=False) == {'a':'observation_refresh','b':'base_continue'}


def test_tiny_selection_never_uses_development_sources():
    paired=pair_decisions(fixture_data())
    paired['gains'][1,0]=-.1
    paired['sources'][:]='s1'
    assert tiny_train_sources(paired)==['s1']
    paired['roles']=np.array(['development','development'])
    assert tiny_train_sources(paired)==[]


def test_gain_ranking_handles_ties_and_empty_support():
    assert gain_diagnostics(np.array([1.,-1.]),np.array([0.,0.]))['auc']==.5
    assert gain_diagnostics(np.array([-1.,-1.]),np.array([0.,0.]))['auc'] is None


def test_gain_model_can_learn_opposite_effects_and_saved_scaler(tmp_path):
    torch=pytest.importorskip('torch');torch.set_num_threads(1);torch.manual_seed(7)
    x=np.array([[1.,-1.],[-1.,1.]],dtype=np.float32)
    mean,scale=train_standardization(x,['train','train'])
    model=PairedGainModel(2,normalization='train_standardized',mean=mean,scale=scale,hidden_dim=16)
    optimizer=torch.optim.Adam(model.parameters(),lr=.02)
    y=torch.tensor([[1.,-.5],[-1.,.5]])
    for _ in range(100):
        optimizer.zero_grad();loss=model.loss(model(torch.from_numpy(x)),y,torch.ones(2));loss.backward();optimizer.step()
    np.testing.assert_allclose(model(torch.from_numpy(x)).detach().numpy(),y.numpy(),atol=.03)
    state=model.state_dict()
    assert 'feature_mean' in state and 'feature_scale' in state
    with pytest.raises(ValueError):model.loss(y,y,torch.ones(2),positive_weight=float('nan'))
