"""End-to-end synthetic persisted loader and OOF exclusion regression."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import grouped_analysis as ga

class OutcomePipeline(unittest.TestCase):
    def test_loader_and_oof_exclude_invalid_and_reject_partial_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);layouts=[dict(id='G'+str(i),split=('train' if i<3 else 'validation' if i==3 else 'test')) for i in range(6)]
            protocol=dict(ga.P,layouts=layouts,variants=['success','negative','timeout','invalid'],group_permutation=[r['id'] for r in layouts])
            rng=np.random.default_rng(42)
            for layout in layouts:
                for family in ('slit','cage'):
                    for variant in protocol['variants']:
                        folder=root/'input'/'shard_0'/'output'/(layout['id']+'_'+family+'_'+variant);folder.mkdir(parents=True)
                        label='infeasible' if variant=='negative' else 'feasible'
                        outcome={'negative':'collision','success':'safe_completion','timeout':'safe_timeout','invalid':'invalid'}[variant]
                        summary=dict(outcome=outcome,steps=300 if variant=='timeout' else 1,execution_horizon_T=300,
                            initial_valid=True,initial_label=label,safe_history=variant!='negative',safe_success=variant=='success')
                        def put(name,value):(folder/name).write_text(json.dumps(value))
                        put('summary.json',summary);put('INHERITED.json',dict(label=label));put('VERIFIED.json',dict(passed=True))
                        put('FEATURE_AUDIT.json',dict(physics_unchanged=True,layer_accepted=True,attention_accepted=True))
                        v=rng.normal(size=3)+int(label=='infeasible')
                        np.savez(folder/'layer_features.npz',own_vision_tower=v,projected_vision=v,native_final=v,layers=v[None,:],image_layers=v[None,:])
                        np.savez(folder/'general_vision.npz',dino_patch=v)
                        np.savez(folder/'cue_baselines.npz',rgb_pooled_and_red=v)
                        np.savez(folder/'attention_cues.npz',visibility=v,geometry=v,knows_fixed=np.clip(v,0,1))
                        np.savez(folder/'actual_policy_input.npz',**{'observation/state':v})
            with patch.object(ga,'P',protocol),patch.object(ga.ro,'cluster_interval',return_value={}),patch.object(ga,'paired_interval',return_value={}):
                rows,features=ga.load(root/'input')
                # Strict fitting/metric entry points in production reject -1.
                ga.analyze(rows,features,root/'analysis')
                result=json.loads((root/'analysis/INCREMENTAL.json').read_text())
                self.assertEqual(result['undefined_outcome_exclusion_counts'],dict(train=3,validation=1,test=2))
                self.assertEqual(result['training_failure_classes'],[0,1])
                self.assertTrue(all('nested_selection_trials' in fold for fold in result['folds']))
                file=root/'input/shard_0/output/G0_slit_timeout/summary.json'
                broken=json.loads(file.read_text());broken['steps']=1;file.write_text(json.dumps(broken))
                with self.assertRaisesRegex(ValueError,'complete T-step'):ga.load(root/'input')
if __name__=='__main__':unittest.main()
