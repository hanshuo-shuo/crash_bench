"""Guard the supplemental train/test boundary and UNKNOWN coverage on synthetic data."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import nuisance_diagnostics as nd


class NuisanceSeparationTest(unittest.TestCase):
    def test_test_labels_cannot_change_selected_heads_or_scores(self):
        rows=[];geometry=[];visibility=[]
        for gi,layout in enumerate(nd.ga.P['layouts']):
            for family in ('slit','cage'):
                for variant in nd.ga.P['variants']:
                    label='unknown' if variant in ('.034','.060') else 'infeasible' if variant in ('.020','.028') else 'feasible'
                    case=layout['id']+'_'+family+'_'+variant
                    outcome='not_run' if label=='unknown' else 'invalid' if case=='L00_cage_.020' else 'safe_timeout'
                    rows.append(dict(case=case,scene_group=layout['id'],split=layout['split'],family=family,variant=variant,
                        label=label,initial_label=label,initial_valid=True,outcome=outcome,
                        execution_horizon_T=300,steps=0 if outcome=='not_run' else 158 if outcome=='invalid' else 300,
                        safe_history=True,safe_success=False))
                    geo=np.zeros(14);geo[6]=.020 if variant in ('open','irrelevant') else float(variant);geometry.append(geo)
                    vis=np.zeros(6);vis[0]=(.08 if label=='infeasible' else .4 if label=='feasible' else .9)+gi*.001
                    vis[3]=.002*gi;visibility.append(vis)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);analysis=root/'synthetic_input';analysis.mkdir()
            np.savez(analysis/'FEATURES.npz',geometry=geometry,visibility=visibility)
            (analysis/'MANIFEST.json').write_text(json.dumps(dict(rows=rows)))
            with patch.object(nd.ro,'cluster_interval',return_value=dict(interval=None,groups=4)):
                nd.analyze(analysis,root/'first')
                for row in rows:
                    if row['split']=='test' and row['label']!='unknown':
                        row['label']='feasible' if row['label']=='infeasible' else 'infeasible'
                        row['initial_label']=row['label']
                (analysis/'MANIFEST.json').write_text(json.dumps(dict(rows=rows)))
                nd.analyze(analysis,root/'flipped_test_labels')
            first=json.loads((root/'first/NUISANCE_DIAGNOSTICS.json').read_text())
            second=json.loads((root/'flipped_test_labels/NUISANCE_DIAGNOSTICS.json').read_text())
            self.assertFalse(first['unknown_accuracy_computed']);self.assertEqual(first['all_unknown'],48)
            for population,expected_known in [('all_known_variants',20),('numeric_gap_variants',12)]:
                record=first['populations'][population]
                self.assertEqual(record['coverage']['test']['known'],expected_known)
                self.assertEqual(record['coverage']['test']['unknown'],8)
                for name,value in record['models'].items():
                    self.assertEqual(value['selection'],second['populations'][population]['models'][name]['selection'])
                    self.assertEqual(value['failed_subset']['n'],expected_known-1)
            for filename in ('known_predictions.csv','unknown_predictions.csv'):
                with (root/'first'/filename).open() as stream:a=list(csv.DictReader(stream))
                with (root/'flipped_test_labels'/filename).open() as stream:b=list(csv.DictReader(stream))
                np.testing.assert_allclose([float(r['score']) for r in a],[float(r['score']) for r in b],rtol=0,atol=1e-12)
                if filename.startswith('unknown'):
                    self.assertEqual(len(a),288)
                    self.assertTrue(all(r['label']=='unknown' and r['binary_accuracy']=='' for r in a))


if __name__=='__main__':unittest.main()
