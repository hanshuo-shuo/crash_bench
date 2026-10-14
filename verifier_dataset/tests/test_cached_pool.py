import unittest
from verifier_dataset.cached_pool import GROUPS,select_pool,run_spec
from verifier_dataset.inventory import identity


class CachedSourcePool(unittest.TestCase):
    def fixture(self):
        rows=[]
        for suite,task in GROUPS:
            for episode in range(6):
                scene=dict(suite=suite,level='I',task=task,episode=episode)
                rows.append(dict(scene=scene,source_id=identity(scene),method='aegis',
                                 proxy_safe_success=True,feasibility_label='unknown'))
        return rows

    def test_new_sources_and_shards_are_distinct_and_remain_unknown(self):
        rows=select_pool(self.fixture())
        self.assertEqual(len(rows),41)
        self.assertEqual(len(set(r['source_id'] for r in rows)),41)
        self.assertTrue(all(r['feasibility_label']=='unknown' for r in rows))
        self.assertFalse(any(r['scene']['suite']=='safelibero_goal' and r['scene']['episode']==0 for r in rows))
        for shard in range(2):
            selected=rows[shard::2]
            self.assertEqual(len({r['scene']['suite'] for r in selected}),3)
            self.assertTrue(all(run_spec('collect-cached-batch',i,r)['method']=='aegis' for i,r in enumerate(selected)))

    def test_incomplete_pool_cannot_silently_shrink_coverage(self):
        with self.assertRaisesRegex(RuntimeError,'Insufficient'):
            select_pool(self.fixture()[:5])


if __name__=='__main__':unittest.main()
