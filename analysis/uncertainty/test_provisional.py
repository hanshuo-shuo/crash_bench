"""Deterministic descriptive-context checks; no model/calibrator fitting."""
import unittest
from provisional import stratify


class ContextTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        for split in ['train','test']:
            for task,labels,x,p in [('A',[0,0,0,1],0.,.25),('B',[0,1,1,1],1.,.75)]:
                for i,y in enumerate(labels):
                    rows.append(dict(stage='first_infer',proxy='disagreement',id=split+task+str(i),scene=task+'/'+split,arm='nominal',split=split,x=x,y=y,probability=p))
        return rows,{'completion/first_infer/disagreement':dict(prevalence=.5)}

    def test_task_mixture_auc_is_not_within_task_discrimination(self):
        rows,models=self.fixture();result=stratify(rows,models)
        chosen=[r for r in result if r['stage']=='first_infer' and r['proxy']=='disagreement' and r['arm']=='pooled']
        pooled=next(r for r in chosen if r['task']=='pooled')
        self.assertAlmostEqual(pooled['AUROC'],.75)
        self.assertEqual(pooled['positive'],4);self.assertEqual(pooled['negative'],4)
        for r in chosen:
            if r['task']!='pooled':self.assertAlmostEqual(r['AUROC'],.5)
        self.assertAlmostEqual(pooled['Brier'],.1875);self.assertAlmostEqual(pooled['global_train_prior_Brier'],.25)
        self.assertAlmostEqual(pooled['task_train_prior_Brier'],.1875)
        prevalence=pooled['weighted_positive_prevalence']
        self.assertAlmostEqual(pooled['Brier'],prevalence*pooled['Brier_positive']+(1-prevalence)*pooled['Brier_negative'])

    def test_single_class_auc_missing_and_test_labels_do_not_change_priors(self):
        rows,models=self.fixture()
        for r in rows:
            if r['split']=='test':r['y']=0
        pooled=next(r for r in stratify(rows,models) if r['stage']=='first_infer' and r['proxy']=='disagreement' and r['task']=='pooled' and r['arm']=='pooled')
        self.assertIsNone(pooled['AUROC']);self.assertIsNone(pooled['BA']);self.assertIsNone(pooled['Brier_positive'])
        self.assertAlmostEqual(pooled['task_train_prior_Brier'],.3125)


if __name__=='__main__':unittest.main()
