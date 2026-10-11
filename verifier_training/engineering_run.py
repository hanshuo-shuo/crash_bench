"""Immutable campaign Slurm dispatch."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from .common import write


def main(stage,root,assets):
    if stage=='audit':
        from .prepare import main as prepare
        from .observability import audit
        prepare(root,assets)
        audit(root/'dataset',root/'audit')
    elif stage.startswith('fit_'):
        from .train import train
        variant=stage[-1]
        train(SimpleNamespace(out=root/variant,config=root/('execution_'+variant+'.json'),
            dataset=root/'dataset',assets=assets,variant=variant,observability=root/'audit/AUDIT.json'))
    elif stage.startswith('diagnose_'):
        from .diagnostics import main as diagnose
        diagnose(root,assets,stage[-1])
    elif stage=='report':
        from .engineering_report import main as report
        report(root)
    else:raise ValueError('Unknown stage')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('root',type=Path);p.add_argument('assets',type=Path)
    a=p.parse_args()
    try:main(a.stage,a.root,a.assets)
    except BaseException as e:
        write(a.root/(a.stage+'_STOP.json'),{'error':type(e).__name__,'reason':str(e),'job':os.environ.get('SLURM_JOB_ID')})
        raise
