"""Execute one600-case shard, checkpoint whole cases and retain failures."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from common import atomic_json, full_schedule


def check_tensors(root, results):
    import numpy as np
    from common import disagreement
    count=0
    for result in results:
        directory=root/'runs'/result['spec']['name']
        proof=json.loads((directory/'CONDITIONAL_ACTION_EQUALITY.json').read_text())
        if not proof['passed']: raise RuntimeError('Conditional action gate failed')
        for line in (directory/'inferences.jsonl').read_text().splitlines():
            info=json.loads(line)
            with np.load(directory/('infer_%03d.npz'%info['infer_index']),allow_pickle=False) as saved:
                samples=saved['diagnostic_actions']; scales=saved['normalization_scales']
                if samples.shape!=(8,10,7) or not np.isfinite(samples).all():
                    raise RuntimeError('Diagnostic tensor missing/nonfinite')
                if saved['sample_seeds'].tolist()!=info['sample_seeds']:
                    raise RuntimeError('Diagnostic seed pairing mismatch')
                reduced=disagreement(samples.tolist(),scales.tolist())
                if abs(reduced-info['disagreement'])>1e-12:
                    raise RuntimeError('Saved diagnostic reduction differs')
                count+=1
    return count


def main(root,port):
    from runtime import Runner
    plan=json.loads((root/'plan.json').read_text()); cfg=plan['configuration']
    runner=Runner(root,port,cfg); results=[]; drain=[False]
    def request_drain(signum, frame):
        drain[0]=True
        atomic_json(root/'DRAIN_REQUEST.json',dict(signal=signum,unix=time.time()))
    signal.signal(signal.SIGUSR1,request_drain)
    deadline=int(os.environ['CB_ALLOCATION_START'])+cfg['resources']['minutes']*60
    schedule=list(full_schedule(cfg,plan['shard']))
    try:
        for spec in schedule:
            if (root/'STOP.json').exists() or (root.parent.parent/'STOP.json').exists():
                raise RuntimeError('Campaign worker stopped')
            if drain[0] or (root/'DRAIN_SIGNAL').exists() or time.time()+cfg['case_checkpoint_seconds']>=deadline:
                atomic_json(root/'PARTIAL.json',dict(completed=len(results),expected=600,
                    reason='No new case inside allocation checkpoint margin',results=results))
                return
            result=runner.run(spec); results.append(result)
            atomic_json(root/'progress.json',dict(completed=len(results),expected=600,results=results))
            if plan['shard']==0 and len(results)==8:
                count=check_tensors(root,results)
                atomic_json(root/'FULL_SMOKE_PASS.json',dict(passed=True,runs=8,
                    inferences=count,states=[0,42],repeats=[0,1],native_same_input_rng_exact=True,
                    diagnostic_tensor_reduction_passed=True))
        atomic_json(root/'COMPUTE_COMPLETE.json',dict(completed=len(results),expected=600,
            results=results,slurm_job=os.environ['SLURM_JOB_ID']))
    except BaseException as error:
        atomic_json(root/'STOP.json',dict(stage='compute',error=type(error).__name__,
            reason=str(error),completed=len(results),slurm_job=os.environ['SLURM_JOB_ID']))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--port',type=int,required=True)
    a=p.parse_args();main(a.root,a.port)
