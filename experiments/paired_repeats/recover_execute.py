"""Execute only the frozen missing suffix with the original scientific runner."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

CONTROL=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(CONTROL/'scripts'))
from api_budget import atomic_json
from recovery import load_inherited, scheduled, first_chunk_audit

def main(root,port):
    plan=json.loads((root/'plan.json').read_text())
    science=Path(plan['science_source'])
    sys.path.insert(0,str(science/'experiments/paired_repeats'))
    from runtime import Runner
    from execute import review_gate
    from protocol import write_csv,summarize,SCENARIOS
    if Path(sys.modules['runtime'].__file__).resolve()!=science/'experiments/paired_repeats/runtime.py':
        raise RuntimeError('Recovery must use the original scientific runner')
    threshold=review_gate(Path(plan['checks_root']))
    if (root/'STOP.json').exists():raise RuntimeError('Recovery is stopped')
    rows=load_inherited(plan)
    if plan['pending_runs']!=scheduled()[len(rows):]:
        raise RuntimeError('Pending suffix differs from the original schedule')
    directory=root/'full';directory.mkdir()
    runner=Runner(directory,port)
    new_rows=[]
    write_csv(directory/'raw.csv',rows)
    for item in plan['pending_runs']:
        if (root/'STOP.json').exists():raise RuntimeError('Recovery stopped')
        scenario=(item['suite'],item['level'],item['task'])
        row=runner.run(scenario,item['episode'],item['repeat'],item['method'],'full',threshold=threshold)
        reference=plan['expected_initial_states'][row['scenario']+'|'+str(row['episode'])]
        if any(row[name]!=reference[name] for name in ['qpos_sha256','active_obstacle']):
            raise RuntimeError('Recovered run does not match the checked initial state')
        atomic_json(directory/'runs'/row['run_id']/'recovery_provenance.json',{
            'science_commit':plan['code_commit'],'execution_commit':plan['execution_commit'],
            'parent_root':plan['recovery_parent'],'seed':row['seed'],'slurm_job':row['slurm_job']})
        rows.append(row);new_rows.append(row)
        write_csv(directory/'raw.csv',rows);write_csv(directory/'new_runs.csv',new_rows)
        atomic_json(root/'progress.json',{'completed_runs':len(rows),'expected_runs':600,
            'inherited_runs':len(plan['inherited_records']),'new_completed_runs':len(new_rows),
            'new_expected_runs':len(plan['pending_runs']),'updated_unix':time.time()})
    # Recheck inherited hashes at aggregation, so old mutation cannot slip through.
    if load_inherited(plan)!=rows[:len(plan['inherited_records'])]:
        raise RuntimeError('Inherited results changed during recovery')
    quality=first_chunk_audit(rows)
    summarize(rows,root/'summary')
    atomic_json(root/'summary/pairing_quality.json',quality)
    atomic_json(root/'COMPLETE.json',{'runs':600,'inherited_runs':len(plan['inherited_records']),
        'new_runs':len(new_rows),'quality_warning':bool(quality['first_chunk_mismatch_count']),
        'first_chunk_mismatch_count':quality['first_chunk_mismatch_count'],
        'summary':str(root/'summary'),'finished_unix':time.time()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['full']);p.add_argument('root',type=Path);p.add_argument('--port',type=int,required=True)
    a=p.parse_args();root=a.root.resolve()
    try:main(root,a.port)
    except BaseException as error:
        atomic_json(root/'STOP.json',{'stage':'recovery_compute','reason':type(error).__name__+': '+str(error),'unix':time.time()})
        raise
