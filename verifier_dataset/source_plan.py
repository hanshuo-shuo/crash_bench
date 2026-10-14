"""Hash-checked official outcome inventory for additional task families."""
from collections import Counter
import os
from pathlib import Path
import sys
from .inventory import ASSETS, read, sha, write, identity


def main(root):
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('CPU allocation required')
    batch=ASSETS/'batches/20260927T204911Z_d22035816647'
    meta=read(batch/'batch.json');complete=read(batch/'COMPLETE.json');rows=[];files=[]
    for cell in meta['cells']:
        inherited=meta['inherited_cells'].get(str(cell['index']))
        if inherited:
            path=Path(inherited['evaluation_dir'])
            for name,expected in inherited['sha256'].items():
                if sha(path/name)!=expected:raise RuntimeError('Inherited official evidence changed')
        else:
            paths=[p for p in (batch/'runs').glob('%02d_*/evaluation'%cell['index'])
                   if (p/'manifest.json').is_file() and read(p/'manifest.json').get('status')=='complete']
            if len(paths)!=1:raise RuntimeError('Ambiguous completed official cell '+str(cell['index']))
            path=paths[0]
        manifest=read(path/'manifest.json');episodes=read(path/'episodes.json')
        if manifest['status']!='complete' or [r['episode'] for r in episodes]!=list(range(50)):
            raise RuntimeError('Incomplete official source cell')
        files.append(dict(path=str(path),manifest_sha256=sha(path/'manifest.json'),episodes_sha256=sha(path/'episodes.json')))
        for item in episodes:
            scene=dict(suite=cell['suite'],level=cell['level'],task=cell['task'],episode=item['episode'])
            rows.append(dict(scene=scene,source_id=identity(scene),method=cell['mode'],
                proxy_safe_success=item['safe_success'],source_success=item['success'],
                source_collision=item['collision'],source_evaluation=str(path),source_code_commit=manifest['code_commit'],
                feasibility_label='unknown',requires_instrumented_execution=True))
    if len(rows)!=3200 or sum(r['proxy_safe_success'] for r in rows)!=1021:
        raise RuntimeError('Official outcome counts differ from terminal report')
    # Four untouched Goal task compositions complement the three instrumented
    # Spatial/Object compositions. Five different initial states per task.
    selected=[]
    for task in range(4):
        pool=[r for r in rows if r['scene']['suite']=='safelibero_goal' and r['scene']['level']=='I'
              and r['scene']['task']==task and r['method']=='aegis' and r['proxy_safe_success']]
        chosen=sorted(pool,key=lambda r:r['scene']['episode'])[:5]
        if len(chosen)!=5:raise RuntimeError('Insufficient fixed Goal source pool')
        selected.extend(chosen)
    out=root/'source_plan';out.mkdir(exist_ok=False)
    write(out/'OFFICIAL_OUTCOMES.json',rows);write(out/'SOURCE_HASHES.json',files)
    write(out/'GOAL20_CANDIDATES.json',selected)
    write(out/'SUMMARY.json',dict(passed=True,official_episodes=len(rows),proxy_successes=1021,
        distinct_original_identities=len({r['source_id'] for r in rows}),
        selected_original_states=len(selected),selected_task_compositions=4,selected_family='safelibero_goal',
        selected_by_task=dict(Counter(r['scene']['task'] for r in selected)),
        selection_conditioned_on_historical_proxy_success=True,
        selection_scope='Collection efficiency only, not unbiased policy evaluation or contact-safe label evidence',
        inherited_terminal_report_sha256=sha(batch/'COMPLETE.json'),accepted_dataset_rows=0,
        api_calls=0,slurm_job=os.environ['SLURM_JOB_ID']))
    write(out/'HASHES.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(read(out/'SUMMARY.json'),flush=True)


if __name__=='__main__':main(Path(sys.argv[1]))
