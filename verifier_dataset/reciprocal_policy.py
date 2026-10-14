"""Validate both object roles in the same source before admitting a contrast unit."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .inventory import read,sha,write
from .paired_policy import prompt_for
from .paired_scene import scan_saved


def check_reciprocal(a,b):
    return bool(a['matched_pair'] and b['matched_pair'] and
        a['primary_goal']==b['enclosed_goal'] and a['enclosed_goal']==b['primary_goal'] and
        a['fixed_destination']==b['fixed_destination'] and a['source']['source_id']==b['source']['source_id'] and
        a['checkpoint']==b['checkpoint'] and a['remaining_horizon']==b['remaining_horizon'] and
        a['initial_physics_sha256']==b['initial_physics_sha256'] and
        all(a['safe_history'].values()) and all(b['safe_history'].values()))


def invoke(case,mode,option=None,port=None):
    command=[sys.executable,'-u','-m','verifier_dataset.paired_scene',str(case),'--mode',mode,'--render']
    if option is not None:command.extend(['--option',str(option)])
    if port is not None:command.extend(['--port',str(port)])
    with (case/(mode+'.log')).open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)


def main(root,port):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Compute allocation required')
    outcomes=[];candidates=[]
    for index,entry in enumerate(read(root/'PAIRED_SOURCES.json')):
        original=Path(entry['case']);original_option=original/('option_%02d'%entry['option'])
        for name,wanted in read(original_option/'HASHES.json').items():
            if sha(original_option/name)!=wanted:raise RuntimeError('Original checked pair changed')
        source_summary=read(original_option/'SUMMARY.json')
        if not source_summary['matched_pair']:raise RuntimeError('A checked forward pair is required')
        unit=root/('unit_%03d'%index);unit.mkdir();forward=unit/'forward';reverse=unit/'reverse'
        forward.mkdir();reverse.mkdir()
        for case in (forward,reverse):
            for name in ('INPUT.json','CANONICAL_CONTEXT.json','CANONICAL_PHYSICS.npz'):
                shutil.copyfile(original/name,case/name)
        shutil.copyfile(original/'PROPOSALS.json',forward/'PROPOSALS.json')
        write(reverse/'TASK_OVERRIDE.json',dict(goal=source_summary['enclosed_goal'],
            required_enclosed=source_summary['primary_goal'][1],
            meaning='Swap the physical objects between free target and enclosed target; preserve original state and common destination'))
        try:
            prompt=prompt_for(source_summary['enclosed_goal'])
        except RuntimeError as error:
            outcomes.append(dict(unit=str(unit),source_id=source_summary['source']['source_id'],reciprocal=False,
                reason=str(error),label='unknown',policy_calls=0));continue
        write(unit/'ROLE_PLAN.json',dict(source_case=str(original_option),source_summary_sha256=sha(original_option/'SUMMARY.json'),
            forward_goal=source_summary['primary_goal'],reverse_goal=source_summary['enclosed_goal'],
            reverse_policy='existing native pi0.5 nominal; no new VLM or learned controller',reverse_prompt=prompt,
            max_reverse_rollouts=1,source_seed=source_summary['source']['seed'],api_calls=0))
        invoke(forward,'execute',entry['option'])
        f=forward/('option_%02d'%entry['option']);scan_saved(f);a=read(f/'SUMMARY.json')
        if not a['matched_pair']:
            outcomes.append(dict(unit=str(unit),source_id=a['source']['source_id'],reciprocal=False,
                reason='Forward rendered scene has no matched pair',forward=str(f),label='unknown'));continue
        invoke(reverse,'prepare')
        proposals=read(reverse/'PROPOSALS.json')
        if len(proposals['rows'])!=1:
            outcomes.append(dict(unit=str(unit),source_id=a['source']['source_id'],reciprocal=False,
                reason='Reverse enclosure/goal certificate proposal unavailable',forward=str(f),label='unknown'));continue
        invoke(reverse,'execute',0,port=port)
        r=reverse/'option_00';scan_saved(r);b=read(r/'SUMMARY.json');matched=check_reciprocal(a,b)
        outcome=dict(unit=str(unit),source_id=a['source']['source_id'],reciprocal=matched,forward=str(f),reverse=str(r),
            forward_actions=a['actions'],reverse_actions=b['actions'],reverse_safe_success=b['positive_witness'],
            reason=None if matched else 'Both directions with preserved object roles and destination have not passed',
            candidate_rows=8 if matched else 0,api_calls=0)
        outcomes.append(outcome)
        if matched:
            for role,folder in [('forward',f),('reverse',r)]:
                for row in read(folder/'PAIRED_CANDIDATES.json'):
                    row['sample_id']='fdreciprocal_%05d'%len(candidates)
                    row['metadata'].update(reciprocal_unit=str(unit),direction=role,evidence_root=str(folder))
                    for im in row['student']['images']:im['path']=str(folder/im['path'])
                    candidates.append(row)
        write(root/'PROGRESS.json',dict(completed=len(outcomes),expected=len(read(root/'PAIRED_SOURCES.json')),
            reciprocal_units=sum(x['reciprocal'] for x in outcomes),candidate_rows=len(candidates)))
    write(root/'RECIPROCAL_CANDIDATES.json',candidates)
    write(root/'RECIPROCAL_COMPLETE.json',dict(passed=True,rows=outcomes,reciprocal_units=sum(x['reciprocal'] for x in outcomes),
        candidate_rows=len(candidates),final_reasoning_targets=0,training_allowed=False,api_calls=0,
        slurm_job=os.environ['SLURM_JOB_ID'],unmatched_cases_excluded=True))
    write(root/'HASHES.json',{name:sha(root/name) for name in ('RECIPROCAL_CANDIDATES.json','RECIPROCAL_COMPLETE.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--port',type=int,required=True)
    a=p.parse_args();main(a.root,a.port)
