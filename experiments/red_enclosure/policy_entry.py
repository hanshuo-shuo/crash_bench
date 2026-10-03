"""Apply the shared safety prompt without changing the inherited CPU gate runner."""
import argparse,json,os
from pathlib import Path
import run
from policy_prompt import NATIVE_TASK,SAFETY_INSTRUCTION,EFFECTIVE_PROMPT,adapt_make_env

def main(root,port):
    original=run.make_env
    run.make_env=adapt_make_env(original,run.active_obstacle,run.write,
                                lambda p:json.loads(p.read_text()))
    run.write(root/'POLICY_PROMPT.json',dict(native_language=NATIVE_TASK,
        effective_policy_prompt=EFFECTIVE_PROMPT,safety_instruction=SAFETY_INSTRUCTION,
        shared_across_arms=True,feasibility_label_disclosed=False,
        source_files={x:run.sha(Path(__file__).parent/x) for x in ['policy_prompt.py','policy_entry.py']}))
    try:
        run.main(root,'policy',port)
        checks=[]
        for name in ['open_pi05','sealed_pi05']:
            folder=root/name;task=json.loads((folder/'task.json').read_text())
            rows=[json.loads(x) for x in (folder/'policy.jsonl').read_text().splitlines()]
            if task['effective_policy_prompt']!=EFFECTIVE_PROMPT or not rows:
                raise RuntimeError('Missing effective policy prompt or model calls')
            if not all(r['prompt']==EFFECTIVE_PROMPT for r in rows):
                raise RuntimeError('Policy request prompt mismatch')
            checks.append(dict(run=name,requests=len(rows),task_sha256=run.sha(folder/'task.json'),
                               policy_log_sha256=run.sha(folder/'policy.jsonl')))
        run.write(root/'POLICY_PROMPT_VERIFIED.json',dict(passed=True,effective_prompt=EFFECTIVE_PROMPT,checks=checks))
    finally:
        run.make_env=original

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--port',type=int,required=True);a=p.parse_args()
    try:main(a.root,a.port)
    except BaseException as e:
        run.write(a.root/'STOP.json',dict(stage='policy_shared_safety_prompt',error=type(e).__name__,reason=str(e),job=os.environ.get('SLURM_JOB_ID')))
        raise
