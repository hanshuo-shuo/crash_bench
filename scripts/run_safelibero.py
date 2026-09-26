#!/usr/bin/env python3
"""Small auditable adapters around the pinned official evaluation implementation."""
import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/'third_party/vlsa-aegis'
CFG=json.loads((ROOT/'configs/reproduction.json').read_text())
# Set from the public file at the fixed upstream revision; checked before patching.
EXPECTED_SOURCE_SHA256='8491b9381dd0d6dbd1ba9bd0360db7c93a932d7e5913e517b4f2291abd64042c'

class PerceptionPrepared(Exception):pass

def replace_once(source, old, new):
    if source.count(old)!=1:raise RuntimeError('Upstream adapter anchor changed: '+old[:70])
    return source.replace(old,new,1)

def adapt_source(source, mode):
    if hashlib.sha256(source.encode()).hexdigest()!=EXPECTED_SOURCE_SHA256:
        raise RuntimeError('Unexpected upstream evaluation source; refusing to patch')
    if mode in ('nominal','prepare'):
        start=source.index('    from groundingdino.util.inference import load_model')
        end=source.index('    # Start evaluation\n',start)
        source=source[:start]+'    model_groundingdino = None\n'+source[end:]
    if mode=='prepare':
        source=replace_once(source,'    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)','    client = None')
    if mode=='nominal':
        start=source.index('            # Detect obstacles\n')
        end=source.index('            # print("Joint names (qpos):"',start)
        source=source[:start]+'            flag_safety_control = False\n            t = 0\n'+source[end:]
    # Upstream catches runtime failures as episode termination. Do not count these as outcomes.
    source=source.replace('logging.error(f"Caught exception: {e}")','raise RuntimeError("Upstream runtime failure; no scientific label") from e')
    source=replace_once(source,'            # Log current results\n',
        '            _record_episode(task_id, episode_idx, bool(done), bool(collide_flag), int(t + 1 if done else t), int(collide_time), bool(flag_safety_control), str(video_path))\n            # Log current results\n')
    return source

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['nominal','prepare','aegis'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=8000)
    p.add_argument('--suite',default='safelibero_spatial')
    p.add_argument('--level',choices=['I','II'],default='I')
    p.add_argument('--task',type=int,default=0)
    p.add_argument('--episode',type=int,default=0)
    args=p.parse_args()
    if args.suite not in CFG['horizons'] or not 0<=args.task<4 or not 0<=args.episode<50:
        p.error('Invalid official benchmark cell')
    args.output.mkdir(parents=True,exist_ok=False)
    assets=Path(os.environ.get('CB_ASSETS','/projects/p33100/siosio/crashbench_safelibero'))
    config_dir=args.output/'libero_config';config_dir.mkdir()
    benchmark_root=UPSTREAM/'safelibero/libero/libero'
    config={'benchmark_root':str(benchmark_root),'bddl_files':str(benchmark_root/'bddl_files'),'init_states':str(benchmark_root/'init_files'),'assets':str(benchmark_root/'assets'),'datasets':str(UPSTREAM/'safelibero/libero/datasets')}
    (config_dir/'config.yaml').write_text(json.dumps(config,indent=2))
    os.environ['LIBERO_CONFIG_PATH']=str(config_dir)
    sys.path[:0]=[str(ROOT/'scripts'),str(UPSTREAM/'main'),str(UPSTREAM/'safelibero'),str(UPSTREAM/'openpi/packages/openpi-client/src')]
    os.chdir(args.output)
    if args.mode=='aegis':(args.output/'GroundingDINO').symlink_to(assets/'GroundingDINO',target_is_directory=True)
    from openrouter_perception import export_request,read_response
    import utils
    def perception(image,instruction,suite):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        buf=io.BytesIO();plt.imsave(buf,image,format='png')
        directory=export_request(buf.getvalue(),instruction,suite,assets/'perception_cache')
        (args.output/'perception_request.json').write_text(json.dumps({'directory':str(directory),'request_sha256':directory.name},indent=2)+'\n')
        if args.mode=='prepare':raise PerceptionPrepared(str(directory))
        if not (directory/'response.json').exists():raise RuntimeError('Perception cache missing; fill exported request on the networked host: '+str(directory))
        return read_response(directory)
    utils.obstacle_detection=perception
    source=(UPSTREAM/'main/main_aegis.py').read_text()
    patched=adapt_source(source,args.mode)
    (args.output/'adapted_main_aegis.py').write_text(patched)
    records=[]
    def record(task,episode,success,collision,steps,collision_time,safety_enabled,video):
        row={'task':task,'episode':episode,'success':success,'collision':collision,'safe_success':success and not collision,'steps':steps,'collision_time_upstream_zero_based':collision_time if collision else None,'safety_enabled':safety_enabled,'video':video}
        records.append(row)
        (args.output/'episodes.json').write_text(json.dumps(records,indent=2)+'\n')
    actual_commit=(UPSTREAM/'.git/HEAD').read_text().strip()
    if actual_commit.startswith('ref:'):raise RuntimeError('Upstream must be at the pinned detached commit')
    if actual_commit!=CFG['upstream_commit']:raise RuntimeError('Upstream commit mismatch')
    metadata={'mode':args.mode,'suite':args.suite,'level':args.level,'task':args.task,'episode':args.episode,'seed':CFG['seed'],'upstream_commit':actual_commit,'source_sha256':EXPECTED_SOURCE_SHA256,'code_commit':os.environ.get('CB_CODE_COMMIT'),'slurm_job':os.environ.get('SLURM_JOB_ID'),'config':CFG,'started_unix':time.time(),'python':sys.version,'status':'started'}
    (args.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    namespace={'__name__':'safelibero_upstream_adapter','__file__':str(UPSTREAM/'main/main_aegis.py'),'_record_episode':record}
    try:
        exec(compile(patched,str(UPSTREAM/'main/main_aegis.py'),'exec'),namespace)
        settings=namespace['Args'](host='127.0.0.1',port=args.port,task_suite_name=args.suite,safety_level=args.level,task_index=[args.task],episode_index=[args.episode],video_out_path=str(args.output/'videos'),seed=CFG['seed'],replan_steps=CFG['replan_steps'],num_steps_wait=CFG['settle_steps'])
        namespace['eval_libero'](settings)
        if len(records)!=1:raise RuntimeError('Incomplete evaluation; expected exactly one episode')
        metadata['status']='complete'
    except PerceptionPrepared as e:
        if args.mode!='prepare':raise
        metadata['status']='perception_prepared';metadata['request_directory']=str(e)
    except BaseException as e:
        metadata['status']='failed';metadata['error']=str(e);raise
    finally:
        metadata['finished_unix']=time.time()
        (args.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
