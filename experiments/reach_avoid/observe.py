"""Initial-only native visible inputs and read-only layer-hook validation."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image
import baseline as b
r,d=b.r,b.d
from visibility import audit_visibility

P=json.loads((Path(__file__).parent/'observe_protocol.json').read_text())
NUMERIC=('layers','image_layers','trace_final','native_final','projected_vision',
    'own_vision_tower','tokenized_prompt','tokenized_prompt_mask')


def blocked(*args,**kwargs):raise RuntimeError('Initial-only observation stage prohibits stepping')


def restore_case(root,case,allow_actions=False):
    from libero.libero.envs.env_wrapper import ControlEnv
    source=Path(P['label_root'])/case
    folder=root/case;folder.mkdir()
    fixture=json.loads((source/'fixture.json').read_text())
    d.setup_layout(0);d.CURRENT_BOXES=fixture['boxes']
    # Native constructor/reset and forward-only restoration, no env commands.
    env=ControlEnv(bddl_file_name=source/'task.bddl',camera_heights=1024,camera_widths=1024,
        use_camera_obs=True,has_offscreen_renderer=True,camera_names=['agentview','robot0_eye_in_hand'],camera_depths=False)
    if not allow_actions:env.step=blocked;env.sim.step=blocked
    env.seed(7);env.reset()
    if not allow_actions:env.sim.step=blocked
    env.set_init_state(np.load(source/'official_init.npy'))
    snapshot=json.loads((source/'initial_restore.json').read_text())
    controller=env.robots[0].controller
    for k,v in snapshot['controller'].items():
        if not hasattr(controller,k):setattr(controller,k,np.asarray(v) if isinstance(v,list) else v)
    r.write(folder/'pre_restore.json',r.capture(env))
    try:obs=r.restore(env,snapshot)
    finally:r.write(folder/'post_restore.json',r.capture(env))
    # Reject altered collision geometry, not harmless render byte differences.
    geo=json.loads((source/'geometry.json').read_text());model=env.sim.model;data=env.sim.data
    for box in geo['boxes']:
        ident=model.geom_name2id(box['name']);lo=np.asarray(box['lower']);hi=np.asarray(box['upper'])
        if not (np.max(np.abs(model.geom_size[ident]-(hi-lo)/2))<1e-9 and
            np.max(np.abs(data.geom_xpos[ident]-(hi+lo)/2))<1e-9 and
            int(model.geom_bodyid[ident])==0 and model.geom_contype[ident] and model.geom_conaffinity[ident]):
            raise RuntimeError('Restored fixture geometry mismatch')
    audit=SimpleNamespace(red={model.geom_name2id(x['name']) for x in geo['boxes']},boxes=geo['boxes'],
        fixed_goal=geo['fixed_goal'],target_id=env.env.obj_body_id[r.STATE['target']],forward=lambda:env.sim.data)
    r.write(folder/'INHERITED.json',dict(source=str(source),
        hashes={name:r.sha(source/name) for name in ('summary.json','CERTIFICATE.json','VERIFIED.json','geometry.json','initial_restore.json')},
        label=json.loads((source/'summary.json').read_text())['label'],environment_actions=0))
    return folder,env,obs,audit


def save_extraction(client,folder,data,name):
    result=client.infer(dict(data,__ra_layers__=True))
    # Persist every scientific comparison array before acceptance checks.
    np.savez_compressed(folder/(name+'.npz'),**{key:np.asarray(result[key]) for key in NUMERIC})
    r.write(folder/(name+'.json'),dict(result['metadata'],transport_metadata={
        k:v for k,v in result.items() if k not in NUMERIC and k!='metadata'}))
    return {k:np.asarray(result[k]) for k in NUMERIC},result['metadata']


def main(root,port):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from openpi_client import image_tools
    r.configure(root);r.write(root/'protocol.json',P)
    client=WebsocketClientPolicy('127.0.0.1',port);rows=[];anchor=None
    for case in P['cases']:
        folder,env,obs,audit=restore_case(root,case)
        try:
            before=r.capture(env);state_hash=r.statehash(before)
            data=r.policy_input(obs,b.P['prompt_safe'])
            np.savez_compressed(folder/'actual_policy_input.npz',**data)
            # Save actual policy RGB and direct RGB before any agreement check.
            align=[]
            for camera,key in [('agentview','observation/image'),('robot0_eye_in_hand','observation/wrist_image')]:
                direct=np.asarray(env.sim.render(width=1024,height=1024,camera_name=camera))[::-1,::-1]
                direct=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(direct),224,224))
                Image.fromarray(data[key]).save(folder/(camera+'_actual.png'))
                Image.fromarray(direct).save(folder/(camera+'_direct.png'))
                delta=np.abs(data[key].astype(int)-direct.astype(int))
                Image.fromarray(delta.max(2).astype(np.uint8)).save(folder/(camera+'_difference.png'))
                align.append(dict(camera=camera,exact=bool(np.array_equal(data[key],direct)),
                    changed_pixels=int(np.any(delta,axis=2).sum()),linf=int(delta.max())))
            audit_visibility(env,obs,audit,folder,data)
            # Save segmentation camera transforms and model geometry so spatial
            # alignment is auditable even if renderer byte agreement is imperfect.
            np.savez_compressed(folder/'segmentation_provenance.npz',cam_xpos=env.sim.data.cam_xpos.copy(),
                cam_xmat=env.sim.data.cam_xmat.copy(),geom_xpos=env.sim.data.geom_xpos.copy(),
                geom_xmat=env.sim.data.geom_xmat.copy(),geom_size=env.sim.model.geom_size.copy())
            client.infer({'__paired_reset_rng__':7,'run_id':case})
            arrays,meta=save_extraction(client,folder,data,'layer_features')
            after=r.capture(env);r.write(folder/'after_observation.json',after)
            r.write(folder/'ALIGNMENT.json',dict(views=align,physics_unchanged=state_hash==r.statehash(after),
                policy_input_is_actual_observation=True,
                visibility_accepted=all(x['exact'] for x in align),
                mismatch_policy='Retain extraction and both RGB arrays; mark segmentation visibility unresolved on any mismatch, do not relabel state or stop all cases.'))
            if state_hash!=r.statehash(after):raise RuntimeError('Observation changed initial physics')
            row=dict(case=case,label=json.loads((folder/'INHERITED.json').read_text())['label'],
                trace_native_relative_l2=meta['trace_native_relative_l2'],trace_native_linf=meta['trace_native_linf'],
                feature_agreement=meta['trace_native_relative_l2']<=P['trace_relative_l2_tolerance'],
                visibility_alignment=all(x['exact'] for x in align),environment_actions=0)
            rows.append(row);r.write(root/'PROGRESS.json',dict(rows=rows))
            if anchor is None:anchor=(folder,data,arrays)
        finally:env.close()
    folder,data,first=anchor
    repeated,meta=save_extraction(client,folder,data,'end_anchor')
    difference={k:float(np.max(np.abs(first[k].astype(float)-repeated[k].astype(float)))) for k in NUMERIC}
    r.write(root/'ANCHOR.json',dict(linf=difference,same_process=True,arrays_saved=True))
    r.write(root/'COMPLETE.json',dict(rows=rows,code_commit=os.environ['CB_CODE_COMMIT'],job=os.environ['SLURM_JOB_ID'],
        api_calls=0,environment_actions=0,policy_rollouts=0,readout_training=0,
        all_layer_comparisons_accepted=all(x['feature_agreement'] for x in rows),anchor_linf=difference))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--port',type=int,required=True);a=p.parse_args()
    try:main(a.root,a.port)
    except BaseException as error:
        r.write(a.root/'STOP.json',dict(error=type(error).__name__,reason=str(error),job=os.environ.get('SLURM_JOB_ID')));raise
