"""Capture both marker assignments once, then revalidate actual scene execution."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .inventory import read,sha,write


def capture(env,folder,physical_hash):
    import numpy as np
    from PIL import Image
    from openpi_client import image_tools
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
    import robosuite.macros as macros
    from .paired_inputs import COLORS
    m=env.sim.model;before=physical_hash(env);all_images=[]
    marker={role:[g for g in range(m.ngeom) if (m.geom_id2name(g) or '').startswith('fd_visual_'+role+'_')]
            for role in ('primary','enclosed')}
    if any(len(v)!=12 for v in marker.values()):raise RuntimeError('Incomplete visual target outlines')
    visual=[g for g in range(m.ngeom) if (m.geom_id2name(g) or '').startswith('fd_visual_')]
    if any(m.geom_contype[g] or m.geom_conaffinity[g] for g in visual):raise RuntimeError('A visual annotation affects collisions')
    original=np.asarray(m.geom_rgba).copy()
    try:
        for assignment in (0,1):
            colors=('orange','purple') if assignment==0 else ('purple','orange')
            for role,color in zip(('primary','enclosed'),colors):
                m.geom_rgba[marker[role]]=list(map(float,COLORS[color].split()))
            directory=folder/('assignment_%d'%assignment);directory.mkdir();images=[]
            for camera,name in [('agentview','external'),('robot0_eye_in_hand','wrist')]:
                rgb=np.asarray(env.sim.render(width=1024,height=1024,camera_name=camera))
                rgb=rgb[::IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]][::-1,::-1]
                rgb=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(rgb),224,224))
                path=directory/(name+'.png');Image.fromarray(rgb).save(path)
                images.append(dict(camera=name,path=str(path.relative_to(folder)),sha256=sha(path),width=224,height=224))
            all_images.append(images)
            if physical_hash(env)!=before:raise RuntimeError('Marker rendering mutated initial physics')
    finally:m.geom_rgba[:]=original
    write(folder/'IMAGES.json',dict(assignments=all_images,physical_arrays_unchanged=True,
        shared_observation='Each actual camera capture is referenced by both target queries; no separate label-dependent render',
        visual_geoms_have_no_collision=True,input_grounding='pending visual review',training_allowed=False))
    return all_images


def main(root):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Rendering allocation required')
    from .paired_scene import scan_saved
    results=[];candidate_rows=[]
    for index,source in enumerate(read(root/'PAIRED_SOURCES.json')):
        cpu_case=Path(source['case']);cpu_option=cpu_case/('option_%02d'%source['option'])
        for name,wanted in read(cpu_option/'HASHES.json').items():
            if sha(cpu_option/name)!=wanted:raise RuntimeError('CPU pair evidence changed')
        old=read(cpu_option/'SUMMARY.json')
        if not old['matched_pair'] or not old.get('synchronized_native_goal_required') or not old.get('initial_fixture_valid'):
            raise RuntimeError('Fully checked pair required before rendering')
        case=root/('case_%03d'%index);case.mkdir()
        for name in ('INPUT.json','CANONICAL_CONTEXT.json','CANONICAL_PHYSICS.npz','PROPOSALS.json'):
            shutil.copyfile(cpu_case/name,case/name)
        option=source['option']
        with (case/'render.log').open('w') as log:
            subprocess.run([sys.executable,'-u','-m','verifier_dataset.paired_scene',str(case),
                '--mode','execute','--option',str(option),'--render'],stdout=log,stderr=subprocess.STDOUT,check=True)
        folder=case/('option_%02d'%option);scan_saved(folder)
        current=read(folder/'SUMMARY.json')
        if not current['matched_pair']:raise RuntimeError('Actual rendered scene lost a witness/certificate; no paired row admitted')
        rows=read(folder/'PAIRED_CANDIDATES.json')
        for row in rows:
            row['sample_id']='fdpair_%05d'%len(candidate_rows)
            row['metadata']['evidence_root']=str(folder)
            row['metadata']['cpu_pair_root']=str(cpu_option)
            for im in row['student']['images']:im['path']=str(folder/im['path'])
            candidate_rows.append(row)
        results.append(dict(case=str(folder),source_id=current['source']['source_id'],matched_pair=True,
            cpu_actions=old['actions'],rendered_actions=current['actions'],candidate_rows=4))
    write(root/'PAIRED_CANDIDATES.json',candidate_rows)
    write(root/'PAIRED_RENDER_COMPLETE.json',dict(passed=True,rows=results,candidate_rows=len(candidate_rows),
        matched_physical_states=len(results),final_reasoning_targets=0,training_allowed=False,
        api_calls=0,slurm_job=os.environ['SLURM_JOB_ID']))
    write(root/'HASHES.json',{name:sha(root/name) for name in ('PAIRED_CANDIDATES.json','PAIRED_RENDER_COMPLETE.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
