"""Render saved CPU witness states on the already authorized policy GPU.

No simulation step or new reference rollout. Images are explicitly labeled as
synchronized saved-state renders, distinct from policy-native observations.
"""
import json,sys
from pathlib import Path
import numpy as np
import imageio
from robosuite.utils.binding_utils import MjSim,MjRenderContextOffscreen
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
import robosuite.macros as macros
root=Path(sys.argv[1]);out=root/'saved_state_images';out.mkdir()
records=[]
for name in ['open_reference','sealed_reference']:
    run=root/name;sim=MjSim.from_xml_string((run/'model.xml').read_text())
    context=MjRenderContextOffscreen(sim,device_id=0,max_width=1024,max_height=1024)
    context.vopt.geomgroup[0]=0;context.vopt.geomgroup[1]=1
    for label in ['initial','final']:
        saved=json.loads((run/(label+'_restore.json')).read_text())
        sim.set_state_from_flattened(np.asarray(saved['sim_state']))
        for k,v in saved['arrays'].items():
            dest=np.asarray(getattr(sim.data,k));dest[:]=np.asarray(v).reshape(dest.shape)
        sim.forward()
        rgb=sim.render(width=1024,height=1024,camera_name='agentview')
        rgb=rgb[::IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]][::-1,::-1]
        file=out/(name+'_'+label+'.png');imageio.imwrite(file,np.ascontiguousarray(rgb))
        records.append(dict(file=file.name,source=str(run.resolve()/(label+'_restore.json')),time=float(sim.data.time),
            camera='agentview',resolution=[1024,1024],kind='synchronized saved-state render; no integration',
            source_summary=json.loads((run/'summary.json').read_text())))
    del context
    sim.free()
(out/'FIGURES.json').write_text(json.dumps(records,indent=2)+'\n')
