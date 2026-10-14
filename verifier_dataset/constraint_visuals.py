"""Simulator-native, explicitly nonphysical region annotations.

No raster edits or label-dependent captures. Both constraint queries reference
the very same rendered pixels; the second assignment swaps only region colors.
"""
import itertools
import xml.etree.ElementTree as ET
from .fixture_visuals import edges

COLORS={'orange':[1.,.45,.02,1.],'purple':[.82,.04,.92,1.]}


def add(xml):
    tree=ET.fromstring(xml);world=tree.find('worldbody')
    for role,color in [('a','1 .45 .02 1'),('b','.82 .04 .92 1'),('target','1 1 1 1'),('goal','0 .9 .2 1')]:
        edges(world,'fdc_'+role,[99.,99.,99.],[100.,100.,100.],color,width=.0017)
    return ET.tostring(tree,encoding='unicode')


def edge_boxes(bounds,width=.0017):
    lo,hi=bounds;result=[]
    for axis in range(3):
        others=[i for i in range(3) if i!=axis]
        for side in itertools.product((0,1),repeat=2):
            center=[(a+b)/2 for a,b in zip(lo,hi)];half=[width]*3
            half[axis]=(hi[axis]-lo[axis])/2
            for i,value in zip(others,side):center[i]=bounds[value][i]
            result.append((center,half))
    return result


def capture(env,folder,layout,physical_hash):
    import numpy as np
    from PIL import Image
    from openpi_client import image_tools
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
    import robosuite.macros as macros
    from .inventory import sha,write
    m,d=env.sim.model,env.sim.data;before=physical_hash(env)
    ids={role:[g for g in range(m.ngeom) if (m.geom_id2name(g) or '').startswith('fdc_'+role+'_')]
         for role in ('a','b','target','goal')}
    if any(len(v)!=12 for v in ids.values()):raise RuntimeError('Incomplete virtual-region annotations')
    for role,geoms in ids.items():
        bounds=layout[role]
        for g,(center,half) in zip(geoms,edge_boxes(bounds)):
            if m.geom_contype[g] or m.geom_conaffinity[g] or int(m.geom_bodyid[g])!=0:
                raise RuntimeError('Virtual annotation affects the physical scene')
            m.geom_pos[g]=center;m.geom_size[g]=half
            # World-fixed, noncolliding display geoms only. Do not forward the
            # executing simulation: that would refresh native contact caches.
            d.geom_xpos[g]=center;d.geom_xmat[g]=np.eye(3).reshape(-1)
    output=[]
    for assignment in (0,1):
        for role,color in zip(('a','b'),('orange','purple') if assignment==0 else ('purple','orange')):
            m.geom_rgba[ids[role]]=COLORS[color]
        path=folder/('assignment_%d'%assignment);path.mkdir();images=[]
        for camera,name in [('agentview','external'),('robot0_eye_in_hand','wrist')]:
            rgb=np.asarray(env.sim.render(width=1024,height=1024,camera_name=camera))
            rgb=rgb[::IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]][::-1,::-1]
            rgb=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(rgb),224,224))
            image=path/(name+'.png');Image.fromarray(rgb).save(image)
            images.append(dict(camera=name,path=str(image.relative_to(folder)),sha256=sha(image),width=224,height=224))
        output.append(images)
        if physical_hash(env)!=before:raise RuntimeError('Virtual-region rendering altered physical arrays')
    write(folder/'IMAGES.json',dict(assignments=output,shared_pixels_per_constraint_pair=True,
        all_annotations_noncolliding_world_geoms=True,physical_arrays_unchanged=True,
        native_action_hashes_still_require_full_replay=True,grounding_review='pending'))
    return output
