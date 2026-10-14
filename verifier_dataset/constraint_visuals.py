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
    target_root=int(env.env.obj_body_id[layout['native_goal'][1]])
    target_geoms=[]
    for g in range(m.ngeom):
        ancestor=int(m.geom_bodyid[g])
        while ancestor and ancestor!=target_root:ancestor=int(m.body_parentid[ancestor])
        if ancestor==target_root:target_geoms.append(g)
    output=[];visibility=[]
    for assignment in (0,1):
        for role,color in zip(('a','b'),('orange','purple') if assignment==0 else ('purple','orange')):
            m.geom_rgba[ids[role]]=COLORS[color]
        path=folder/('assignment_%d'%assignment);path.mkdir();images=[];seen=[]
        for camera,name in [('agentview','external'),('robot0_eye_in_hand','wrist')]:
            rgb=np.asarray(env.sim.render(width=1024,height=1024,camera_name=camera))
            rgb=rgb[::IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]][::-1,::-1]
            rgb=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(rgb),224,224))
            image=path/(name+'.png');Image.fromarray(rgb).save(image)
            images.append(dict(camera=name,path=str(image.relative_to(folder)),sha256=sha(image),width=224,height=224))
            segmentation=np.asarray(env.sim.render(width=1024,height=1024,camera_name=camera,segmentation=True))
            segmentation=segmentation[::IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]][::-1,::-1]
            if segmentation.shape!=(1024,1024,2):raise RuntimeError('Unexpected native segmentation format')
            # MuJoCo geometry object type is read from the installed enum.
            import mujoco
            geom_pixels=segmentation[:,:,0]==int(mujoco.mjtObj.mjOBJ_GEOM)
            entities={}
            for entity,which in dict(ids,target_object=target_geoms).items():
                mask=geom_pixels&np.isin(segmentation[:,:,1],which)
                small=np.asarray(Image.fromarray(mask.astype('uint8')).resize((224,224),Image.Resampling.NEAREST if hasattr(Image,'Resampling') else Image.NEAREST))
                entities[entity]=dict(pixels_1024=int(mask.sum()),pixels_224_nearest=int(small.sum()))
            seg_file=path/(name+'_segmentation.npz')
            np.savez_compressed(seg_file,object_type=segmentation[:,:,0],object_id=segmentation[:,:,1])
            seen.append(dict(camera=name,entities=entities,segmentation_path=str(seg_file.relative_to(folder)),
                segmentation_sha256=sha(seg_file),alignment='Same camera, resolution, row convention and orientation as RGB; nearest labels are diagnostic only'))
        output.append(images)
        visibility.append(seen)
        if physical_hash(env)!=before:raise RuntimeError('Virtual-region rendering altered physical arrays')
    write(folder/'IMAGES.json',dict(assignments=output,shared_pixels_per_constraint_pair=True,
        all_annotations_noncolliding_world_geoms=True,physical_arrays_unchanged=True,
        native_action_hashes_still_require_full_replay=True,grounding_review='pending'))
    write(folder/'VISIBILITY.json',dict(assignments=visibility,geom_ids=dict(ids,target_object=target_geoms),
        scope='Native instance-segmentation measurements assist review; pixel counts alone do not establish recognizable objects or complete geometric observability'))
    return output
