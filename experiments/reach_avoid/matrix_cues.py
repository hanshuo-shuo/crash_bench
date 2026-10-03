"""Declared privileged localization and simple input-geometry comparators."""
import json
import numpy as np

def projected_ellipsoid(env,geom,camera):
    m=env.sim.model;d=env.sim.data;cam=m.camera_name2id(camera)
    y,x=np.indices((224,224));f=112/np.tan(np.deg2rad(float(m.cam_fovy[cam]))/2)
    ray=np.stack([(111-x)/f,(111-y)/f,-np.ones_like(x)],axis=-1)
    world=ray@np.asarray(d.cam_xmat[cam]).reshape(3,3).T
    rotation=np.asarray(d.geom_xmat[geom]).reshape(3,3);size=np.asarray(m.geom_size[geom])
    # Ellipsoid axes are the target's executed box half-sizes; localization is
    # privileged ground truth, disclosed as an oracle tracker replacement.
    direction=(world@rotation)/size
    origin=((np.asarray(d.cam_xpos[cam])-np.asarray(d.geom_xpos[geom]))@rotation)/size
    a=(direction*direction).sum(-1);b=2*(direction*origin).sum(-1);c=(origin*origin).sum()-1
    disc=b*b-4*a*c
    return ((disc>=0)&((-b+np.sqrt(np.maximum(disc,0)))/(2*a)>0)).astype(float)


def extract(env,audit,folder,attention,metadata,fixture):
    model=env.sim.model;data=env.sim.data
    target=int(audit.ball_source['geom']);maps=[];patches=[]
    for camera in ('agentview','robot0_eye_in_hand'):
        mask=projected_ellipsoid(env,target,camera);maps.append(mask)
        patches.append(mask.reshape(16,14,16,14).mean((1,3)).reshape(-1))
    names=metadata['image_order'];counts=metadata['patches_per_image']
    # OpenPI Libero mapping uses base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb.
    by_name={'base_0_rgb':patches[0],'left_wrist_0_rgb':patches[1]}
    fraction=np.concatenate([by_name.get(n,np.zeros(c)) for n,c in zip(names,counts)])
    if any(n not in by_name and 'right_wrist' not in n for n in names):raise RuntimeError('Unexpected image order')
    mass=(attention*fraction[None,None,:]).sum(-1)
    area=fraction.sum()/sum(len(v) for v in patches)
    density=mass/max(area,1e-12)
    entropy=-(attention*np.log(np.maximum(attention,1e-30))).sum(-1)
    with (folder/'VISIBILITY.json').open() as f:vis=json.load(f)
    visibility=np.asarray([v['entities'][key]['pixels_224']/224**2 for v in vis['views'] for key in ('target','goal','panels')])
    point=np.asarray(data.body_xpos[audit.target_id]);lo=np.asarray(fixture['lower']);hi=np.asarray(fixture['upper'])
    # Simple explicit metadata comparator, deliberately privileged and not a
    # learned native-image semantic test. No certificate/label bit is supplied.
    gap=.020 if fixture['variant'] in ('open','irrelevant') else float(fixture['variant'])
    geometry=np.r_[point-(lo+hi)/2,hi-lo,gap,float(fixture['variant']=='open'),np.asarray(audit.fixed_goal).reshape(-1)-np.tile(point,2)]
    np.savez_compressed(folder/'attention_cues.npz',mass=mass,density=density,entropy=entropy,
        projected_target_masks=np.stack(maps),patch_fraction=fraction,visibility=visibility,geometry=geometry,
        knows_fixed=np.asarray([mass[11,2],density[11,2],entropy[11,2]]))
    (folder/'ATTENTION_CUES.json').write_text(json.dumps(dict(scope='KNOWS-inspired initial K=1, actual action-query/vision-key attention; oracle projected target ellipsoid replaces tracker; no outcome information',
        fixed_unit='One-based layer12/head3 interpreted as array[11,2]',target_projected_area_fraction=float(area)),indent=2)+'\n')
