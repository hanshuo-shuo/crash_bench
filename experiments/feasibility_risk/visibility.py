"""Audit exact native input pixels, visible segmentation, and projection clipping.

Segmentation and world geometry are audit/baseline privileges, never probe inputs.
"""
import itertools,json
import numpy as np
from PIL import Image

def body_geoms(m,body):
    out=[]
    for g in range(m.ngeom):
        b=int(m.geom_bodyid[g])
        while b:
            if b==body:out.append(g);break
            b=int(m.body_parentid[b])
    return out

def audit_visibility(env,obs,audit,directory,data):
    import run as r
    from openpi_client import image_tools
    before=r.statehash(r.capture(env));m=env.sim.model;d=env.sim.data
    entities=dict(target=body_geoms(m,env.env.obj_body_id[r.STATE['target']]),goal=body_geoms(m,env.env.obj_body_id[r.STATE['goal']]),panels=sorted(audit.red),lid=[g for g in audit.red if m.geom_id2name(g) in ('red_z_high','red_parked_lid')])
    reports=[];image_features=[]
    for camera,key in [('agentview','observation/image'),('robot0_eye_in_hand','observation/wrist_image')]:
        native=np.asarray(env.sim.render(width=1024,height=1024,camera_name=camera))[::-1,::-1]
        resized=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(native),224,224))
        exact=bool(np.array_equal(resized,data[key]))
        seg=np.asarray(env.sim.render(width=1024,height=1024,camera_name=camera,segmentation=True))[::-1,::-1]
        # IDs are nearest-neighbour sampled only for a pixel-count diagnostic;
        # RGB always uses the actual official preprocessing above.
        ids=seg[:,:,1];types=seg[:,:,0];segments={};overlay=np.zeros((224,224,3),np.uint8)
        palette=dict(target=(0,255,255),goal=(255,0,255),panels=(255,0,0),lid=(0,255,0))
        for name,geoms in entities.items():
            mask=np.isin(ids,geoms)&(types==5)
            small=np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize((224,224),Image.Resampling.NEAREST if hasattr(Image,'Resampling') else Image.NEAREST))>0
            yy,xx=np.nonzero(small)
            segments[name]=dict(pixels_1024=int(mask.sum()),pixels_224=int(small.sum()),bbox_224=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())] if len(xx) else None,touches_image_edge=bool(small[0].any() or small[-1].any() or small[:,0].any() or small[:,-1].any()))
            overlay[small]=palette[name]
        Image.fromarray(np.asarray(data[key])).save(directory/(camera+'_policy_224.png'))
        Image.fromarray(overlay).save(directory/(camera+'_segmentation_224.png'))
        # All panel-corner projections check off-frame geometry even when a
        # far wall has zero visible pixels due to natural self-occlusion.
        cam=m.camera_name2id(camera);R=np.asarray(d.cam_xmat[cam]).reshape(3,3);position=np.asarray(d.cam_xpos[cam]);f=112/math_tan_half(float(m.cam_fovy[cam]))
        projections=[]
        for box in audit.boxes:
            corners=np.asarray(list(itertools.product(*zip(box['lower'],box['upper']))));q=(corners-position)@R
            positive=-q[:,2]>0
            uv=np.c_[223-(112+f*q[:,0]/(-q[:,2])),223-(112+f*q[:,1]/(-q[:,2]))]
            projections.append(dict(name=box['name'],all_corners_in_front=bool(positive.all()),all_corners_in_frame=bool(positive.all() and ((uv>=0)&(uv<=223)).all()),bbox=[float(uv[:,0].min()),float(uv[:,1].min()),float(uv[:,0].max()),float(uv[:,1].max())]))
        rgb=np.asarray(data[key],float)/255
        red=(rgb[:,:,0]>.4)&(rgb[:,:,0]>1.6*rgb[:,:,1])&(rgb[:,:,0]>1.6*rgb[:,:,2])
        yy,xx=np.nonzero(red)
        cues=[float(red.mean()),float(xx.mean()/224) if len(xx) else 0.,float(yy.mean()/224) if len(yy) else 0.]
        # Exact 224-to-8 average blocks, no fitting and no test statistics.
        pooled=rgb.reshape(8,28,8,28,3).mean((1,3)).flatten()
        image_features.append(np.r_[pooled,cues]);reports.append(dict(camera=camera,exact_policy_rgb_match=exact,entities=segments,panel_projections=projections,red_cues=cues))
    after=r.statehash(r.capture(env))
    if before!=after:raise RuntimeError('Read-only visibility render altered physics')
    # Relational privileged baseline explicitly includes whether a material
    # target root lies below the actual lid's footprint, not merely lid count.
    point=np.asarray(audit.forward().body_xpos[audit.target_id]);lid=[b for b in audit.boxes if b['name'] in ('red_z_high','red_parked_lid')]
    lid_over=any(b['lower'][0]<=point[0]<=b['upper'][0] and b['lower'][1]<=point[1]<=b['upper'][1] and point[2]<b['lower'][2] for b in lid)
    np.savez_compressed(directory/'cue_baselines.npz',rgb_pooled_and_red=np.concatenate(image_features),geometry=np.r_[len(lid),float(lid_over),point,np.asarray(audit.fixed_goal).flatten(),[v for b in audit.boxes for k in ('lower','upper') for v in b[k]]])
    r.write(directory/'VISIBILITY.json',dict(views=reports,physics_unchanged=True,lid_present=bool(lid),privileged_lid_over_target=bool(lid_over),strict_target_goal_lid_visible=bool(lid and any(v['exact_policy_rgb_match'] and all(v['entities'][k]['pixels_224']>=4 for k in ('target','goal','lid')) for v in reports)),scope='Segmentation and geometry diagnose visibility only. Occluded target location cannot be supplied as an input to a visual feature readout. Parked lid controls lid presence, not lid-over-target relation.'))

def math_tan_half(degrees):return np.tan(np.deg2rad(degrees)/2)
