"""Independent formulas for the pinned native On/In atomic goal predicates."""
import math


def definition(env,goal):
    predicate,target,receiver=goal;m=env.sim.model
    target_object=env.env.get_object(target)
    result=dict(goal=goal,target_body=int(env.env.obj_body_id[target]),
        target_contact_geoms=[m.geom_name2id(n) for n in target_object.contact_geoms])
    if receiver in env.env.object_sites_dict:
        site_object=env.env.object_sites_dict[receiver]
        parent=env.env.get_object(site_object.parent_name)
        result.update(kind='site',site=m.site_name2id(receiver),size=list(map(float,site_object.size)),
            contact_required=predicate=='on' and parent is not None,
            receiver_contact_geoms=[] if parent is None else [m.geom_name2id(n) for n in parent.contact_geoms])
    elif predicate=='on':
        receiver_object=env.env.get_object(receiver)
        result.update(kind='object',receiver_body=int(env.env.obj_body_id[receiver]),contact_required=True,
            receiver_contact_geoms=[m.geom_name2id(n) for n in receiver_object.contact_geoms])
    else:raise RuntimeError('Unsupported independent native goal formula')
    return result


def snapshot(data,definition,synchronized=False):
    import numpy as np
    positions=data.xpos if synchronized else data.body_xpos
    result=dict(target_position=np.asarray(positions[definition['target_body']]).tolist())
    if definition['kind']=='site':
        site=definition['site']
        result.update(receiver_position=np.asarray(data.site_xpos[site]).tolist(),
                      receiver_rotation=np.asarray(data.site_xmat[site]).reshape(3,3).tolist())
    else:result['receiver_position']=np.asarray(positions[definition['receiver_body']]).tolist()
    return result


def evaluate(definition,state,contacts):
    target=state['target_position'];receiver=state['receiver_position'];pred=definition['goal'][0]
    target_geoms=set(definition['target_contact_geoms']);receiver_geoms=set(definition['receiver_contact_geoms'])
    # robosuite check_contact uses presence in the contact array, including a
    # positive-distance margin contact; safety separately uses distance <= 0.
    contact=any((a in target_geoms and b in receiver_geoms) or (b in target_geoms and a in receiver_geoms)
                for a,b,_ in contacts)
    if definition['kind']=='object':
        return bool(pred=='on' and receiver[2]<=target[2] and contact and
                    math.hypot(target[0]-receiver[0],target[1]-receiver[1])<.03)
    R=state['receiver_rotation'];size=definition['size']
    if pred=='in':
        extent=[abs(sum(R[i][j]*size[j] for j in range(3))) for i in range(3)]
        lower=[receiver[i]-extent[i]-(.01 if i==2 else 0.) for i in range(3)]
        upper=[receiver[i]+extent[i] for i in range(3)]
        return all(a<x<b for a,x,b in zip(lower,target,upper))
    if pred!='on':raise RuntimeError('Unsupported site goal')
    delta=[sum(R[i][j]*(target[j]-receiver[j]) for j in range(3)) for i in range(3)]
    within=size[2]-.005<delta[2]<size[2]+.10 and all(abs(delta[i])<size[i] for i in range(2))
    return bool(within and (contact or not definition['contact_required']))
