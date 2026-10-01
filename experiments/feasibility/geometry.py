"""Read-only collision geometry, with an explicitly validated ellipsoid enclosure."""
import itertools
import numpy as np

def active_obstacle(env,obs):
 names=[n.replace('_joint0','') for n in env.sim.model.joint_names if 'obstacle' in n]
 matches=[n for n in names if obs[n+'_pos'][2]>0 and -.5<obs[n+'_pos'][0]<.5 and -.5<obs[n+'_pos'][1]<.5]
 if len(matches)!=1:raise RuntimeError('Active obstacle must be unambiguous: '+str(matches))
 return matches[0]

def object_points(env,name):
 m,d=env.sim.model,env.sim.data
 ids=[];points=[];records=[]
 # Object naming prefixes include all collision child bodies/geoms.
 for i in range(m.ngeom):
  gn=m.geom_id2name(i) or '';bn=m.body_id2name(int(m.geom_bodyid[i])) or ''
  if not (gn.startswith(name+'_') or bn.startswith(name+'_')):continue
  if not (m.geom_contype[i] or m.geom_conaffinity[i]):continue
  kind=int(m.geom_type[i]);size=np.asarray(m.geom_size[i]).copy();rot=np.asarray(d.geom_xmat[i]).reshape(3,3);pos=np.asarray(d.geom_xpos[i])
  if kind==7:
   mesh=int(m.geom_dataid[i]);a=int(m.mesh_vertadr[mesh]);n=int(m.mesh_vertnum[mesh])
   local=np.asarray(m.mesh_vert[a:a+n],dtype=float)
  else:
   # Conservative local bounding boxes for primitives enclose their entire surface.
   if kind==2:half=np.repeat(size[0],3)
   elif kind==3:half=np.array([size[0],size[0],size[0]+size[1]])
   elif kind==5:half=np.array([size[0],size[0],size[1]])
   elif kind in [4,6]:half=size
   else:raise RuntimeError('Unsupported collision primitive '+str(kind))
   local=np.asarray(list(itertools.product(*[[-x,x] for x in half])))
  world=local@rot.T+pos;points.extend(world);ids.append(i)
  records.append({'geom_id':i,'name':gn,'type':kind,'size':size.tolist(),'vertices':len(local)})
 if not points:raise RuntimeError('No collision geometry found for '+name)
 return np.asarray(points),records

def verified_ellipsoid(env,name):
 from utils import fit_ellipse
 points,records=object_points(env,name)
 p,R,axes=fit_ellipse(points,plot=False)
 values=np.sum((((points-p)@R)/axes)**2,axis=1)
 scale=max(1.,float(np.sqrt(values.max())))
 axes=axes*scale+1e-6
 verified=np.sum((((points-p)@R)/axes)**2,axis=1)
 if not np.isfinite(verified).all() or verified.max()>1+1e-8:raise RuntimeError('Geometry enclosure not verified')
 return p,R,axes,points,{'object':name,'collision_geoms':records,'coverage_max':float(verified.max()),'numeric_enclosure_scale':scale,
  'scope':'all mesh collision vertices and conservative primitive bounds; same AEGIS single ellipsoid representation'}
