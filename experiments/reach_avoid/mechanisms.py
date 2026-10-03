"""Visible separator pilot. Certificates inspect executed geometry; witnesses act."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import baseline as b
from certificate import certify, classify, inscribed_radius
from fixtures import slit, cage
r, d = b.r, b.d
P = json.loads((Path(__file__).parent/'mechanism_protocol.json').read_text())


class Audit(d.Audit):
    def __init__(self, env, directory):
        self.radius = 0.; self.rho = 0.; self.ball_source = None
        super().__init__(env, directory)
        dd = self.forward()
        for geom in self.target:
            if int(self.m.geom_type[geom]) != 6: continue
            local = np.asarray(dd.geom_xmat[geom]).reshape(3,3).T @ (dd.body_xpos[self.target_id]-dd.geom_xpos[geom])
            radius = inscribed_radius(self.m.geom_size[geom].tolist(), local.tolist())
            if radius > self.radius:
                self.radius = radius
                self.ball_source = dict(geom=geom, half_sizes=self.m.geom_size[geom].copy(), root_local=local)
        self.rho = max(0., self.radius/math.sqrt(3)-1e-6)
        r.write(directory/'MATERIAL_BALL.json', dict(radius=self.radius, rho=self.rho, source=self.ball_source,
            proof='Root-centered ball contained in an executing target collision box; invariant under all rigid target orientations.'))

    def sample(self, dd, phase, sweep):
        contacts = self.contacts(dd)
        points = np.vstack([np.asarray(dd.body_xpos[self.target_id]), np.asarray(dd.geom_xpos[self.actor_order])])
        hits = []; ball_hits = []
        if not np.isfinite(points).all(): raise RuntimeError('Nonfinite geometry')
        if sweep:
            start = points if self.prev is None else self.prev
            for i,(a,z) in enumerate(zip(start,points)):
                for box in self.boxes:
                    if r.segment_box(a,z,box['lower'],box['upper']): hits.append([i,box['id']])
            for box in self.boxes:
                lo = [x-self.rho for x in box['lower']]; hi = [x+self.rho for x in box['upper']]
                if r.segment_box(start[0],points[0],lo,hi): ball_hits.append(box['id'])
            self.prev = points.copy()
        bad = bool(hits or ball_hits or any(x[3] for x in contacts))
        if bad and self.safe:
            self.first_violation = dict(step=self.step, substep=self.substep, phase=phase,
                hits=hits, material_cube_hits=ball_hits, contacts=[x for x in contacts if x[3]])
        self.safe &= not bad
        row = dict(step=self.step, substep=self.substep, phase=phase, points=points,
            contacts=contacts, sweep_hits=hits, material_cube_hits=ball_hits, safe_history=self.safe)
        self.samples.write(json.dumps(r.clean(row))+'\n'); self.total_samples += 1
        return row


def verify(directory):
    geo=json.loads((directory/'geometry.json').read_text())
    rho=json.loads((directory/'MATERIAL_BALL.json').read_text())['rho']
    actors=set(geo['robot']+geo['target']); obstacles=set(geo['protected'])|{x['id'] for x in geo['boxes']}
    prev=None; safe=True; n=0
    with gzip.open(directory/'samples.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line); n+=1; hits=[]; ball=[]
            if row['phase']!='native_cached':
                start=row['points'] if prev is None else prev
                for i,(a,z) in enumerate(zip(start,row['points'])):
                    for box in geo['boxes']:
                        if r.segment_box(a,z,box['lower'],box['upper']):hits.append([i,box['id']])
                for box in geo['boxes']:
                    if r.segment_box(start[0],row['points'][0],[x-rho for x in box['lower']],[x+rho for x in box['upper']]):ball.append(box['id'])
                prev=row['points']
            contact=any(c[2]<=0 and ((c[0] in actors and c[1] in obstacles) or (c[1] in actors and c[0] in obstacles)) for c in row['contacts'])
            safe &= not bool(contact or hits or ball)
            if hits!=row['sweep_hits'] or ball!=row['material_cube_hits'] or safe!=row['safe_history']:
                raise RuntimeError('Persisted safety audit mismatch')
    result=json.loads((directory/'summary.json').read_text())
    if n!=result['samples'] or safe!=result['safe_history']:raise RuntimeError('Incomplete witness record')
    steps=[json.loads(x) for x in (directory/'steps.jsonl').read_text().splitlines()]
    if any(not r.previous.legal_action(row['action']) for row in steps[1:]):raise RuntimeError('Illegal witness command')
    r.write(directory/'VERIFIED.json',dict(passed=True,samples=n,steps=len(steps)-1,
        hashes={x.name:r.sha(x) for x in directory.iterdir() if x.is_file() and x.name!='VERIFIED.json'}))


def definition(layout, mechanism, variant):
    center = layout['target_xy'] if variant!='irrelevant' else P['irrelevant_center_xy']
    lower=[center[0]-.15,center[1]-.17,-.03]
    upper=[center[0]+.15,center[1]+.17,.17]
    gap=.020 if variant in ('open','irrelevant') else float(variant)
    boxes=(slit if mechanism=='slit' else cage)(lower,upper,gap,open_roof=variant=='open')
    return lower,upper,boxes


def execute(root, layout, mechanism, variant, canonical):
    folder=root/(layout['id']+'_'+mechanism+'_'+variant);folder.mkdir()
    lower,upper,boxes=definition(layout,mechanism,variant)
    r.write(folder/'fixture.json',dict(lower=lower,upper=upper,boxes=boxes,layout=layout,mechanism=mechanism,variant=variant))
    env=None;start=time.monotonic()
    try:
        env,obs=b.make(folder,0,'clear',boxes=boxes,placements={'milk_1':layout['target_xy'],'basket_1':layout['goal_xy']})
        # Fixture settling is disposable construction, not the compared state.
        # Restore the shared naturally settled layout before any scored action.
        r.write(folder/'pre_restore_state.json',r.capture(env))
        obs=r.restore(env,json.loads((canonical/'initial_restore.json').read_text()))
        (folder/'SETTLED_POSE.json').rename(folder/'PRE_RESTORE_SETTLED_POSE.json')
        (folder/'SETTLED_POSE.json').write_bytes((canonical/'SETTLED_POSE.json').read_bytes())
        r.write(folder/'CANONICAL_RESTORE.json',dict(source=str(canonical),
            sha256=r.sha(canonical/'initial_restore.json'),same_full_state=True,
            reason='Do not let obstacle-dependent construction settling change paired initial states.'))
        audit=Audit(env,folder);audit.sample(audit.forward(),'initial_synchronized',True)
        pose=json.loads((folder/'SETTLED_POSE.json').read_text())
        initial_valid=bool(audit.safe and pose['valid'])
        dd=audit.forward()
        cert=certify(audit.boxes,lower,upper,dd.body_xpos[audit.target_id].tolist(),audit.fixed_goal,
            audit.radius,initial_valid,audit.static,audit.ball_source is not None,True)
        r.write(folder/'CERTIFICATE.json',cert);r.write(folder/'initial_restore.json',r.capture(env))
        r.picture(folder/'initial.png',obs)
        ref=r.Reference(r.STATE,'center');row=audit.endpoint(obs,ref)
        # Still attempt the fixed reference on negative states to expose safety
        # behavior. Its failure never supplies the certificate or changes it.
        if initial_valid:
            for step in range(1,P['execution_horizon_T']+1):
                action=ref.step(env,obs)
                if not r.previous.legal_action(action):raise RuntimeError('Illegal reference')
                audit.step=step;audit.substep=0;obs,_,done,_=env.step(action)
                row=audit.endpoint(obs,ref,action)
                if row['safe_success'] or not audit.safe:break
        audit.finish(ref);r.picture(folder/'final.png',obs)
        label=classify(row['safe_success'] and initial_valid,audit.step,P['execution_horizon_T'],cert)
        if not initial_valid:label='unknown'
        summary=dict(layout=layout['id'],mechanism=mechanism,variant=variant,label=label,
            outcome=b.outcome(audit.safe,row['safe_success'],not initial_valid),initial_valid=initial_valid,
            steps=audit.step,samples=audit.total_samples,safe_history=audit.safe,safe_success=row['safe_success'],
            certificate_label=cert['label'],first_violation=audit.first_violation,
            wall_seconds=time.monotonic()-start,execution_horizon_T=P['execution_horizon_T'],seed=7)
        r.write(folder/'summary.json',summary);verify(folder);print(json.dumps(r.clean(summary)),flush=True)
        return summary
    finally:
        if env is not None:env.close()


def main(root):
    r.configure(root);r.write(root/'protocol.json',P);rows=[]
    for layout in P['layouts']:
        canonical=root/('canonical_'+layout['id']);canonical.mkdir()
        env,obs=b.make(canonical,0,'clear',placements={'milk_1':layout['target_xy'],'basket_1':layout['goal_xy']})
        r.write(canonical/'initial_restore.json',r.capture(env));env.close()
        for mechanism in P['mechanisms']:
            for variant in P['variants']:
                rows.append(execute(root,layout,mechanism,variant,canonical))
                r.write(root/'PROGRESS.json',dict(rows=rows))
    r.write(root/'COMPLETE.json',dict(rows=rows,code_commit=os.environ['CB_CODE_COMMIT'],job=os.environ['SLURM_JOB_ID'],api_calls=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    try:main(args.root)
    except BaseException as error:
        r.write(args.root/'STOP.json',dict(error=type(error).__name__,reason=str(error),job=os.environ.get('SLURM_JOB_ID')));raise
