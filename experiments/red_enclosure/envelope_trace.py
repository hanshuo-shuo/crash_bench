"""Validated action-only view of the legacy two-step-zero trace format."""
def recorded_rows(rows):
    rows=[r for r in rows if r['stage']=='suffix']
    initial=[r for r in rows if r['step']==0]
    actions=[r for r in rows if r['step']>0]
    if not initial or any(r['action'] is not None for r in initial):raise ValueError('Bad initial records')
    if len({r['physics_sha256'] for r in initial})!=1:raise ValueError('Initial records disagree')
    if [r['step'] for r in actions]!=list(range(1,len(actions)+1)):raise ValueError('Missing/duplicated action step')
    if any(r['action'] is None or len(r['action'])!=7 for r in actions):raise ValueError('Bad action record')
    return [initial[0]]+actions
