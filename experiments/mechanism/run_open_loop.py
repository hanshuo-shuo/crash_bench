"""Open-loop patching runner and summary for the mechanism pilot (HIL_MECHANISM_IDEA.md sections 6-8).

run (GPU, openpi environment, no simulator): for each state of the shard, feed the stored switch-point policy input (the SW
branch's first_policy_input.npz) to pi0.5 under prompt A (base) and B (donor). It uses K noise seeds starting at the state's
own seed. Output goes to ROOT/states/<id>/{ACTIONS.npz, ROW.json, HASHES.json}; ACTIONS.npz keeps every environment-space
chunk, so every reading can be recomputed.
Per-state checks:
  - The reconstructed B input hashes to the SW run's logged input.
  - The A twin's stored input is compared pixel by pixel.
  - openpi's own Policy.infer, after an RNG reset, is compared with the logged native chunks of the A twin and SW.
  - The patcher's base and donor are compared with Policy.infer on the same inputs in the same job.
summarize (CPU): the P1 reading (metrics.fidelity_reading), the P2 reading at f = 0 (metrics.interface_reading), and cell tables
per (pair, phase, spec) -> ROOT/SUMMARY.json.

Usage:
  run_open_loop.py run ROOT --shard K --shards N --checkpoint DIR
  run_open_loop.py summarize ROOT
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent/'feasibility'))
import metrics as M  # noqa: E402

SEED_SPAN = 2 ** 32


def read(path):
    with open(path) as stream:
        return json.load(stream)


def write(path, data):
    temporary = Path(str(path) + '.tmp'); temporary.write_text(json.dumps(data, indent=1)); temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    return dict(api_calls=0, code_commit=os.environ.get('CB_CODE_COMMIT'), slurm_job=os.environ.get('SLURM_JOB_ID'),
                policy_config='pi05_libero', weights_frozen=True, training_steps=0)


def inputs(state):
    """(A on the SW input, B on the SW input, A on the A twin's own stored input, max |pixel difference| of the two inputs)."""
    sw = np.load(state['input_npz']); twin = np.load(state['a_input_npz'])
    obs = {k: sw[k] for k in sw.files}; own = {k: twin[k] for k in twin.files}
    if set(obs) != set(own):
        raise RuntimeError('Input keys differ for %s' % state['id'])
    pixel = max(float(np.abs(obs[k].astype(float) - own[k].astype(float)).max()) for k in obs)
    return dict(obs, prompt=state['prompt_a']), dict(obs, prompt=state['prompt_b']), dict(own, prompt=state['prompt_a']), pixel


def native(policy, data, seed):
    """openpi's Policy.infer on its first request after an RNG reset to `seed` (serve.py's ControlledPolicy protocol)."""
    import jax
    saved = policy._rng
    try:
        policy._rng = jax.random.key(seed)
        return np.asarray(policy.infer(dict(data))['actions']).copy()
    finally:
        policy._rng = saved


def chunk_sha(actions):
    return hashlib.sha256(np.asarray(actions).tobytes()).hexdigest()


def run_state(patcher, policy, state, seeds_per_state, folder):
    from serve import digest
    import patching as P
    started = time.monotonic()
    data_a, data_b, data_a_twin, pixel = inputs(state)
    seeds = [(state['seed'] + j) % SEED_SPAN for j in range(seeds_per_state)]
    nat = dict(a_twin=native(policy, data_a_twin, seeds[0]), a=native(policy, data_a, seeds[0]), b=native(policy, data_b, seeds[0]))
    stored_a, stored_b = np.asarray(state['a_first']['actions']), np.asarray(state['b_first']['actions'])
    names, layers, groups = P.interface_grid(patcher.depth)
    iface, counts = patcher.interface(data_a, data_b, seeds, layers, groups)
    text = counts['count_base'] == counts['count_donor']
    bnames, blayer, bgroups = P.backbone_grid(patcher.depth, text=text)
    backbone, _ = patcher.backbone(data_a, data_b, seeds, blayer, bgroups)
    gaps = patcher.gaps(data_a, data_b)
    floor = M.noise_floor(iface[0]); ok = M.defined(iface[0], iface[1], floor)
    full = M.effect(iface[3], iface[0], iface[1])
    distance = lambda x, y: float(np.linalg.norm(M.path(x) - M.path(y)))
    row = dict(
        id=state['id'], pair=state['pair'], phase=state['phase'], init=state['init'], b_done=state['b_done'],
        twin_a_done=state['twin_a_done'], seeds=seeds, counts=counts,
        input_b_exact=digest(data_b) == state['b_first']['input_sha256'],
        input_a_twin_exact=digest(data_a_twin) == state['a_first']['input_sha256'], a_twin_pixel_max_abs=pixel,
        native_vs_logged=dict(a_twin_linf=float(np.abs(nat['a_twin'] - stored_a).max()),
                              a_twin_sha_equal=chunk_sha(nat['a_twin']) == state['a_first']['native_chunk_sha256'],
                              b_linf=float(np.abs(nat['b'] - stored_b).max()),
                              b_sha_equal=chunk_sha(nat['b']) == state['b_first']['native_chunk_sha256']),
        mech_vs_native=dict(a_linf=float(np.abs(iface[0, 0] - nat['a']).max()), b_linf=float(np.abs(iface[1, 0] - nat['b']).max()),
                            a_path=distance(iface[0, 0], nat['a']), b_path=distance(iface[1, 0], nat['b'])),
        backbone_base_linf=float(np.abs(backbone[0] - iface[0]).max()), backbone_donor_linf=float(np.abs(backbone[1] - iface[1]).max()),
        noise_floor=floor, ab_gap=[distance(iface[1, j], iface[0, j]) for j in range(len(seeds))], defined=ok.tolist(),
        mech_native_ratio=distance(iface[0, 0], nat['a']) / floor if floor > 0 else None,
        noop_max_abs=float(np.abs(iface[2] - iface[0]).max()), full_e=[float(e) for e, d in zip(full, ok) if d],
        interface=M.state_effects(iface, names)['state'], backbone=M.state_effects(backbone, bnames)['state'],
        gaps=dict(image=gaps[:, 0].tolist(), text=gaps[:, 1].tolist()),
        elapsed_seconds=time.monotonic() - started, **provenance())
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'ACTIONS.npz', interface=iface, backbone=backbone, native_a_twin=nat['a_twin'], native_a=nat['a'],
                        native_b=nat['b'], seeds=np.asarray(seeds, np.uint64), gaps=gaps)
    write(folder/'SPECS.json', dict(interface=names, backbone=bnames))
    write(folder/'ROW.json', row)
    write(folder/'HASHES.json', {p.name: sha(p) for p in sorted(folder.iterdir()) if p.name != 'HASHES.json'})
    return row


def run(root, shard, shards, checkpoint):
    import jax
    from openpi.policies import policy_config
    from openpi.training import config
    import patching as P
    root = Path(root); plan = read(root/'PLAN.json'); states = read(root/'STATES.json')['states']
    if sha(root/'STATES.json') != plan['states_sha256']:
        raise RuntimeError('STATES.json changed')
    mine = [s for i, s in enumerate(s for s in states if s['id'] in set(plan['state_ids'])) if i % shards == shard]
    policy = policy_config.create_trained_policy(config.get_config('pi05_libero'), checkpoint)
    if policy._is_pytorch_model:
        raise RuntimeError('JAX policy required')
    patcher = P.Patcher(policy); progress = root/('PROGRESS_%d.json' % shard); done = []
    for state in mine:
        folder = root/'states'/state['id']
        if (folder/'HASHES.json').exists():
            raise RuntimeError('State already written: %s' % state['id'])
        row = run_state(patcher, policy, state, plan['seeds'], folder); done.append(state['id'])
        write(progress, dict(done=len(done), total=len(mine), last=state['id'], last_seconds=row['elapsed_seconds']))
    write(root/('SHARD_%d_COMPLETE.json' % shard), dict(passed=True, states=done, jax_devices=[str(d) for d in jax.devices()],
                                                       **provenance()))


def cells(rows, key, names):
    """(pair, phase) -> spec -> bootstrap median of the state values (None where E is undefined)."""
    out = {}
    for pair in sorted({r['pair'] for r in rows}):
        for phase in sorted({r['phase'] for r in rows}):
            group = [r for r in rows if r['pair'] == pair and r['phase'] == phase]
            if group:
                out['%s f=%s' % (pair, phase)] = {n: M.bootstrap_median([r[key][n] for r in group]) for n in names}
    return out


def summarize(root):
    root = Path(root); plan = read(root/'PLAN.json')
    for k in range(plan['shards']):
        if not read(root/('SHARD_%d_COMPLETE.json' % k))['passed']:
            raise RuntimeError('Shard %d did not pass' % k)
    rows = []
    for sid in plan['state_ids']:
        folder = root/'states'/sid
        for name, digest in read(folder/'HASHES.json').items():
            if sha(folder/name) != digest:
                raise RuntimeError('State output changed: %s/%s' % (sid, name))
        rows.append(read(folder/'ROW.json'))
    tool, detail = M.fidelity_reading(rows)
    names = list(rows[0]['interface']); windows = [n for n in names if n.endswith('_all') and n[1:3] != n[4:6]]
    table = cells(rows, 'interface', names)
    best = {}
    for pair in sorted({r['pair'] for r in rows}):
        cell = table.get('%s f=0' % pair, {})
        values = [cell[n]['median'] for n in windows if cell.get(n)]
        if values:
            best[pair] = max(values)
    backbone_names = sorted({n for r in rows for n in r['backbone']})
    route, route_cells = M.route_reading(rows)
    summary = dict(
        tool=tool, tool_detail=detail, states=len(rows),
        interface_reading=M.interface_reading(best) if best and tool == 'tool_ok' else None, best_window_f0=best,
        route_reading=route if tool == 'tool_ok' else None, route_cells=route_cells,
        switch_auc=M.switch_auc(rows) if tool == 'tool_ok' else None, reading_inits=list(M.READING_INITS),
        native_vs_logged=dict(a_twin_sha_equal=sum(r['native_vs_logged']['a_twin_sha_equal'] for r in rows),
                              b_sha_equal=sum(r['native_vs_logged']['b_sha_equal'] for r in rows),
                              a_twin_linf_max=max(r['native_vs_logged']['a_twin_linf'] for r in rows),
                              b_linf_max=max(r['native_vs_logged']['b_linf'] for r in rows)),
        input_a_twin_exact=sum(r['input_a_twin_exact'] for r in rows), defined_states=sum(sum(r['defined']) >= M.MIN_SEEDS for r in rows),
        interface=table,
        backbone={c: v for c, v in cells([dict(r, backbone={n: r['backbone'].get(n) for n in backbone_names}) for r in rows],
                                          'backbone', backbone_names).items()},
        gaps={'%s f=%s' % (p, f): {g: np.median([r['gaps'][g] for r in rows if r['pair'] == p and r['phase'] == f], axis=0).tolist()
                                    for g in ('image', 'text')}
              for p in sorted({r['pair'] for r in rows}) for f in sorted({r['phase'] for r in rows})},
        **provenance())
    write(root/'SUMMARY.json', summary)
    print(json.dumps({k: summary[k] for k in ('tool', 'tool_detail', 'interface_reading', 'best_window_f0', 'route_reading', 'switch_auc', 'native_vs_logged')},
                     indent=1))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['run', 'summarize']); parser.add_argument('root')
    parser.add_argument('--shard', type=int, default=0); parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--checkpoint'); args = parser.parse_args()
    if args.action == 'run':
        run(args.root, args.shard, args.shards, args.checkpoint)
    else:
        summarize(args.root)
