"""Route completed collection tapes to replay, preserving unresolved sources."""
from .inventory import candidate, identity, read, sha
from .cached_pool import run_spec


def replay_candidates(root):
    receipt=read(root/'RECEIPT.json')
    if receipt['stage'] not in ('collect-cached-smoke','collect-cached-batch','collect-approved-cache','collect-frozen-perception') or receipt.get('api_mode')!='cache_only':
        raise RuntimeError('This continuation accepts only the approved cache-only collection')
    api=read(root/'API_FINAL.json')
    if api.get('paid_calls')!=0 or api.get('worker_started') is not False:
        raise RuntimeError('Unexpected paid-worker activity')
    complete=read(root/'COLLECTION_COMPLETE.json');sources=read(root/'COLLECTION_SOURCES.json')
    if not complete['passed'] or complete['runs']!=len(complete['rows']):
        raise RuntimeError('Incomplete collection accounting')
    expected={run_spec(receipt['stage'],i,entry)['name']:entry['scene'] for i,entry in enumerate(sources)}
    seen=set();rows=[];unresolved=[]
    for result in complete['rows']:
        spec=result['spec'];name=spec['name']
        if name not in expected or name in seen or spec['scene']!=expected[name]:
            raise RuntimeError('Unexpected or duplicate source identity')
        if spec['method']!='aegis' or spec['repeat']!=0 or spec['diagnostics'] is not False:
            raise RuntimeError('Collection policy specification changed')
        seen.add(name);directory=root/'runs'/name
        if read(directory/'RESULT.json')!=result:raise RuntimeError('Collection/source results disagree')
        manifest=read(directory/'manifest.json')
        if manifest['status']!='complete' or manifest['code_commit']!=receipt['code_commit'] or str(manifest['slurm_job'])!=receipt['job']:
            raise RuntimeError('Collection provenance or terminal status changed')
        row=candidate(result,directory)
        if row['proxy_safe_success'] and row['has_action_tape'] and row['has_settled_state']:
            rows.append(row)
        else:
            unresolved.append(dict(source=row['source_id'],reason='No proxy-safe complete tape',feasibility_label='unknown'))
    for deferred in complete.get('deferred',[]):
        name=deferred['run_name']
        if (name not in expected or name in seen or deferred['source']!=identity(expected[name]) or
            deferred['feasibility_label']!='unknown' or deferred['paid_request_sent'] is not False):
            raise RuntimeError('Deferred source accounting changed')
        seen.add(name);unresolved.append(deferred)
    if seen!=set(expected):raise RuntimeError('Missing collection source accounting')
    if not rows:raise RuntimeError('No completed successful tapes available for contact replay')
    inheritance=dict(collection_root=str(root),source_job=receipt['job'],unresolved=unresolved,
        hashes={name:sha(root/name) for name in ('RECEIPT.json','COLLECTION_COMPLETE.json','COLLECTION_SOURCES.json','API_FINAL.json')},
        input_rows=len(rows),all_input_labels_unknown=True,api_calls=0)
    return rows,inheritance
