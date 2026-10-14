"""Offline collection from the exact human-approved perception cache."""
from pathlib import Path
import shutil
from .approved_requests import AUTH_SHA256, STAGE, authorization, validate_payload
from .inventory import ASSETS, read, sha, write
from scripts.openrouter_perception import read_response


def sources(api_root, shard):
    if shard not in (0,1):raise RuntimeError('Only two approved collection shards')
    receipt=read(api_root/'RECEIPT.json');complete=read(api_root/'API_COMPLETE.json')
    if (receipt['stage']!=STAGE or receipt['authorization_sha256']!=AUTH_SHA256 or
            complete.get('passed') is not True or complete.get('logical_requests')!=32 or
            complete.get('authorization_sha256')!=AUTH_SHA256 or (api_root/'STOP.json').exists()):
        raise RuntimeError('Exact approved API batch must complete before collection')
    auth=authorization(api_root/'AUTHORIZATION.json')
    results={r['request_identity']:r for r in complete['rows']}
    if (len(results)!=32 or len(complete['rows'])!=32 or
            set(results)!={r['request_identity'] for r in auth['rows']}):
        raise RuntimeError('Incomplete exact response accounting')
    chosen=[]
    for row in auth['rows']:
        directory=api_root/'requests'/row['request_identity']
        validate_payload(directory,row);read_response(directory)
        response=read(directory/'response.json')['response']
        result=results[row['request_identity']]
        if (result['source_id']!=row['source_id'] or response.get('provider')!='Z.AI' or
                result['response_sha256']!=sha(directory/'response.json') or
                response['choices'][0].get('finish_reason')=='length'):
            raise RuntimeError('Approved response changed or wrong model/provider')
        original=ASSETS/'verifier_dataset'/row['source_collection']
        if Path(row['request_directory']).parent.parent.parent!=original:
            raise RuntimeError('Original source collection path changed')
        candidates=[r for r in read(original/'COLLECTION_SOURCES.json') if r['source_id']==row['source_id']]
        if len(candidates)!=1 or candidates[0]['feasibility_label']!='unknown':
            raise RuntimeError('Source must be unique and remain unknown')
        entry=dict(candidates[0],approved_request_identity=row['request_identity'],
                   approved_response_sha256=result['response_sha256'],approved_source_collection=str(original))
        chosen.append(entry)
    inherited=dict(api_root=str(api_root),authorization_sha256=AUTH_SHA256,
                   completion_sha256=sha(api_root/'API_COMPLETE.json'),shard=shard,shards=2,
                   expected_total_sources=32,additional_paid_calls_allowed=0)
    return chosen[shard::2],inherited


def stage_cache(root,api_root,selected):
    rows=[]
    for entry in selected:
        identity=entry['approved_request_identity'];source=api_root/'requests'/identity
        dest=root/'perception_cache'/identity;dest.mkdir(parents=True,exist_ok=False)
        files={}
        for name in ('request.json','input.png','response.json'):
            shutil.copyfile(source/name,dest/name);files[name]=sha(dest/name)
        rows.append(dict(source_id=entry['source_id'],identity=identity,files=files))
    write(root/'APPROVED_CACHE.json',dict(rows=rows,api_root=str(api_root),authorization_sha256=AUTH_SHA256))


def check_cache(root):
    manifest=read(root/'APPROVED_CACHE.json')
    if manifest['authorization_sha256']!=AUTH_SHA256:raise RuntimeError('Wrong cache authorization')
    for row in manifest['rows']:
        for name,wanted in row['files'].items():
            if sha(root/'perception_cache'/row['identity']/name)!=wanted:
                raise RuntimeError('Staged exact cache changed')
