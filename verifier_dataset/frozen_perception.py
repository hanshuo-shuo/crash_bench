"""Explicit frozen-approved-image perception, never an exact-request cache alias.

This is a dataset witness-collection variant, not an unchanged public AEGIS
benchmark score. Policy inference still uses the actual live images. Only the
one obstacle-name response comes from its identified previously approved image.
"""
from pathlib import Path
from .inventory import read,sha,write
from .replay import ARRAYS
from scripts.openrouter_perception import read_response

MODE='full_AEGIS_with_frozen_approved_perception_v1'


class UnsupportedFrozenInput(RuntimeError):
    pass


def check_inputs(actual,approved,current_state,original_state):
    import numpy as np
    from PIL import Image
    a=read(actual/'request.json');b=read(approved/'request.json')
    for req in (a,b):req['messages'][0]['content'][0]['image_url']['url']='IMAGE'
    if a!=b:raise UnsupportedFrozenInput('Frozen perception task/routing differs')
    with Image.open(actual/'input.png') as im:x=np.asarray(im.convert('RGBA')).copy()
    with Image.open(approved/'input.png') as im:y=np.asarray(im.convert('RGBA')).copy()
    if x.shape!=y.shape:raise UnsupportedFrozenInput('Frozen perception render shape differs')
    delta=np.abs(x.astype(np.int16)-y.astype(np.int16))
    if int(delta.max())>1:raise UnsupportedFrozenInput('Difference exceeds one stored-image quantization level')
    with np.load(current_state,allow_pickle=False) as xstate,np.load(original_state,allow_pickle=False) as ystate:
        if not all(np.array_equal(xstate[field],ystate[field]) for field in ARRAYS):
            raise UnsupportedFrozenInput('Frozen perception saved initial physical arrays differ')
    return dict(changed_pixels=int(np.any(x!=y,axis=2).sum()),pixel_count=int(x.shape[0]*x.shape[1]),
                max_channel_difference=int(delta.max()),all_saved_initial_arrays_equal=True,
                full_model_camera_identity_claim=False)


def inherit(actual,approved,entry,current_run):
    original=Path(entry['approved_source_collection'])
    parents=list((original/'requests').glob('*/'+entry['approved_request_identity']))
    if len(parents)!=1:raise RuntimeError('Ambiguous approved source perception')
    state=original/'runs'/parents[0].parent.name/'settled_state.npz'
    diagnostics=check_inputs(actual,approved,current_run/'settled_state.npz',state)
    result=read_response(approved)
    if (sha(approved/'response.json')!=entry['approved_response_sha256'] or
            read(approved/'response.json')['response'].get('provider')!='Z.AI'):
        raise RuntimeError('Frozen response provenance changed')
    write(actual/'FROZEN_PERCEPTION_INHERITANCE.json',dict(mode=MODE,exact_request_cache_hit=False,
        current_request_sha256=actual.name,approved_request_sha256=approved.name,
        response_file=str(approved/'response.json'),response_sha256=sha(approved/'response.json'),
        current_initial_state_sha256=sha(current_run/'settled_state.npz'),
        original_initial_state_sha256=sha(state),original_initial_state=str(state),
        diagnostics=diagnostics,new_api_request_sent=False,
        scope='Previously approved image supplies only the obstacle-name observation; live policy images remain unchanged; new trajectory must be independently replayed'))
    # Do not publish response.json under the different current request identity.
    return result


def select_sources(drift_roots):
    sources=[];api_roots=set();provenance=[]
    for drift in drift_roots:
        report=read(drift/'REQUEST_DRIFT.json')
        if not report['passed']:raise RuntimeError('Failed drift diagnostic')
        collection=Path(report['source_collection'])
        if sha(collection/'RECEIPT.json')!=report['collection_receipt_sha256']:
            raise RuntimeError('Collection provenance changed')
        entries={x['source_id']:x for x in read(collection/'COLLECTION_SOURCES.json')}
        api_roots.add(read(collection/'COLLECTION_PLAN.json')['api_root'])
        for row in report['rows']:
            if row['expected_request']==row['actual_request']:continue
            if not row['nonimage_request_equal'] or not row['all_settled_arrays_equal'] or row['max_channel_difference']>1:
                raise RuntimeError('Unresolved non-quantization drift; not eligible for this variant')
            entry=entries[row['source_id']]
            if entry['approved_request_identity']!=row['expected_request']:raise RuntimeError('Wrong approved source mapping')
            sources.append(entry)
        provenance.append(dict(root=str(drift),sha256=sha(drift/'REQUEST_DRIFT.json')))
    if len(api_roots)!=1 or len(sources)!=18 or len({x['source_id'] for x in sources})!=18:
        raise RuntimeError('This bounded repair requires the exact 18 diagnosed deferred sources')
    return sources,Path(next(iter(api_roots))),dict(mode=MODE,drift_evidence=provenance,
        sources=18,paid_calls_allowed=0,unchanged_public_benchmark_reproduction=False,
        source_selection='All and only the 18 quantified image-drift deferrals; no outcome or threshold search')
