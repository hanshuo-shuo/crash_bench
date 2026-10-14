"""Read-only CPU diagnosis of exact-cache misses after approved perception."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from .inventory import read,sha,write
from .replay import ARRAYS


def diagnose(root, collection):
    import numpy as np
    from PIL import Image
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('CPU allocation required')
    if read(collection/'RECEIPT.json')['stage']!='collect-approved-cache':
        raise RuntimeError('Only approved-cache collection is supported')
    rows=[]
    for index,entry in enumerate(read(collection/'COLLECTION_SOURCES.json')):
        name='fd_approved%02d_aegis'%index
        actual=list((collection/'requests'/name).glob('*/request.json'))
        if len(actual)!=1:raise RuntimeError('Exactly one initial perception input required')
        actual=actual[0].parent;wanted=collection/'perception_cache'/entry['approved_request_identity']
        original=Path(entry['approved_source_collection'])
        originals=list((original/'requests').glob('*/'+entry['approved_request_identity']))
        if len(originals)!=1:raise RuntimeError('Approved original source is ambiguous')
        original_run=original/'runs'/originals[0].parent.name
        with Image.open(actual/'input.png') as image:a=np.asarray(image.convert('RGBA')).copy()
        with Image.open(wanted/'input.png') as image:b=np.asarray(image.convert('RGBA')).copy()
        if a.shape!=b.shape:raise RuntimeError('Render dimensions changed')
        delta=np.abs(a.astype(np.int16)-b.astype(np.int16))
        req_a=read(actual/'request.json');req_b=read(wanted/'request.json')
        for req in (req_a,req_b):req['messages'][0]['content'][0]['image_url']['url']='IMAGE'
        equal={}
        with np.load(collection/'runs'/name/'settled_state.npz',allow_pickle=False) as x, np.load(original_run/'settled_state.npz',allow_pickle=False) as y:
            for field in ARRAYS:equal[field]=bool(np.array_equal(x[field],y[field]))
        rows.append(dict(source_id=entry['source_id'],expected_request=wanted.name,actual_request=actual.name,
            nonimage_request_equal=req_a==req_b,png_bytes_equal=sha(actual/'input.png')==sha(wanted/'input.png'),
            rgba_pixels_equal=bool(np.array_equal(a,b)),changed_pixels=int(np.any(a!=b,axis=2).sum()),
            pixel_count=int(a.shape[0]*a.shape[1]),mean_absolute_channel_difference=float(delta.mean()),
            max_channel_difference=int(delta.max()),shape=list(a.shape),settled_arrays_equal=equal,
            all_settled_arrays_equal=all(equal.values()),original_run=str(original_run),current_run=str(collection/'runs'/name),
            expected_image_sha256=sha(wanted/'input.png'),actual_image_sha256=sha(actual/'input.png'),
            rgba_expected_sha256=hashlib.sha256(b.tobytes()).hexdigest(),rgba_actual_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
            cache_alias_allowed=False,automatic_new_request_allowed=False,labels_created=False))
    result=dict(passed=True,rows=rows,source_collection=str(collection),
        collection_receipt_sha256=sha(collection/'RECEIPT.json'),
        counts=dict(total=len(rows),exact_requests=sum(r['expected_request']==r['actual_request'] for r in rows),
                    identical_decoded_images=sum(r['rgba_pixels_equal'] for r in rows),
                    identical_saved_initial_arrays=sum(r['all_settled_arrays_equal'] for r in rows)),
        scope='Read-only drift diagnosis, no aliases, paid requests, feasibility labels or training authorization',
        limitations='Historical cache-miss roots saved physical arrays but not the full rendered model/camera state; equal arrays alone do not prove full observation identity',
        api_calls=0,slurm_job=os.environ['SLURM_JOB_ID'])
    write(root/'REQUEST_DRIFT.json',result)
    write(root/'HASHES.json',{'REQUEST_DRIFT.json':sha(root/'REQUEST_DRIFT.json')})
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--collection-root',type=Path,required=True)
    a=p.parse_args();diagnose(a.root,a.collection_root)
