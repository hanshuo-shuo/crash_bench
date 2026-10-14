"""Record additional original-policy executions; outputs remain unlabeled."""
import argparse
import io
import os
from pathlib import Path
import shutil
import sys
import time
from .inventory import ASSETS, read, sha, write
from .cached_pool import run_spec


class CacheMiss(RuntimeError):
    pass


def check_expected_request(request,expected):
    if request.name!=expected:
        write(request/'CACHE_MISS.json',dict(reason='Reconstructed input differs from approved screenshot/task',
            paid_request_allowed=False,expected=expected,actual=request.name))
        raise CacheMiss('Unapproved reconstructed input; no new payload may leave Quest')


def prepare_response(request,cache_roots,cache_only):
    from scripts.openrouter_perception import read_response
    for cache in cache_roots:
        cached=cache/request.name
        if not (cached/'response.json').is_file():continue
        if (cached/'request.json').read_bytes()!=(request/'request.json').read_bytes():
            raise RuntimeError('Exact cache key has different request bytes')
        read_response(cached)
        if read(cached/'response.json')['response'].get('provider')!='Z.AI':
            raise RuntimeError('Cached provider differs from pinned Z.AI')
        temp=request/'response.cached.tmp';shutil.copyfile(cached/'response.json',temp)
        temp.replace(request/'response.json')
        write(request/'CACHE_INHERITANCE.json',dict(source=str(cached),sha256=sha(cached/'response.json')))
        return
    if cache_only:
        write(request/'CACHE_MISS.json',dict(request_sha256=request.name,paid_request_allowed=False))
        raise CacheMiss('No exact cached perception response; no network call allowed')
    write(request/'NEEDS_API.json',dict(request_sha256=request.name))


def main(root,port):
    repo=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(repo/'analysis/uncertainty'),str(repo/'scripts')]
    from common import config
    import runtime
    from openrouter_perception import export_request,read_response
    original_observer=runtime.Observer
    campaign=read(root/'CAMPAIGN.json')
    cache_only=campaign.get('api_mode')=='cache_only'
    cache_roots=[ASSETS/'batches/20260927T204911Z_d22035816647/perception_cache',
                 ASSETS/'batches/20260926T233314Z_af6ac874d168/perception_cache',ASSETS/'perception_cache']
    entries=read(root/'COLLECTION_SOURCES.json')
    approved={}
    if campaign['stage'] in ('collect-approved-cache','collect-frozen-perception'):
        if not cache_only:raise RuntimeError('Approved cache collection must be offline')
        from .approved_collection import check_cache
        check_cache(root);cache_roots=[root/'perception_cache']
        approved={run_spec(campaign['stage'],i,e)['name']:e['approved_request_identity'] for i,e in enumerate(entries)}
    frozen={run_spec(campaign['stage'],i,e)['name']:e for i,e in enumerate(entries)} if campaign['stage']=='collect-frozen-perception' else {}

    class CachedObserver(original_observer):
        def perception(self,image,instruction,suite):
            import matplotlib.pyplot as plt
            buf=io.BytesIO();plt.imsave(buf,image,format='png')
            request=export_request(buf.getvalue(),instruction,suite,self.runner.root/'requests'/self.spec['name'],budgeted=True)
            if frozen:
                from .frozen_perception import inherit,UnsupportedFrozenInput
                entry=frozen[self.spec['name']]
                try:
                    return inherit(request,root/'perception_cache'/entry['approved_request_identity'],entry,self.directory)
                except UnsupportedFrozenInput as error:
                    write(request/'FROZEN_INPUT_REJECTED.json',dict(reason=str(error),label='unknown',paid_request_allowed=False))
                    raise CacheMiss('Frozen-input diagnostic failed; source remains unknown') from error
            if approved:check_expected_request(request,approved[self.spec['name']])
            prepare_response(request,cache_roots,cache_only)
            deadline=time.monotonic()+300
            while not (request/'response.json').exists():
                if (root/'STOP.json').exists():raise RuntimeError('Budget/network worker stopped')
                if time.monotonic()>deadline:raise RuntimeError('Perception response timed out')
                time.sleep(2)
            return read_response(request)

    runtime.Observer=CachedObserver
    cfg=config();cfg['save_diagnostic_samples']=False
    # No uncertainty sampling or extra action proposals. Retain native policy,
    # original AEGIS QP, original RNG protocol, preprocessing and all tapes.
    runner=runtime.Runner(root,port,cfg);results=[];deferred=[]
    try:
        for i,entry in enumerate(read(root/'COLLECTION_SOURCES.json')):
            if (root/'STOP.json').exists():raise RuntimeError('External stop')
            spec=run_spec(campaign['stage'],i,entry)
            try:
                result=runner.run(spec);results.append(result)
            except CacheMiss:
                deferred.append(dict(source=entry['source_id'],run_name=spec['name'],reason='exact_cache_missing',
                                     feasibility_label='unknown',paid_request_sent=False))
                write(root/'DEFERRED.json',deferred)
            write(root/'COLLECTION_PROGRESS.json',dict(completed=len(results),expected=len(read(root/'COLLECTION_SOURCES.json')),
                deferred=len(deferred),proxy_safe_successes=sum(x['outcome']=='safe_success' for x in results),accepted_dataset_rows=0))
        write(root/'COLLECTION_COMPLETE.json',dict(passed=True,runs=len(results),rows=results,
            deferred=deferred,all_requested_sources_executed=not deferred,
            accepted_dataset_rows=0,labels_created=False,slurm_job=os.environ['SLURM_JOB_ID']))
    except BaseException as error:
        write(root/'STOP.json',dict(stage='collection',error=type(error).__name__,reason=str(error),completed=len(results)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--port',type=int,required=True)
    a=p.parse_args();main(a.root,a.port)
