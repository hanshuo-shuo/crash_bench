"""Record additional original-policy executions; outputs remain unlabeled."""
import argparse
import io
import os
from pathlib import Path
import shutil
import sys
import time
from .inventory import ASSETS, read, sha, write


def main(root,port):
    repo=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(repo/'analysis/uncertainty'),str(repo/'scripts')]
    from common import config
    import runtime
    from openrouter_perception import export_request,read_response
    original_observer=runtime.Observer

    class CachedObserver(original_observer):
        def perception(self,image,instruction,suite):
            import matplotlib.pyplot as plt
            buf=io.BytesIO();plt.imsave(buf,image,format='png')
            request=export_request(buf.getvalue(),instruction,suite,self.runner.root/'requests'/self.spec['name'],budgeted=True)
            cached=ASSETS/'perception_cache'/request.name
            if (cached/'response.json').is_file():
                # read_response checks request and response identity/model. Copy
                # only a byte-identical request's response, never a nearby image.
                if (cached/'request.json').read_bytes()!=(request/'request.json').read_bytes():
                    raise RuntimeError('Exact cache key has different request bytes')
                read_response(cached)
                if read(cached/'response.json')['response'].get('provider')!='Z.AI':
                    raise RuntimeError('Cached provider differs from pinned Z.AI')
                temp=request/'response.cached.tmp';shutil.copyfile(cached/'response.json',temp)
                temp.replace(request/'response.json')
                write(request/'CACHE_INHERITANCE.json',dict(source=str(cached),sha256=sha(cached/'response.json')))
            else:
                # Publish eligibility only after checking the cache, avoiding a
                # race in which the worker charges while a cached reply is copied.
                write(request/'NEEDS_API.json',dict(request_sha256=request.name))
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
    runner=runtime.Runner(root,port,cfg);results=[]
    try:
        for i,entry in enumerate(read(root/'COLLECTION_SOURCES.json')):
            if (root/'STOP.json').exists():raise RuntimeError('External stop')
            spec=dict(name='fd_goal%02d_aegis'%i,scene=entry['scene'],repeat=0,method='aegis',diagnostics=False)
            result=runner.run(spec);results.append(result)
            write(root/'COLLECTION_PROGRESS.json',dict(completed=len(results),expected=len(read(root/'COLLECTION_SOURCES.json')),
                proxy_safe_successes=sum(x['outcome']=='safe_success' for x in results),accepted_dataset_rows=0))
        write(root/'COLLECTION_COMPLETE.json',dict(passed=True,runs=len(results),rows=results,
            accepted_dataset_rows=0,labels_created=False,slurm_job=os.environ['SLURM_JOB_ID']))
    except BaseException as error:
        write(root/'STOP.json',dict(stage='collection',error=type(error).__name__,reason=str(error),completed=len(results)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--port',type=int,required=True)
    a=p.parse_args();main(a.root,a.port)
