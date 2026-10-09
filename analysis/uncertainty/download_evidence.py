"""Resume an audited data package over existing SSH, then verify all local bytes."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from common import atomic_json,config
from preflight import stream_sha
from receive_evidence import receive


def download(directory,remote_archive):
    directory=directory.resolve()
    report_file=directory/'SHARD_AUDIT.json';manifest=directory/'FILES.json'
    report=json.loads(report_file.read_text())
    if not report['passed'] or report['runs']!=600 or report['shard'] not in [0,1]:raise ValueError('Terminal audited600-case shard required')
    if stream_sha(manifest)!=report['manifest_sha256']:raise ValueError('Manifest changed')
    canonical=Path(remote_archive)
    if not remote_archive.startswith('/projects/p33100/siosio/crashbench_safelibero/uncertainty/') or canonical.name!='evidence.tar':raise ValueError('Only this project audited data package allowed')
    # Resolve /gpfs/projects and /projects aliases without accessing remote data.
    declared=report['archive'].replace('/gpfs/projects/','/projects/',1)
    if declared!=remote_archive:raise ValueError('Remote package differs from audit receipt')
    if any(c in remote_archive+str(directory) for c in ['\n','\r','"','\\']):raise ValueError('Unsupported transfer path characters')
    archive=directory/'evidence.tar';destination=directory/'delivered'
    if destination.exists():raise ValueError('Preserve an existing delivery; do not overwrite')
    if archive.exists() and archive.stat().st_size>report['archive_bytes']:raise ValueError('Local partial archive is oversized')
    parameters=config()['integrity_audit']['delivery']
    batch=directory/'SFTP_RESUME.batch'
    batch.write_text('reget "'+remote_archive+'" "'+str(archive)+'"\n')
    status=directory/'DELIVERY_STATUS.json';attempts=[];started=time.time()
    try:
        for attempt in range(parameters['attempts']):
            atomic_json(status,dict(state='transferring',attempt=attempt+1,started_unix=started,archive=str(archive),experimental_data_fitted=False))
            result=subprocess.run(['sftp','-R',str(parameters['requests']),'-B',str(parameters['buffer_bytes']),'-b',str(batch),
                '-o','ControlPath=/tmp/quest.sock','-o','BatchMode=yes','quest.northwestern.edu'])
            attempts.append(dict(attempt=attempt+1,exit_code=result.returncode,local_bytes=archive.stat().st_size if archive.exists() else 0))
            if result.returncode==0:break
            if attempt+1<parameters['attempts']:time.sleep(parameters['retry_seconds'])
        if result.returncode:raise RuntimeError('Bounded SFTP resume attempts exhausted')
        if archive.stat().st_size!=report['archive_bytes']:raise ValueError('Terminal package size differs')
        atomic_json(status,dict(state='verifying',attempts=attempts,started_unix=started,experimental_data_fitted=False))
        proof=receive(archive,manifest,report_file,destination)
        atomic_json(status,dict(state='verified',attempts=attempts,started_unix=started,finished_unix=time.time(),proof=proof,experimental_data_fitted=False))
    except Exception as error:
        atomic_json(status,dict(state='failed',attempts=attempts,started_unix=started,finished_unix=time.time(),error_type=type(error).__name__,error=str(error),experimental_data_fitted=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('remote_archive');a=p.parse_args();download(a.directory,a.remote_archive)
