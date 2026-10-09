"""Verify a terminal audit package and extract files before restoring inert symlinks."""
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import tarfile
from common import atomic_json
from preflight import stream_sha


def checked_paths(manifest,shard):
    paths=set();links=set()
    for item in manifest:
        path=PurePosixPath(item['path'])
        if path.is_absolute() or '..' in path.parts or not path.parts or str(path)!=item['path']:
            raise ValueError('Unsafe or noncanonical package path')
        if item['path'] in paths:raise ValueError('Duplicate package path')
        if item['kind'] not in ['file','symlink']:raise ValueError('Unsupported file kind')
        paths.add(item['path'])
        if item['kind']=='symlink':links.add(item['path'])
    for path in paths:
        if any(str(parent) in links for parent in PurePosixPath(path).parents):
            raise ValueError('Package contains a file beneath a symlink')
    return {'shard_%d/'%shard+item['path']:item for item in manifest}


def receive(archive,manifest_file,report_file,destination):
    report=json.loads(report_file.read_text());manifest=json.loads(manifest_file.read_text())
    if not report['passed'] or report['shard'] not in [0,1] or report['runs']!=600:raise ValueError('Full terminal audit required')
    if stream_sha(manifest_file)!=report['manifest_sha256'] or stream_sha(archive)!=report['archive_sha256']:
        raise ValueError('Downloaded package/manifest hash differs')
    expected=checked_paths(manifest,report['shard'])
    with tarfile.open(archive,'r') as tar:
        members=tar.getmembers()
        if len(members)!=len(expected) or set(x.name for x in members)!=set(expected):raise ValueError('Tar member coverage differs')
        for member in members:
            item=expected[member.name]
            if item['kind']=='file' and (not member.isfile() or member.size!=item['bytes']):raise ValueError('Tar regular file type/size differs')
            if item['kind']=='symlink' and (not member.issym() or member.linkname!=item['target']):raise ValueError('Tar symlink target differs')
        destination.mkdir(exist_ok=False)
        atomic_json(destination/'INCOMPLETE.json',dict(archive=str(archive),archive_sha256=report['archive_sha256']))
        count=0
        for member in members:
            item=expected[member.name]
            if item['kind']!='file':continue
            target=destination/member.name;target.parent.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256()
            with tar.extractfile(member) as source,target.open('xb') as output:
                while True:
                    block=source.read(1024*1024)
                    if not block:break
                    digest.update(block);output.write(block)
            if digest.hexdigest()!=item['sha256']:raise ValueError('Extracted file hash differs: '+member.name)
            count+=1
            if count%2000==0:print(json.dumps(dict(verified_regular_files=count)),flush=True)
        for member in members:
            item=expected[member.name]
            if item['kind']=='symlink':
                target=destination/member.name;target.parent.mkdir(parents=True,exist_ok=True);os.symlink(item['target'],target)
    proof=dict(passed=True,shard=report['shard'],runs=report['runs'],actions=report['actions'],diagnostic_inferences=report['diagnostic_inferences'],
        archive_sha256=report['archive_sha256'],manifest_sha256=report['manifest_sha256'],regular_files=count,
        symlinks=sum(x['kind']=='symlink' for x in manifest),all_delivered_regular_file_hashes_verified=True,
        symlink_targets_preserved_without_dereference=True,experimental_data_fitted=False,collection_commit=report['collection_commit'],audit_commit=report['audit_commit'])
    atomic_json(destination/'LOCAL_EVIDENCE_VERIFIED.json',proof);(destination/'INCOMPLETE.json').unlink()
    print(json.dumps(proof,indent=2));return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('manifest',type=Path);p.add_argument('report',type=Path);p.add_argument('destination',type=Path)
    a=p.parse_args();receive(a.archive,a.manifest,a.report,a.destination)
