"""Allocated-node byte verification for frozen source and consumed input assets."""
import argparse
import json
from pathlib import Path
from common import atomic_json
from preflight import stream_sha


def verify(root):
    plan=json.loads((root/'plan.json').read_text());parent=root.parent.parent
    for item in json.loads((parent/'SOURCE_FILES.json').read_text()):
        if stream_sha(parent/'source'/item['path'])!=item['sha256']:raise RuntimeError('Source bytes changed')
    fingerprints=json.loads((parent/'FINGERPRINTS.json').read_text())
    upstream=Path(plan['upstream_root'])
    for item in fingerprints['upstream_files']:
        if stream_sha(upstream/item['path'])!=item['sha256']:raise RuntimeError('Frozen upstream bytes changed')
    for item in fingerprints['assets']+[fingerprints['container']]:
        path=Path(item['path'])
        if stream_sha(path)!=item['sha256']:raise RuntimeError('Staged asset bytes changed: '+str(path))
    atomic_json(root/'INPUT_BYTES_VERIFIED.json',dict(passed=True,
        source_files=len(json.loads((parent/'SOURCE_FILES.json').read_text())),
        upstream_files=len(fingerprints['upstream_files']),assets=len(fingerprints['assets']),
        source_commit=plan['code_commit'],fingerprints_sha256=stream_sha(parent/'FINGERPRINTS.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();verify(a.root)
