"""Accept only four complete, byte-checked smoke cases; never resumes a trajectory."""
import json
from pathlib import PurePosixPath
from preflight import stream_sha


def read_inherited(root,expected):
    path=root/'INHERITED_SMOKE.json'
    if not path.exists():return []
    saved=json.loads(path.read_text());results=saved['results']
    if not saved['passed'] or len(results)!=4 or [r['spec'] for r in results]!=expected[:4] or any(r['status']!='complete' for r in results):
        raise RuntimeError('Only the exact four complete smoke cases may be inherited')
    if not saved.get('files'):raise RuntimeError('Inherited evidence manifest required')
    names=set();allowed={'runs/'+r['spec']['name'] for r in results}
    for item in saved['files']:
        name=item['path'];p=PurePosixPath(name)
        if name in names or p.is_absolute() or '..' in p.parts or str(p)!=name or '/'.join(p.parts[:2]) not in allowed:
            raise RuntimeError('Unsafe or duplicate inherited evidence path')
        names.add(name)
        if stream_sha(root/name)!=item['sha256']:raise RuntimeError('Inherited evidence changed: '+name)
    for r in results:
        directory=root/'runs'/r['spec']['name']
        actual={str(p.relative_to(root)) for p in directory.rglob('*') if p.is_file() and not p.is_symlink()}
        if actual!={n for n in names if n.startswith('runs/'+r['spec']['name']+'/')}:raise RuntimeError('Inherited file coverage changed')
        required={'runs/'+r['spec']['name']+'/'+n for n in ['RESULT.json','settled_state.npz','steps.jsonl','rows.jsonl','inferences.jsonl','CONDITIONAL_ACTION_EQUALITY.json','ZERO_COMMAND_SEMANTICS.json']}
        if not required<=names:raise RuntimeError('Incomplete inherited evidence coverage')
        if json.loads((root/'runs'/r['spec']['name']/'RESULT.json').read_text())!=r:raise RuntimeError('Inherited result changed')
    return results
