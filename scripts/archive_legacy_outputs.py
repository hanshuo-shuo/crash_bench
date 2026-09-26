#!/usr/bin/env python3
"""Move only this checkout's leftover legacy outputs into a verified raw archive."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

root=Path.home()/'crash_bench'
assert root.resolve()==Path.cwd().resolve()
assert 'hanshuo-shuo/crash_bench' in subprocess.check_output(['git','remote','get-url','origin'],text=True)
assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
archive=Path('/projects/p33100/siosio/crashbench_archive/20260926')
archive.mkdir(parents=True,exist_ok=True)
records=[]
# Do not follow links into model caches, project data, another checkout or another project.
candidates=[root/'results']+[p for p in root.glob('*.log') if not p.name.startswith('cb_')]
for src in candidates:
    if not src.exists() and not src.is_symlink():continue
    target=archive/src.name
    if target.exists():raise RuntimeError('Archive destination already exists: '+str(target))
    entries=[src] if src.is_file() or src.is_symlink() else sorted(src.rglob('*'))
    manifest=[]
    for p in entries:
        rel=str(p.relative_to(src)) if p!=src else '.'
        if p.is_symlink():manifest.append({'path':rel,'symlink':os.readlink(p),'resolved_target':str(p.resolve())})
        elif p.is_file():
            h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
            manifest.append({'path':rel,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
    # Copy and verify before removing any original regular files (cross-fileset safe).
    if src.is_symlink():target.symlink_to(os.readlink(src))
    elif src.is_dir():shutil.copytree(src,target,symlinks=True)
    else:shutil.copy2(src,target)
    for row in manifest:
        p=target if row['path']=='.' else target/row['path']
        if 'symlink' in row:
            original_target=Path(row['resolved_target'])
            try:expected=target/original_target.relative_to(src)
            except ValueError:expected=original_target
            if p.resolve()!=expected.resolve():
                p.unlink();p.symlink_to(expected)
            assert p.is_symlink() and p.resolve()==expected.resolve()
            row['archived_symlink']=os.readlink(p)
        else:
            h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
            assert h.hexdigest()==row['sha256'],p
    record={'original':str(src),'archive':str(target),'files':manifest}
    (archive/(src.name+'.manifest.json')).write_text(json.dumps(record,indent=2)+'\n')
    if src.is_dir() and not src.is_symlink():shutil.rmtree(src)
    else:src.unlink()
    records.append({'original':str(src),'archive':str(target),'entries':len(manifest)})
(archive/'INDEX.json').write_text(json.dumps({'job':os.environ.get('SLURM_JOB_ID'),'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'archives':records},indent=2)+'\n')
print(json.dumps(records,indent=2))
