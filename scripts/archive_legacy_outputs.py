#!/usr/bin/env python3
"""Archive only legacy files owned by this checkout; verify before removal."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def archive_one(src,target):
    src,target=Path(src),Path(target)
    if target.exists() or target.is_symlink():raise RuntimeError('Archive already exists: '+str(target))
    entries=[src] if src.is_file() or src.is_symlink() else sorted(src.rglob('*'))
    manifest=[]
    for p in entries:
        rel=str(p.relative_to(src)) if p!=src else '.'
        if p.is_symlink():manifest.append({'path':rel,'symlink':os.readlink(p),'resolved_target':str(p.resolve())})
        elif p.is_file():manifest.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha256(p)})
    if src.is_symlink():target.symlink_to(os.readlink(src))
    elif src.is_dir():shutil.copytree(src,target,symlinks=True)
    else:shutil.copy2(src,target)
    for row in manifest:
        original=src if row['path']=='.' else src/row['path']
        p=target if row['path']=='.' else target/row['path']
        if 'symlink' in row:
            assert original.is_symlink() and os.readlink(original)==row['symlink']
            original_target=Path(row['resolved_target'])
            try:expected=target/original_target.relative_to(src)
            except ValueError:expected=original_target
            if p.resolve()!=expected.resolve():
                p.unlink();p.symlink_to(expected)
            assert p.is_symlink() and p.resolve()==expected.resolve()
            row['archived_symlink']=os.readlink(p)
        else:
            assert sha256(p)==row['sha256'],p
            assert sha256(original)==row['sha256'],'Source changed during archival: '+str(original)
    # Detect new/deleted files before cleanup, including concurrent additions.
    now=[src] if src.is_file() or src.is_symlink() else sorted(src.rglob('*'))
    assert {str(p.relative_to(src)) if p!=src else '.' for p in now if p.is_file() or p.is_symlink()}=={row['path'] for row in manifest}
    record={'original':str(src),'archive':str(target),'files':manifest}
    (target.parent/(src.name+'.manifest.json')).write_text(json.dumps(record,indent=2)+'\n')
    if src.is_dir() and not src.is_symlink():shutil.rmtree(src)
    else:src.unlink()
    return {'original':str(src),'archive':str(target),'entries':len(manifest)}


def main():
    root=Path.home()/'crash_bench'
    assert root.resolve()==Path.cwd().resolve()
    assert 'hanshuo-shuo/crash_bench' in subprocess.check_output(['git','remote','get-url','origin'],text=True)
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    archive=Path('/projects/p33100/siosio/crashbench_archive/20260926')
    archive.mkdir(parents=True,exist_ok=True)
    candidates=[root/'results']+[p for p in root.glob('*.log') if not p.name.startswith('cb_')]
    records=[]
    for src in candidates:
        if src.exists() or src.is_symlink():records.append(archive_one(src,archive/src.name))
    (archive/'INDEX.json').write_text(json.dumps({'job':os.environ.get('SLURM_JOB_ID'),'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'archives':records},indent=2)+'\n')
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
