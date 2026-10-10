import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import shutil
from .common import *


def main(root,assets):
    staged=read(assets/'STAGED.json')
    for name,version in staged['base_packages'].items():require(metadata.version(name)==version,'Base runtime changed: '+name)
    packages=assets/'packages';packages.mkdir(exist_ok=True)
    for name,expected in staged['wheels'].items():
        path=assets/'wheels'/name;require(sha(path)==expected,'Wheel changed')
        with zipfile.ZipFile(path) as archive:
            require(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in archive.namelist()),'Unsafe wheel path')
            for entry in archive.infolist():
                if entry.is_dir():continue
                target=packages/entry.filename
                data=archive.read(entry)
                if target.exists():require(target.read_bytes()==data,'Existing overlay differs')
                else:
                    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    env=dict(os.environ,PYTHONPATH=str(packages)+':'+str(root/'source'))
    with (root/'unit_tests.txt').open('w') as log:
        subprocess.run([sys.executable,'-m','unittest','discover','-s','verifier_training/tests','-v'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    inherited=os.environ.get('CB_VERIFIER_DATASET')
    if inherited:
        source=Path(inherited);audit=read(source/'AUDIT.json')
        require(audit['passed'] and sha(source/'dataset.jsonl')==audit['dataset_sha256'], 'Inherited dataset changed')
        shutil.copytree(source,root/'dataset',ignore=shutil.ignore_patterns('TOKEN_AUDIT.json'))
        require(sha(root/'dataset/dataset.jsonl')==audit['dataset_sha256'], 'Dataset copy changed')
        write(root/'INHERITED_DATASET.json',{'source':str(source),'dataset_sha256':audit['dataset_sha256'],
            'audit_sha256':sha(source/'AUDIT.json'),'reason':'Recheck full identical A/B inference inputs; evidence and targets are unchanged'})
    else:
        subprocess.run([sys.executable,'-m','verifier_training.export','--out',str(root/'dataset')],env=env,check=True)
    # Real Qwen tokenizer/processor integration catches image expansion and chat boundary errors.
    subprocess.run([sys.executable,'-m','verifier_training.token_audit',str(root/'dataset'),str(assets)],env=env,check=True)
    write(root/'PREPARED.json',{'passed':True,'dataset_sha256':sha(root/'dataset/dataset.jsonl'),
        'code_commit':os.environ['CB_CODE_COMMIT'],'slurm_job':os.environ['SLURM_JOB_ID'],'assets':str(assets)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('assets',type=Path);a=p.parse_args();main(a.root,a.assets)
