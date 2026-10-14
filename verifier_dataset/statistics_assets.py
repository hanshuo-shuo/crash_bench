"""Download fixed wheels on login; extract and test only on compute nodes."""
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request
import zipfile
from .inventory import ASSETS,read,sha,write

ROOT=ASSETS/'verifier_dataset/statistics_assets/20261014_sklearn_1_5_2'
BASE={'numpy':'1.26.4','scipy':'1.15.3','pillow':'11.2.1'}
PACKAGES={'scikit-learn':'1.5.2','joblib':'1.4.2','threadpoolctl':'3.5.0'}


def base_check():
    if sys.version_info[:2]!=(3,11) or platform.machine()!='x86_64':raise RuntimeError('Unexpected statistical runtime')
    for name,version in BASE.items():
        if metadata.version(name)!=version:raise RuntimeError('Statistical base package changed: '+name)


def stage():
    base_check();ROOT.mkdir(parents=True,exist_ok=False);(ROOT/'wheels').mkdir()
    wheels={}
    for name,version in PACKAGES.items():
        with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json') as response:info=json.load(response)
        suffix='cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl' if name=='scikit-learn' else 'py3-none-any.whl'
        matches=[x for x in info['urls'] if x['filename'].endswith(suffix)]
        if len(matches)!=1:raise RuntimeError('Ambiguous fixed platform wheel')
        wheel=matches[0];path=ROOT/'wheels'/wheel['filename'];urllib.request.urlretrieve(wheel['url'],path)
        if sha(path)!=wheel['digests']['sha256']:raise RuntimeError('Authoritative PyPI wheel hash differs')
        wheels[path.name]=sha(path)
    write(ROOT/'STAGED.json',dict(base_packages=BASE,overlay_packages=PACKAGES,wheels=wheels,
        downloaded_from='Official PyPI package/version metadata with SHA256 verification',
        runtime=sys.version,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),api_calls=0))
    print(json.dumps(dict(staged=str(ROOT),wheels=len(wheels))),flush=True)


def prepare(run):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Dependency extraction/testing belongs on a compute node')
    base_check();staged=read(ROOT/'STAGED.json')
    if read(run/'RECEIPT.json')['statistics_assets_sha256']!=sha(ROOT/'STAGED.json'):
        raise RuntimeError('Statistical staging differs from the submitted receipt')
    if staged['base_packages']!=BASE or staged['overlay_packages']!=PACKAGES:raise RuntimeError('Fixed statistical dependency set changed')
    package=run/'statistics_packages';package.mkdir(exist_ok=True);files={}
    for name,digest in staged['wheels'].items():
        path=ROOT/'wheels'/name
        if sha(path)!=digest:raise RuntimeError('Staged statistical wheel changed')
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                if entry.is_dir():continue
                rel=Path(entry.filename)
                if rel.is_absolute() or '..' in rel.parts:raise RuntimeError('Unsafe wheel member')
                target=package/rel;data=archive.read(entry)
                if target.exists():
                    if target.read_bytes()!=data:raise RuntimeError('Statistical overlay changed')
                else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
                files[str(rel)]=sha(target)
    sys.path.insert(0,str(package))
    for name,version in PACKAGES.items():
        if metadata.version(name)!=version:raise RuntimeError('Wrong statistical overlay version')
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    model=LogisticRegression().fit([[-2],[-1],[1],[2]],[0,0,1,1])
    if roc_auc_score([0,0,1,1],model.predict_proba([[-2],[-1],[1],[2]])[:,1])!=1:raise RuntimeError('Statistical runtime smoke failed')
    record=dict(passed=True,staged_sha256=sha(ROOT/'STAGED.json'),base_packages=BASE,overlay_packages=PACKAGES,files=files)
    dest=run/'STATISTICS_READY.json'
    if dest.exists():
        if read(dest)!=record:raise RuntimeError('Statistical runtime provenance changed')
    else:write(dest,record)
    print('Fixed statistical runtime verified',flush=True)


if __name__=='__main__':
    if sys.argv[1]=='stage':stage()
    elif sys.argv[1]=='prepare':prepare(Path(sys.argv[2]))
    else:raise RuntimeError('Unknown dependency stage')
