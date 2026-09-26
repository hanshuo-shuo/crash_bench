#!/usr/bin/env python3
"""Network-only asset staging. Compilation and rendering belong in Slurm."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'configs/reproduction.json').read_text())
ASSETS = Path(os.environ.get('CB_ASSETS', '/projects/p33100/siosio/crashbench_safelibero'))
UPSTREAM = ROOT / 'third_party/vlsa-aegis'


def get(url, headers=None):
    req = urllib.request.Request(url, headers={'User-Agent': 'CrashBench-reproduction/1', **(headers or {})})
    return urllib.request.urlopen(req, timeout=120)


def download(url, dest, expected=None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected:
        with dest.open('rb') as f:
            valid = hashlib.file_digest(f, 'sha256').hexdigest() == expected
        if valid:
            return {'path': str(dest), 'sha256': expected, 'bytes': dest.stat().st_size, 'url': url}
    if dest.exists() and not expected:
        raise RuntimeError('Existing unverified destination: ' + str(dest))
    tmp = dest.with_name(dest.name + '.partial')
    h = hashlib.sha256()
    with get(url) as src, tmp.open('wb') as out:
        while chunk := src.read(4 * 1024 * 1024):
            out.write(chunk)
            h.update(chunk)
    digest = h.hexdigest()
    if expected and digest != expected:
        raise RuntimeError('SHA256 mismatch: ' + str(dest))
    tmp.replace(dest)
    return {'path': str(dest), 'sha256': digest, 'bytes': dest.stat().st_size, 'url': url}


def stage_oci():
    """Stage an OCI layout without extraction, squashfs conversion or compilation."""
    image = CFG['container_image']
    repo, tag = image.rsplit(':', 1)
    token_url = 'https://auth.docker.io/token?' + urllib.parse.urlencode({'service':'registry.docker.io','scope':f'repository:{repo}:pull'})
    token = json.load(get(token_url))['token']
    headers = {'Authorization': 'Bearer ' + token, 'Accept': ', '.join(['application/vnd.oci.image.index.v1+json','application/vnd.docker.distribution.manifest.list.v2+json','application/vnd.docker.distribution.manifest.v2+json','application/vnd.oci.image.manifest.v1+json'])}
    base = f'https://registry-1.docker.io/v2/{repo}'
    data = get(base + '/manifests/' + tag, headers).read()
    manifest = json.loads(data)
    if 'manifests' in manifest:
        match = [m for m in manifest['manifests'] if m.get('platform', {}).get('architecture') == 'amd64' and m.get('platform', {}).get('os') == 'linux']
        if len(match) != 1:
            raise RuntimeError('Ambiguous linux/amd64 container')
        data = get(base + '/manifests/' + match[0]['digest'], headers).read()
        manifest = json.loads(data)
    digest = hashlib.sha256(data).hexdigest()
    if digest != CFG['container_manifest_sha256']:
        raise RuntimeError('Container digest changed; refusing mutable tag')
    layout = ASSETS / 'oci/cudagl-ubuntu2004'
    blobdir = layout / 'blobs/sha256'
    blobdir.mkdir(parents=True, exist_ok=True)
    (blobdir / digest).write_bytes(data)
    (layout / 'oci-layout').write_text('{"imageLayoutVersion":"1.0.0"}\n')
    (layout / 'index.json').write_text(json.dumps({'schemaVersion':2,'manifests':[{'mediaType':manifest['mediaType'],'digest':'sha256:'+digest,'size':len(data),'annotations':{'org.opencontainers.image.ref.name':'aegis'}}]}, indent=2)+'\n')
    descriptors = [manifest['config'], *manifest['layers']]
    def fetch_blob(d):
        sha = d['digest'].split(':')[1]
        dest = blobdir / sha
        if dest.exists():
            with dest.open('rb') as f:
                h = hashlib.file_digest(f, 'sha256').hexdigest()
            if h == sha:
                return d
            raise RuntimeError('Corrupt existing OCI blob: ' + str(dest))
        tmp = dest.with_suffix('.partial')
        h = hashlib.sha256()
        with get(base + '/blobs/' + d['digest'], headers) as src, tmp.open('wb') as out:
            while chunk := src.read(4*1024*1024):
                out.write(chunk); h.update(chunk)
        if h.hexdigest() != sha or tmp.stat().st_size != d['size']:
            raise RuntimeError('OCI blob verification failed')
        tmp.replace(dest)
        print('OCI layer verified', sha[:12], d['size'], flush=True)
        return d
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch_blob, descriptors))
    return {'image':image,'manifest_sha256':digest,'layout':str(layout),'blobs':descriptors}


def stage_python():
    """Download upstream pins for CPython 3.8; source builds happen offline later."""
    from packaging.tags import cpython_tags, compatible_tags
    from packaging.utils import parse_wheel_filename
    from packaging.specifiers import SpecifierSet
    platforms = [f'manylinux_2_{n}_x86_64' for n in range(31, 4, -1)] + ['manylinux2014_x86_64','manylinux2010_x86_64','manylinux1_x86_64','linux_x86_64']
    tags = list(cpython_tags((3,8), platforms=platforms)) + list(compatible_tags((3,8), interpreter='cp38', platforms=platforms))
    priority = {t:i for i,t in enumerate(tags)}
    pins = []
    omitted = []
    for line in (UPSTREAM / 'main/requirements.txt').read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('-e') or line.split('==')[0] in {'torch','torchvision','torchaudio'}:
            omitted.append(line); continue
        if line.startswith('opencv-python=='):
            line = line.replace('opencv-python==','opencv-python-headless==')
        if '==' not in line:
            raise RuntimeError('Unpinned requirement: ' + line)
        pins.append(line)
    pins += ['msgpack-numpy==0.4.8','wheel==0.45.1','pip==24.3.1']
    wheelhouse = ASSETS / 'wheelhouse'
    def fetch_pin(pin):
        name, version = pin.split('==')
        meta = json.load(get(f'https://pypi.org/pypi/{name}/{version}/json'))
        matches, sdists = [], []
        for row in meta['urls']:
            if row.get('requires_python') and not SpecifierSet(row['requires_python']).contains('3.8.13'):
                continue
            if row['packagetype'] == 'sdist':
                sdists.append(row)
            if row['packagetype'] == 'bdist_wheel':
                wheel_tags = parse_wheel_filename(row['filename'])[3]
                scores = [priority[t] for t in wheel_tags if t in priority]
                if scores:matches.append((min(scores), row))
        if matches:row = sorted(matches,key=lambda x:x[0])[0][1]
        elif sdists:row = sdists[0]
        else:raise RuntimeError('No Python 3.8 distribution: ' + pin)
        item = download(row['url'], wheelhouse / row['filename'], row['digests']['sha256'])
        item.update({'pin':pin,'kind':row['packagetype']})
        print('Python distribution verified', pin, row['packagetype'], flush=True)
        return item
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        distributions = list(pool.map(fetch_pin, sorted(set(pins))))
    (ASSETS / 'simulation-requirements.txt').write_text('\n'.join(sorted(set(pins)))+'\n')
    return {'distributions':distributions,'omitted':omitted,'deviation':'opencv-python-headless at the same version replaces GUI OpenCV; torch/vision come from pinned container; openpi-client comes from pinned source.'}


def stage_torch():
    import html
    import re
    entries=[]
    for package,version in [('torch','1.11.0'),('torchvision','0.12.0'),('torchaudio','0.11.0')]:
        filename=f'{package}-{version}+cu113-cp38-cp38-linux_x86_64.whl'
        index=f'https://download.pytorch.org/whl/cu113/{package}/'
        content=get(index).read().decode()
        links=[html.unescape(x) for x in re.findall(r'href="([^"]+)"',content)]
        matches=[urllib.parse.urljoin(index,x) for x in links if urllib.parse.unquote(urllib.parse.urlsplit(x).path).endswith('/'+filename)]
        if len(matches)!=1:raise RuntimeError('No unique official CUDA 11.3 wheel: '+filename)
        parsed=urllib.parse.urlsplit(matches[0]);expected=parsed.fragment.removeprefix('sha256=')
        if len(expected)!=64:raise RuntimeError('Missing published wheel checksum')
        url=urllib.parse.urlunsplit(parsed._replace(fragment=''))
        entries.append(download(url,ASSETS/'wheelhouse'/filename,expected))
        print('Torch wheel verified',filename,flush=True)
    return entries


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    manifest_path = ASSETS / 'staging.json'
    state = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'config':CFG,'stages':{}}
    if state['config'] != CFG:
        old = state['config']
        allowed = {'container_image','container_manifest_sha256','simulation_python'}
        if any(old.get(k)!=CFG.get(k) for k in set(old)|set(CFG) if k not in allowed):
            raise RuntimeError('Existing asset root belongs to a different scientific configuration')
        backup = manifest_path.with_name('staging_before_container_correction.json')
        if backup.exists():raise RuntimeError('Unexpected repeated container migration')
        shutil.copyfile(manifest_path,backup)
        state['config']=CFG
        state['stages'].pop('oci',None)
        if 'python' in state['stages']:
            state['stages']['python']['deviation']='Same-version headless OpenCV; official CUDA 11.3 torch wheels staged separately; openpi-client from pinned source.'
        manifest_path.write_text(json.dumps(state,indent=2)+'\n')
    UPSTREAM.parent.mkdir(parents=True, exist_ok=True)
    if not UPSTREAM.exists():
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',CFG['upstream_url'],str(UPSTREAM)],check=True)
    subprocess.run(['git','-C',str(UPSTREAM),'fetch','origin',CFG['upstream_commit']],check=True)
    subprocess.run(['git','-C',str(UPSTREAM),'checkout','--detach',CFG['upstream_commit']],check=True)
    if subprocess.check_output(['git','-C',str(UPSTREAM),'status','--porcelain'],text=True).strip():
        raise RuntimeError('Upstream checkout is dirty')
    def save(stage, value):
        state['stages'][stage]=value
        tmp=manifest_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(manifest_path)
        print('STAGE COMPLETE',stage,flush=True)
    if 'checkpoint' not in state['stages']:
        os.environ['OPENPI_DATA_HOME']=str(ASSETS/'openpi_assets')
        from openpi.shared import download as openpi_download
        checkpoint=Path(openpi_download.maybe_download(CFG['checkpoint_uri']))
        save('checkpoint',{'path':str(checkpoint),'uri':CFG['checkpoint_uri']})
    if 'tokenizer' not in state['stages']:
        os.environ['OPENPI_DATA_HOME']=str(ASSETS/'openpi_assets')
        from openpi.shared import download as openpi_download
        tokenizer=Path(openpi_download.maybe_download('gs://big_vision/paligemma_tokenizer.model'))
        save('tokenizer',{'path':str(tokenizer),'sha256':hashlib.sha256(tokenizer.read_bytes()).hexdigest()})
    if 'bert' not in state['stages']:
        from huggingface_hub import HfApi, snapshot_download
        revision=HfApi().model_info('bert-base-uncased').sha
        path=snapshot_download('bert-base-uncased',revision=revision,cache_dir=str(ASSETS/'huggingface/hub'),allow_patterns=['config.json','pytorch_model.bin','tokenizer.json','tokenizer_config.json','vocab.txt'])
        save('bert',{'path':path,'revision':revision})
    if 'groundingdino' not in state['stages']:
        gd=ASSETS/'GroundingDINO'
        weights=download('https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth',gd/'groundingdino_swint_ogc.pth')
        # Config is stored in the pinned PyPI source distribution in a later stage.
        save('groundingdino',weights)
    if 'python' not in state['stages']:save('python',stage_python())
    if 'runtime' not in state['stages']:
        env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(ASSETS/'python'))
        subprocess.run(['/projects/p33100/siosio/bin/uv','python','install',CFG['simulation_python']],env=env,check=True)
        executables=list((ASSETS/'python').glob('cpython-3.8.20-*/bin/python3.8'))
        if len(executables)!=1:raise RuntimeError('Ambiguous Python runtime')
        save('runtime',{'python':str(executables[0]),'version':CFG['simulation_python']})
    if 'torch' not in state['stages']:save('torch',stage_torch())
    if 'oci' not in state['stages']:save('oci',stage_oci())
    print('ALL ASSETS STAGED',manifest_path,flush=True)


if __name__=='__main__':main()
