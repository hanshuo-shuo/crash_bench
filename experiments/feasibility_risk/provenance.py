"""Compute-node-only content/precision/device receipt for common-process FR-1B."""
import argparse,collections,hashlib,importlib.metadata,json,os,platform,subprocess
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main(root,verify=False):
    checkpoint=Path(os.environ['CB_ASSETS'])/'openpi_assets/openpi-assets/checkpoints/pi05_libero'
    files={str(p.relative_to(checkpoint)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(checkpoint.rglob('*')) if p.is_file()}
    manifest=dict(path=str(checkpoint),files=files,total_bytes=sum(x['bytes'] for x in files.values()))
    dest=root/'CHECKPOINT_SHA256.json'
    if verify:
        assert manifest==json.loads(dest.read_text()),'Checkpoint content changed during process'
        (root/'CHECKPOINT_POST_VERIFY.json').write_text(json.dumps(dict(passed=True,manifest_sha256=sha(dest)))+'\n');return
    dest.write_text(json.dumps(manifest,indent=2)+'\n')
    import jax
    from openpi.training import config
    upstream=Path(os.environ['CB_UPSTREAM'])
    relevant=['openpi/src/openpi/policies/policy.py','openpi/src/openpi/policies/policy_config.py','openpi/src/openpi/policies/libero_policy.py','openpi/src/openpi/models/pi0.py','openpi/src/openpi/models/pi0_config.py','openpi/src/openpi/models/model.py','openpi/src/openpi/models/gemma.py','openpi/src/openpi/models/siglip.py','openpi/src/openpi/transforms.py','openpi/src/openpi/training/config.py']
    sources={p:sha(upstream/p) for p in relevant}
    packages={k:importlib.metadata.version(k) for k in ('jax','jaxlib','flax','numpy','orbax-checkpoint','transformers')}
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    metadata=dict(node=platform.node(),python=platform.python_version(),packages=packages,device=gpu,jax_devices=[str(d) for d in jax.devices()],model_config=repr(config.get_config('pi05_libero').model),jax_default_matmul_precision=str(jax.config.jax_default_matmul_precision),jax_enable_x64=bool(jax.config.jax_enable_x64),environment={k:os.environ.get(k) for k in ('XLA_FLAGS','XLA_PYTHON_CLIENT_PREALLOCATE','JAX_DEFAULT_MATMUL_PRECISION')},preprocessing_and_model_source_hashes=sources,checkpoint_manifest_sha256=sha(dest),limitation='Earlier jobs did not capture device UUID, numerical-kernel selection, or checkpoint content hashes; identical historic input/source and checkpoint path are verified, not historic checkpoint bytes. A node difference alone does not establish a numerical cause.')
    (root/'RUNTIME_PROVENANCE.json').write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--verify',action='store_true');a=p.parse_args();main(a.root,a.verify)
