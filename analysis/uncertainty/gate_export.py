"""Independent completed gate-shard reduction, then every-cell Arrow readback."""
import argparse
import json
from pathlib import Path
from common import atomic_json,sha
from gate import check,schedule


def export(root):
    import pyarrow as pa
    import pyarrow.parquet as pq
    plan=json.loads((root/'plan.json').read_text());complete=root/'COMPUTE_COMPLETE.json'
    progress=json.loads((complete if complete.exists() else root/'PARTIAL.json').read_text());results=progress['results']
    expected=schedule(plan['configuration'],plan['shard'])
    if [r['spec'] for r in results]!=expected[:len(results)]:raise RuntimeError('Gate frozen schedule differs')
    proof,rows=check(root,results);out=root/'data';out.mkdir()
    pq.write_table(pa.Table.from_pylist(rows),out/'rollouts.parquet',compression='zstd')
    if pq.read_table(out/'rollouts.parquet').to_pylist()!=rows:raise RuntimeError('Gate every-cell Arrow readback differs')
    proof.update(complete_matrix_shard=len(results)==150,results=results,parquet_sha256=sha(out/'rollouts.parquet'),source_commit=plan['code_commit'],api_calls=0)
    if (root/'INHERITED_SMOKE.json').exists():
        inherited=json.loads((root/'INHERITED_SMOKE.json').read_text());proof['inherited_smoke']={k:inherited[k] for k in ['original_gate_commit','original_gpu_job','cpu_reaudit_job','cpu_reaudit_sha256']};proof['inherited_smoke']['cases']=4
    atomic_json(root/('COMPLETE.json' if len(results)==150 else 'PARTIAL_EXPORTED.json'),proof)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();export(a.root)
