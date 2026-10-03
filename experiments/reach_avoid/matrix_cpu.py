"""Frozen two-layout shards; labels are recomputed on each actual scene."""
import argparse,json,os
from pathlib import Path
import mechanisms as m

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--shard',type=int,required=True);a=p.parse_args()
    if a.shard not in range(6):raise ValueError('Expected shard 0..5')
    protocol=json.loads((Path(__file__).parent/'matrix_protocol.json').read_text())
    protocol['layouts']=protocol['layouts'][a.shard*2:a.shard*2+2]
    m.P=protocol;a.root.mkdir()
    try:m.main(a.root)
    except BaseException as error:
        m.r.write(a.root/'STOP.json',dict(error=type(error).__name__,reason=str(error),job=os.environ.get('SLURM_JOB_ID')));raise
