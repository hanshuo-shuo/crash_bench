"""Exercise report rendering with audited Task2 tables and synthetic outcome fixtures."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from common import atomic_json,sha
from test_gate_summary import sample
from gate_summary import tables
from report import render


def validate(task2):
    with TemporaryDirectory(prefix='uncertainty_report_fixture_') as directory:
        p=Path(directory);fixture=p/'synthetic_gate';fixture.mkdir()
        runs,split=sample();tables(runs,split,dict(bootstrap_replicates=20,bootstrap_seed=20261009),fixture)
        atomic_json(fixture/'PROVENANCE.json',dict(gate_commit='SYNTHETIC_FIXTURE',gate_gpu_accounting=[dict(job='NO_JOB')],analysis_code_commit='SYNTHETIC_FIXTURE',raw_remote='SYNTHETIC_FIXTURE_NOT_DATA',
            series_committed_usd=0,api_final_path='SYNTHETIC',api_final_sha256='SYNTHETIC',actual_series_gpu_seconds=0,authorized_series_gpu_seconds=172800))
        atomic_json(fixture/'GATE_ANALYSIS_COMPLETE.json',dict(passed=True,gate_rollouts=300,test_rollouts_per_arm=90,files={f.name:sha(f) for f in fixture.iterdir()}))
        output=p/'discarded_report';render(task2,fixture,output)
        receipt=json.loads((output/'REPORT_COMPLETE.json').read_text())
        assert receipt['passed'] and len(receipt['figures'])==6
        assert (output/'REPORT.html').stat().st_size>100000
        assert (output/'figures.pdf').stat().st_size>10000
        print(json.dumps(dict(passed=True,real_task2_table_render=True,gate_outcomes='synthetic_only',no_gate_results_claimed=True,temporary_fixture_removed=True,new_rollouts=0,api_calls=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task2',type=Path);a=p.parse_args();validate(a.task2.resolve())
