"""Task-1 artifact export only; no fitting, thresholds or task-2 evaluation."""
import argparse
import json
from pathlib import Path
import statistics
from common import FIELDS, atomic_json, sha


def main(root):
    import pyarrow as pa
    import pyarrow.parquet as pq
    progress = json.loads((root/'COMPUTE_COMPLETE.json').read_text())
    if progress['completed']!=9: raise RuntimeError('Incomplete bounded schedule')
    validation=json.loads((root/'SAMPLING_VALIDATION.json').read_text())
    equality=json.loads((root/'ROLLOUT_EQUALITY.json').read_text())
    if not validation['serial_batch_passed'] or not validation['native_action_array_equal'] or not equality['passed']:
        raise RuntimeError('Numeric or trajectory validation failed')
    rows=[]; summaries=[]
    for result in progress['results']:
        path=root/'runs'/result['spec']['name']
        inferences=[json.loads(x) for x in (path/'inferences.jsonl').read_text().splitlines()]
        steady=inferences[1:]
        result.update(native_steady_median_seconds=statistics.median(x['native_seconds'] for x in steady) if steady else None,
            diagnostic_steady_median_seconds=statistics.median(x['diagnostic_seconds'] for x in steady) if steady else None)
        summaries.append(result)
        if result['spec']['diagnostics']:
            rows.extend(json.loads(x) for x in (path/'rows.jsonl').read_text().splitlines())
    output=root/'data'; output.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), output/'rollouts.parquet', compression='zstd')
    back=pq.read_table(output/'rollouts.parquet')
    if back.num_rows != len(rows) or any(k not in back.column_names for k in FIELDS): raise RuntimeError('Parquet readback failed')
    metrics=dict(runs=summaries, rows=len(rows), samples=8, validation=validation, actual_rollout_equality=equality,
                 parquet_sha256=sha(output/'rollouts.parquet'), task2_started=False, full_matrix_submitted=False)
    gpu_rows=[line.split(',') for line in (root/'gpu_memory.csv').read_text().splitlines()]
    metrics['sampled_gpu_memory_peak_mib']=max(int(x[-1].strip()) for x in gpu_rows)
    metrics['gpu_memory_sampling_seconds']=2
    atomic_json(root/'METRICS.json',metrics)
    (root/'README.md').write_text('''# Task-1 uncertainty timing and smoke

Nine executions: one no-diagnostic nominal timing baseline followed by two initial states × two paired seeds × nominal/AEGIS with M=8.
The baseline is excluded from rollouts.parquet. Same-state/same-seed baseline and diagnostic execution commands and physical bytes must match exactly.
Sampling checks compare fixed-noise serial and shared-prefix batch outputs at atol=rtol=1e-6; no tolerance widening. Production actions bracket diagnostics at the same input/RNG.
disagreement = mean over H=10 and valid D=7 of sample std (ddof=1), divided by frozen checkpoint action std AFTER native output transforms once. Model padding dimensions are excluded.
churn = normalized mean absolute difference of current chunk[0:5] and previous chunk[5:10]; first inference is null. Inference measures are held over the five executed commands; infer_boundary identifies independent inference observations.
act_norm is the seven-dimensional applied env command L2 divided by checkpoint std; act_norm6 is provided separately. Raw/proposed/applied and controller-clipped commands are in steps.jsonl. Command magnitude is not mechanical speed.
min_dist is pre-action signed collision-geometry distance in meters between every robot collision geom and the active protected obstacle, on independent MuJoCo data. Every read checks that live physics/model body poses are unchanged.
step is a one-based executed action after twenty settling actions. Control frequency is 20 Hz; video FPS and wall-clock inference duration are different clocks. No diagnostic time advances simulation.
crashed uses the unchanged official obstacle L1 displacement >1 mm proxy and remains true after first trigger; collisions do not stop execution. It is not a physical no-contact label.
time_to_crash = first collision action - current action; zero at collision, null after collision, -1 throughout never-crashed episodes. outcome is final crash/safe_incomplete/safe_success. Future labels are never sampler inputs.
Nominal snapshots every twenty actions contain physics/integration arrays, marker poses, controller, gripper, observation, environment clock, action queue, previous chunk and Python/NumPy/server JAX RNG. They are save-only; arbitrary mid-rollout restoration has NOT been validated.
The original AEGIS identity/perception/geometry and six-axis QP retain their semantics; each AEGIS rollout gets a fresh GLM-4.5V/Z.AI call. API token usage and actual/unknown costs are preserved in the campaign-wide $3 ledger and API_FINAL.json.
No task-2 analysis, gate, threshold search, training or full matrix submission occurred.
''')
    atomic_json(root/'COMPLETE.json',dict(runs=9,smoke_runs=8,rows=len(rows),parquet_sha256=metrics['parquet_sha256'],sampling_passed=True,rollout_equality_passed=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();main(a.root)
