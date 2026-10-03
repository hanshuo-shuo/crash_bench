"""Package analysis, review tables, and compact source evidence for private delivery.

Raw substep arrays stay in their immutable Quest roots and local retrieval paths.
This deliberately contains no credentials or model weights. The archived Git
source preserves the complete original checkout used by the matrix jobs.
"""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def package(session, analysis, destination):
    repo = Path(__file__).resolve().parents[2]
    files = {}
    def add(path, name):
        if path.is_file():
            if name in files:
                raise ValueError('Duplicate archive entry: ' + name)
            files[name] = path
    def tree(root, prefix, suffixes=None):
        if root.exists():
            for path in sorted(root.rglob('*')):
                if path.is_file() and (suffixes is None or path.suffix in suffixes):
                    add(path, prefix + '/' + path.relative_to(root).as_posix())
    tree(analysis, 'analysis')
    tree(session / 'deliverables/evidence', 'evidence')
    tree(session / 'deliverables/tables', 'tables')
    tree(session / 'deliverables/figures', 'figures')
    tree(session / 'provenance', 'provenance')
    tree(repo / 'docs/reach_avoid', 'docs', {'.md', '.json'})
    tree(repo / 'experiments/reach_avoid', 'current_reporting_source', {'.py', '.json', '.sbatch'})
    for name in ('PROMPT_SUMMARY.json', 'BASELINE_PAIRED_STATE_AUDIT.json',
                 'CPU_MATRIX_READBACK.json', 'A100_DEVELOPMENT_CROSSJOB_ANCHORS.json',
                 'DEVELOPMENT_INPUT_ALIASES.json', 'analysis_requirements.txt',
                 'TERMINAL_ACCOUNTING.json', 'RESULT_SUMMARY.json'):
        add(session / name, 'audits/' + name)
    # The compact per-state records allow independent geometric/goal readback.
    compact = {'summary.json', 'geometry.json', 'fixture.json', 'MATERIAL_BALL.json',
               'CERTIFICATE.json', 'VERIFIED.json', 'INHERITED.json', 'FEATURE_AUDIT.json',
               'initial_restore.json', 'SETTLED_POSE.json', 'steps.jsonl',
               'VISIBILITY.json', 'ATTENTION_CUES.json', 'general_vision.json',
               'layer_features.json', 'agentview_policy_224.png',
               'robot0_eye_in_hand_policy_224.png'}
    for kind in ('matrix_cpu', 'matrix_gpu'):
        root = session / 'raw' / kind
        for path in sorted(root.rglob('*')):
            if path.is_file() and (path.name in compact or path.name in {'receipt.json', 'COMPLETE.json'}):
                add(path, 'compact_raw/' + kind + '/' + path.relative_to(root).as_posix())
    index = {name: dict(bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name, path in files.items()}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    readme = f'''Reach-avoid feasibility evidence — 2026-10-03

Report source commit: {commit}
CPU matrix source: 87c5b06ef2d5e7b0f21f38350044e9f495097a59
GPU matrix and frozen analysis source: 465f112ea18bf12204c2c87f1df99c75dd67bd4a
Public upstream: 2457feed5968ae803926e178c8ce8243b9ecdcf9

Start with tables/readout_metrics.csv, tables/incremental_metrics.csv,
evidence/state_outcomes.csv and analysis/INCREMENTAL.json. UNKNOWN labels remain
in all coverage counts. Binary accuracy excludes them. The two visible obstacle
constructors share a single sufficient separator proof family; the certificate
uses declared digital crossing guards and is not unrestricted physical impossibility.

analysis/FEATURES.npz and MANIFEST.json contain every initial-state feature and
the fixed split/outcome metadata needed to refit the declared shallow readouts.
The immutable original source archive and receipts are in provenance/matrix_gpu.
Reporting-only updates are under current_reporting_source. To replay the frozen
analysis, extract provenance/matrix_gpu/source.tar into an empty directory, add
its experiments/reach_avoid to PYTHONPATH, and call grouped_analysis.analyze with
the rows from MANIFEST.json, all arrays from FEATURES.npz, and a new output Path.
Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. Recorded dependencies accompany
the artifact. No simulator or model weights are needed for the shallow refit.

compact_raw retains geometry, certificates, saved verifier hashes, per-command
goal/safety records, feature audits, and actual camera images. Full physics
substep arrays, actions, model inputs, render masks, and original job logs remain
recoverable in their unchanged Quest run roots and the local raw/ retrievals
named in the manifests. These large raw arrays are intentionally not duplicated
in this compact review bundle. Independent verifier hash records identify them.

SHA256_INDEX.json records byte size and SHA-256 of every included source file.
The report is a separate Library deliverable. All upload receipts are recorded
outside the bundle to avoid circular hashing.
'''
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr('README.txt', readme)
        archive.writestr('SHA256_INDEX.json', json.dumps(index, indent=2) + '\n')
        for name, path in files.items():
            archive.write(path, name)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Archive CRC check failed')
        for name, value in index.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != value['sha256']:
                raise RuntimeError('Archive readback mismatch: ' + name)
    receipt = dict(path=str(destination.resolve()), bytes=destination.stat().st_size,
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), files=len(files),
        all_entry_hashes_read_back=True, reporting_commit=commit)
    destination.with_suffix('.verified.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('session', type=Path)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    package(args.session, args.analysis, args.destination)
