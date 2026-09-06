"""The development evaluator must use the fitted train-only model identities."""
import json
from pathlib import Path
import sys

from scripts.expansion import run_scoped_training as runner


def test_scoped_training_calibrates_and_evaluates_train_only_models(monkeypatch, tmp_path):
    commands = []
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"gate": {"status": "SCOPED_CONTINUE"}, "test_rows_read": 0}))
    data = tmp_path / "data"
    data.mkdir()
    (data / "dataset_manifest.json").write_text('{}')
    out = tmp_path / "output"
    def fake_run(command):
        commands.append(command)
        if "--output-dir" in command:
            directory = Path(command[command.index("--output-dir") + 1])
            directory.mkdir(parents=True, exist_ok=True)
            if "calibrate_selector.py" in command[1]:
                (directory / "statewise_calibration_freeze.json").write_text(json.dumps({"calibration_row_count": 36}))
        if "--output" in command:
            path = Path(command[command.index("--output") + 1])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"gate": {"status": "NO_GO"}}))
    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run", "--merged-dir", str(data), "--artifact-store", str(tmp_path),
        "--gate-a", str(gate), "--output-dir", str(out)])
    runner.main()
    training = [c for c in commands if "train_option_value.py" in c[1]]
    assert len(training) == 5
    assert all(c[c.index("--fit-on") + 1] == "train" for c in training)
    fitted_paths = {c[c.index("--output-dir") + 1] for c in training}
    for script in ("calibrate_selector.py", "analyze_scoped_gate_b.py"):
        command = next(c for c in commands if script in c[1])
        evaluated_paths = {command[i + 1] for i, value in enumerate(command) if value == "--seed-dir"}
        assert evaluated_paths == fitted_paths
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["fit_on"] == "train" and manifest["test_rows_read"] == 0
