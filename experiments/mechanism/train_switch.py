"""Fine-tune pi0.5 on teacher data mixed with LIBERO's own demonstrations (docs/generative_verifier/HIL_TRAIN_PROTOCOL.md).

  train_switch.py ROOT --data    CPU check: builds the configuration and the mixed data loader and draws one batch
  train_switch.py ROOT --probe   chooses the batch size -> ROOT/PROBE.json
  train_switch.py ROOT           reads ROOT/TRAIN.json and ROOT/PROBE.json, trains with openpi's own scripts/train.py,
                                 writes ROOT/TRAINED.json

The batch size is the first of TRAIN.json's batch_sizes that fits the GPU: the probe runs PROBE_STEPS training steps with
each candidate in a process of its own, saves nothing, and moves to the next candidate only on an out-of-memory error.

The run is openpi's pi05_libero training (same model, transforms, loss and normalization statistics; the statistics are
the base checkpoint's own) with three differences: the weights start from the pi05_libero checkpoint, the optimizer
settings come from TRAIN.json, and every training example is drawn from one of several LeRobot datasets with fixed
probabilities (MIX below) instead of from physical-intelligence/libero alone.

TRAIN.json: dict(name, base_checkpoint, datasets=[dict(repo_id, root, share)], batch_sizes, steps, peak_lr, warmup_steps,
ema_decay (or null), save_interval, seed, num_workers).
"""
import dataclasses
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ASSET_ID = 'physical-intelligence/libero'
PROBE_STEPS = 6
OUT_OF_MEMORY = ('RESOURCE_EXHAUSTED', 'Out of memory', 'out of memory')


def read(path):
    with open(path) as stream:
        return json.load(stream)


def write(path, data):
    temporary = Path(str(path) + '.tmp'); temporary.write_text(json.dumps(data, indent=1)); temporary.replace(path)


def pick(index, sizes, shares, seed):
    """(dataset, item) for one training example: the dataset with probability shares[k], the item uniform within it."""
    rng = np.random.default_rng([seed, index])
    k = int(np.searchsorted(np.cumsum(shares), rng.random(), side='right'))
    k = min(k, len(sizes) - 1)
    return k, int(rng.integers(sizes[k]))


class Mix:
    """A map-style dataset of `length` examples, each drawn from one of `parts` (see pick)."""
    def __init__(self, parts, shares, length, seed):
        if len(parts) != len(shares) or abs(sum(shares) - 1) > 1e-9 or min(shares) <= 0:
            raise ValueError('Shares must be positive and sum to one')
        self.parts, self.shares, self.length, self.seed = parts, list(shares), length, seed
        self.sizes = [len(p) for p in parts]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        k, item = pick(index, self.sizes, self.shares, self.seed)
        return self.parts[k][item]


def settings_ok(s):
    need = {'name', 'base_checkpoint', 'datasets', 'batch_sizes', 'steps', 'peak_lr', 'warmup_steps', 'ema_decay', 'save_interval', 'seed',
            'num_workers'}
    if set(s) != need or not s['datasets'] or abs(sum(d['share'] for d in s['datasets']) - 1) > 1e-9 or not s['batch_sizes']:
        raise ValueError('Invalid TRAIN.json')
    return s


def out_of_memory(text):
    return any(mark in text for mark in OUT_OF_MEMORY)


def probe(root):
    """The first batch size of TRAIN.json that trains PROBE_STEPS steps; any failure other than out of memory stops."""
    root = Path(root); s = settings_ok(read(root/'TRAIN.json')); trials = []
    for size in s['batch_sizes']:
        started = time.monotonic(); log = root/('probe_%d.log' % size)
        with log.open('w') as stream:
            code = subprocess.run([sys.executable, '-u', __file__, str(root), '--trial', str(size)], stdout=stream, stderr=subprocess.STDOUT).returncode
        text = log.read_text(errors='replace')
        trials.append(dict(batch_size=size, passed=code == 0, out_of_memory=code != 0 and out_of_memory(text), seconds=time.monotonic() - started))
        if code == 0 or not trials[-1]['out_of_memory']:
            break
    write(root/'PROBE.json', dict(passed=trials[-1]['passed'], batch_size=trials[-1]['batch_size'] if trials[-1]['passed'] else None,
                                  trials=trials, steps=PROBE_STEPS, slurm_job=os.environ.get('SLURM_JOB_ID')))
    if not trials[-1]['passed']:
        raise SystemExit('No batch size trained: %s' % trials)
    print(json.dumps(trials), flush=True)


def main(root, trial=None, data_only=False):
    root = Path(root); s = settings_ok(read(root/'TRAIN.json')); started = time.monotonic()
    batch = s['batch_sizes'][-1] if data_only else trial if trial is not None else read(root/'PROBE.json')['batch_size']
    steps = PROBE_STEPS if trial is not None or data_only else s['steps']
    openpi = Path(os.environ['CB_UPSTREAM'])/'openpi'
    sys.path.insert(0, str(openpi/'scripts'))
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader
    import openpi.training.optimizer as _optimizer
    import openpi.training.weight_loaders as _weight_loaders
    import openpi.transforms as _transforms
    import openpi.training.checkpoints as _checkpoints
    import train as openpi_train

    base = Path(s['base_checkpoint'])
    if not (base/'params').exists() or not (base/'assets'/ASSET_ID).exists():
        raise RuntimeError('Base checkpoint needs params and the LIBERO normalization statistics')
    original = _config.get_config('pi05_libero')
    config = dataclasses.replace(
        original, name='pi05_libero', exp_name=s['name'] if trial is None else 'probe_%d' % trial,
        data=dataclasses.replace(original.data, assets=_config.AssetsConfig(assets_dir=str(base/'assets'), asset_id=ASSET_ID)),
        weight_loader=_weight_loaders.CheckpointWeightLoader(str(base/'params')),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=s['warmup_steps'], peak_lr=s['peak_lr'], decay_steps=max(steps, s['warmup_steps'] + 1),
                                                   decay_lr=s['peak_lr'] / 10),
        ema_decay=s['ema_decay'], batch_size=batch, num_train_steps=steps, save_interval=s['save_interval'],
        keep_period=s['save_interval'], log_interval=20 if trial is None else 1, seed=s['seed'], num_workers=s['num_workers'], wandb_enabled=False,
        checkpoint_base_dir=str(root/'checkpoints'), assets_base_dir=str(root/'assets'), overwrite=False, resume=False)

    def mixed(data_config, action_horizon, model_config):
        parts = []
        for d in s['datasets']:
            meta = lerobot_dataset.LeRobotDatasetMetadata(d['repo_id'], root=d['root'])
            data = lerobot_dataset.LeRobotDataset(
                d['repo_id'], root=d['root'],
                delta_timestamps={key: [t / meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys})
            parts.append(_data_loader.TransformedDataset(data, [_transforms.PromptFromLeRobotTask(meta.tasks)]))
            print('dataset', d['repo_id'], 'frames', len(data), 'episodes', meta.total_episodes, 'share', d['share'], flush=True)
        return Mix(parts, [d['share'] for d in s['datasets']], steps * batch, s['seed'])

    logging.basicConfig(level=logging.INFO)   # openpi's main reformats the root logger's first handler: make sure there is one
    if not data_ok(config, _data_loader):
        raise RuntimeError('The data config did not load the base checkpoint\'s normalization statistics')
    _data_loader.create_torch_dataset = mixed
    if data_only:
        observation, actions = next(iter(_data_loader.create_data_loader(config, shuffle=True, num_batches=1)))
        print('batch', {k: tuple(v.shape) for k, v in observation.images.items()}, 'state', tuple(observation.state.shape),
              'prompt tokens', tuple(observation.tokenized_prompt.shape), 'actions', tuple(actions.shape),
              'action range', float(np.asarray(actions).min()), float(np.asarray(actions).max()), flush=True)
        return
    if trial is not None:
        _checkpoints.save_state = lambda *args, **kwargs: None   # a probe saves nothing
        openpi_train.main(config); shutil.rmtree(config.checkpoint_dir, ignore_errors=True)
        print('probe passed: batch %d, %.1f s for %d steps with loading and compilation' % (batch, time.monotonic() - started, steps), flush=True)
        return
    openpi_train.main(config)
    folder = Path(config.checkpoint_dir); saved = sorted(int(p.name) for p in folder.iterdir() if p.name.isdigit())
    for step in saved:   # the optimizer state is several times the weights and is not needed again
        shutil.rmtree(folder/str(step)/'train_state', ignore_errors=True)
    write(root/'TRAINED.json', dict(passed=True, settings=s, batch_size=batch, checkpoints={str(k): str(folder/str(k)) for k in saved},
                                    final=str(folder/str(saved[-1])), elapsed_seconds=time.monotonic() - started,
                                    code_commit=os.environ.get('CB_CODE_COMMIT'), slurm_job=os.environ.get('SLURM_JOB_ID')))


def data_ok(config, _data_loader):
    data_config = config.data.create(config.assets_dirs, config.model)
    return data_config.norm_stats is not None and data_config.prompt_from_task and data_config.repo_id == ASSET_ID


if __name__ == '__main__':
    arguments = sys.argv[1:]
    if len(arguments) == 1:
        main(arguments[0])
    elif len(arguments) == 2 and arguments[1] == '--data':
        main(arguments[0], data_only=True)
    elif len(arguments) == 2 and arguments[1] == '--probe':
        probe(arguments[0])
    elif len(arguments) == 3 and arguments[1] == '--trial':
        main(arguments[0], int(arguments[2]))
    else:
        raise SystemExit(__doc__)
