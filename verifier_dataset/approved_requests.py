"""Fill only the 32 payloads explicitly approved by the human on 2026-10-14.

No GPU worker or cache miss can extend this allowlist. Originals stay immutable.
An unknown charge stops the worker with its durable reservation still held.
"""
import argparse
import base64
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from .budget import DatasetBudget
from .inventory import ASSETS, read, sha, write
from .launch import command
from scripts.openrouter_perception import fill, read_response, request_hash


AUTH_PATH = Path(__file__).parent/'authorizations/approved_perception_32.json'
AUTH_SHA256 = '397126d61aba759454ec896841407c2f2475738228f716b7d011dfbba70065cf'
STAGE = 'fill-approved-perception-32'


def authorization(path=AUTH_PATH):
    if sha(path) != AUTH_SHA256:
        raise RuntimeError('Approved allowlist changed')
    value = read(path)
    if (len(value['rows']) != 32 or len({r['request_identity'] for r in value['rows']}) != 32
            or len({r['source_id'] for r in value['rows']}) != 32):
        raise RuntimeError('Exactly 32 distinct reviewed payloads and sources required')
    return value


def validate_payload(directory, row):
    if (directory.name != row['request_identity'] or
            sha(directory/'request.json') != row['request_file_sha256'] or
            sha(directory/'input.png') != row['image_sha256']):
        raise RuntimeError('Payload differs from the human-reviewed file hashes')
    request = read(directory/'request.json')
    if request_hash(request) != row['request_identity']:
        raise RuntimeError('Canonical request identity differs')
    contents = request['messages'][0]['content']
    if (len(request['messages']) != 1 or request['messages'][0]['role'] != 'user'
            or len(contents) != 2 or contents[0]['type'] != 'image_url'
            or contents[1]['type'] != 'text'):
        raise RuntimeError('Reviewed payload shape changed')
    url = contents[0]['image_url']['url']
    if not url.startswith('data:image/png;base64,'):
        raise RuntimeError('Only the reviewed inline simulator PNG is allowed')
    pixels = base64.b64decode(url.split(',', 1)[1], validate=True)
    if hashlib.sha256(pixels).hexdigest() != row['image_sha256']:
        raise RuntimeError('Embedded image is not the reviewed PNG')
    return request


class ApprovedBudget:
    """Enforce a campaign-wide allowlist scope across restarts and new roots."""
    def __init__(self, budget, auth):
        self.budget = budget
        self.prefix = 'approved32:'+AUTH_SHA256+':'
        self.allowed = {r['request_identity'] for r in auth['rows']}

    def scoped_calls(self):
        return {k: v for k, v in self.budget.state['calls'].items() if k.startswith(self.prefix)}

    def reserve(self, key, request):
        identity = request_hash(request)
        if identity not in self.allowed or key not in (identity, identity+':attempt2'):
            raise RuntimeError('Unapproved payload or third attempt')
        calls = self.scoped_calls()
        if self.prefix+key in calls:
            raise RuntimeError('This authorized attempt is already charged or reserved')
        if any(v['status'] != 'settled' for v in self.budget.state['calls'].values()):
            raise RuntimeError('Unresolved prior charge requires reconciliation; no automatic retry')
        if key.endswith(':attempt2') and self.prefix+identity not in calls:
            raise RuntimeError('Second attempt lacks a settled first attempt')
        # Count full reservations even after cheap settlements: this is a ceiling
        # on authorized attempts, independently of the shared campaign spend cap.
        if len(calls) >= 64 or Decimal(len(calls)+1)*Decimal('.10') > Decimal('6.40'):
            raise RuntimeError('Approved $6.40 / 64-attempt boundary reached')
        self.budget.reserve(self.prefix+key, request)

    def settle(self, key, response):
        self.budget.settle(self.prefix+key, response)


def validate_response(directory, budget):
    text = read_response(directory)
    record = read(directory/'response.json')['response']
    if record.get('provider') != 'Z.AI' or record['choices'][0].get('finish_reason') == 'length':
        raise RuntimeError('Unexpected cached provider or truncated response')
    entries = [v for k, v in budget.scoped_calls().items()
               if k in (budget.prefix+directory.name, budget.prefix+directory.name+':attempt2')]
    if not any(v.get('status') == 'settled' and v.get('generation_id') == record.get('id')
               for v in entries):
        raise RuntimeError('Response lacks matching settled campaign accounting')
    return text


def work(root):
    budget = None
    try:
        auth = authorization(root/'AUTHORIZATION.json')
        receipt = read(root/'RECEIPT.json')
        expected_budget = ASSETS/'verifier_dataset/budget_fd1000_20261014'
        if (receipt.get('stage') != STAGE or receipt.get('authorization_sha256') != AUTH_SHA256
                or Path(receipt['budget_root']) != expected_budget
                or Path(__file__).resolve().parents[1] != (root/'source').resolve()):
            raise RuntimeError('Wrong execution stage, frozen source or shared ledger')
        # Verify every payload before reading the private key or sending any data.
        expected = {r['request_identity'] for r in auth['rows']}
        if {p.name for p in (root/'requests').iterdir()} != expected:
            raise RuntimeError('Execution root contains extra or missing requests')
        for row in auth['rows']:
            validate_payload(root/'requests'/row['request_identity'], row)
        budget = DatasetBudget(expected_budget)
        scoped = ApprovedBudget(budget, auth)
        budget.refresh_prices()
        write(root/'COST_ESTIMATE.json', dict(logical_requests=32, maximum_attempts=64,
            reservation_ceiling_usd='6.40', campaign_limit_usd='20.00',
            already_committed_usd=str(budget.committed()), pricing_checked=True,
            unknown_charge_behavior='Stop with reservation retained; no automatic retries'))
        write(root/'WORKER_READY.json', dict(authorization_sha256=AUTH_SHA256,
            key_source='Existing private CrashBench OpenRouter credential; never logged',
            allowed_payloads=32, model='z-ai/glm-4.5v', provider='Z.AI'))
        complete = []
        for row in auth['rows']:
            if (root/'STOP.json').exists():
                raise RuntimeError('Execution stop marker present')
            directory = root/'requests'/row['request_identity']
            validate_payload(directory, row)
            if not (directory/'response.json').exists():
                # Transport errors may already be billable. A held reservation
                # blocks further calls; never retry an unknown HTTP charge.
                fill(directory, budget=scoped, attempt=1)
            validate_response(directory, scoped)
            complete.append(dict(source_id=row['source_id'], request_identity=directory.name,
                                 response_sha256=sha(directory/'response.json')))
            write(root/'API_STATUS.json', dict(completed=len(complete), expected=32,
                campaign_committed_usd=str(budget.committed()), rows=complete))
        write(root/'API_COMPLETE.json', dict(passed=True, logical_requests=32, rows=complete,
            campaign_committed_usd=str(budget.committed()),
            scoped_calls=scoped.scoped_calls(), authorization_sha256=AUTH_SHA256,
            labels_created=False, accepted_dataset_rows=0, training_allowed=False))
    except BaseException as error:
        # Never serialize request headers, keys or potentially credential-bearing
        # network exception text. The class and failed stage are enough to triage.
        write(root/'STOP.json', dict(stage=STAGE, error=type(error).__name__,
            reason='Approved request worker stopped; inspect saved hashes and ledger before resuming'))
        raise SystemExit('Approved request worker stopped: '+type(error).__name__)
    finally:
        if budget:
            calls = ApprovedBudget(budget, auth).scoped_calls()
            write(root/'API_FINAL.json', dict(campaign_committed_usd=str(budget.committed()),
                scoped_attempts=len(calls), scoped_calls=calls, campaign_limit_usd='20.00',
                scoped_reservation_ceiling_usd='6.40'))
            budget.close()


def launch():
    repo = Path.cwd().resolve()
    if repo != (Path.home()/'crash_bench').resolve():
        raise RuntimeError('Wrong project checkout')
    if command('git', 'status', '--porcelain', '--untracked-files=all'):
        raise RuntimeError('Dirty source')
    if command('git', 'remote', 'get-url', 'origin') not in (
            'git@github.com:hanshuo-shuo/crash_bench.git', 'https://github.com/hanshuo-shuo/crash_bench.git'):
        raise RuntimeError('Wrong repository')
    commit = command('git', 'rev-parse', 'HEAD')
    if 'origin/' not in command('git', 'branch', '-r', '--contains', commit):
        raise RuntimeError('Unpublished source')
    auth = authorization()
    for row in auth['rows']:
        validate_payload(Path(row['request_directory']), row)
    root = ASSETS/'verifier_dataset'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+STAGE+'_'+commit[:12])
    root.mkdir(parents=True, exist_ok=False)
    (root/'source').mkdir()
    with (root/'source.tar').open('wb') as stream:
        subprocess.run(['git', 'archive', commit], stdout=stream, check=True)
    subprocess.run(['tar', '-xf', str(root/'source.tar'), '-C', str(root/'source')], check=True)
    (root/'SOURCE_COMMIT').write_text(commit+'\n')
    (root/'source.sha256').write_text(sha(root/'source.tar')+'  '+str(root/'source.tar')+'\n')
    shutil.copyfile(AUTH_PATH, root/'AUTHORIZATION.json')
    for row in auth['rows']:
        dest = root/'requests'/row['request_identity']; dest.mkdir(parents=True)
        for filename in ('request.json', 'input.png'):
            shutil.copyfile(Path(row['request_directory'])/filename, dest/filename)
        validate_payload(dest, row)
    receipt = dict(stage=STAGE, run_root=str(root), code_commit=commit, job=None,
        execution_host=os.uname().nodename, execution='network-only login-node worker',
        budget_root=str(ASSETS/'verifier_dataset/budget_fd1000_20261014'),
        authorization_sha256=AUTH_SHA256, max_logical_requests=32, max_paid_attempts=64,
        maximum_reserved_usd='6.40', campaign_limit_usd='20.00',
        seed=None, simulator_or_policy_execution=False, source_archives_immutable=True)
    write(root/'RECEIPT.json', receipt)
    local = repo/'results/verifier_dataset'/root.name; local.mkdir(parents=True, exist_ok=False)
    write(local/'RECEIPT.json', receipt)
    worker = shlex.join(['/projects/p33100/siosio/envs/openpi/bin/python', '-u', '-m',
                        'verifier_dataset.approved_requests', 'work', str(root)])
    worker += ' > '+shlex.quote(str(root/'api_worker.log'))+' 2>&1'
    exists = subprocess.run(['tmux', 'has-session', '-t', '=crashbench'], capture_output=True).returncode == 0
    tmux = (['tmux', 'new-window', '-d', '-t', 'crashbench', '-n', 'fd1000_approved32'] if exists
            else ['tmux', 'new-session', '-d', '-s', 'crashbench', '-n', 'fd1000_approved32'])
    subprocess.run(tmux+['-c', str(root/'source'), worker], check=True)
    print(json.dumps(receipt), flush=True)
    deadline = time.monotonic()+50
    while not (root/'WORKER_READY.json').exists():
        if (root/'STOP.json').exists() or time.monotonic() > deadline:
            raise RuntimeError('Worker has not reported readiness; inspect this root before any new launch')
        time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('launch', 'work'))
    parser.add_argument('root', type=Path, nargs='?')
    args = parser.parse_args()
    if args.mode == 'launch':
        launch()
    elif args.root is not None:
        work(args.root)
    else:
        parser.error('A frozen execution root is required')
