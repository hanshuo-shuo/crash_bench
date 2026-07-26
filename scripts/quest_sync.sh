#!/usr/bin/env bash
# Edit locally, sync only source-controlled files to Quest, and run Slurm there.
#
# The SSH master connection is intentionally opened by the user so password/Duo
# authentication never needs to be stored in this repository:
#   ssh -M -S /tmp/crashbench-quest.sock -o ControlPersist=8h \
#     -fN quest.northwestern.edu

set -euo pipefail

QUEST_HOST="${QUEST_HOST:-quest.northwestern.edu}"
QUEST_REMOTE_DIR="${QUEST_REMOTE_DIR:-crash_bench}"
QUEST_SOCKET="${QUEST_SOCKET:-/tmp/crashbench-quest.sock}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_OPTS=(-S "$QUEST_SOCKET" -o BatchMode=yes -o ConnectTimeout=10)
SSH_TRANSPORT="ssh -S $QUEST_SOCKET -o BatchMode=yes -o ConnectTimeout=10"

usage() {
  cat <<'EOF'
Usage: scripts/quest_sync.sh COMMAND [ARG]

Commands:
  status                 Show local and remote Git status.
  dry-run                Preview files that push would upload.
  push                   Upload tracked and unignored local files (no delete).
  submit FILE.sbatch     Push, then submit FILE with sbatch on Quest.
  queue                  Show this user's Slurm jobs.
  exec 'COMMAND'         Run a shell command from the remote project root.
  pull-result PATH       Pull one path under results/ (no delete).

Environment overrides:
  QUEST_HOST, QUEST_REMOTE_DIR, QUEST_SOCKET

envs/, third_party/, ignored videos/logs/caches, and .git are never uploaded by
push because its manifest is built from Git and respects `.gitignore`.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

check_connection() {
  [[ -S "$QUEST_SOCKET" ]] || die "SSH socket not found: $QUEST_SOCKET"
  ssh "${SSH_OPTS[@]}" "$QUEST_HOST" true \
    || die "Quest connection unavailable; authenticate the SSH master session first"
}

remote_in_project() {
  local command="$1"
  ssh "${SSH_OPTS[@]}" "$QUEST_HOST" \
    "cd \"\$HOME/$QUEST_REMOTE_DIR\" && $command"
}

sync_push() {
  local dry_run="${1:-false}"
  local manifest
  local remote_head
  local -a rsync_args

  manifest="$(mktemp "${TMPDIR:-/tmp}/crashbench-sync.XXXXXX")"
  trap 'rm -f "$manifest"' RETURN

  remote_head="$(remote_in_project 'git rev-parse HEAD')"
  git -C "$ROOT" cat-file -e "$remote_head^{commit}" 2>/dev/null \
    || die "remote commit $remote_head is not available locally; pull/fetch first"
  git -C "$ROOT" merge-base --is-ancestor "$remote_head" HEAD \
    || die "Quest history is not an ancestor of local HEAD; reconcile Git before pushing"

  (
    cd "$ROOT"
    # Candidate files are local tracked changes relative to Quest's current
    # commit plus new, non-ignored files. This keeps checks fast and excludes
    # the large cloud-only environments declared in .gitignore.
    {
      git diff --name-only --diff-filter=ACMRT "$remote_head" --
      git ls-files --others --exclude-standard
    } | LC_ALL=C sort -u >"$manifest"
  )

  if [[ ! -s "$manifest" ]]; then
    printf '%s\n' 'Already in sync; no local source changes to upload.'
    return 0
  fi

  # Git checkouts on macOS and Quest have unrelated mtimes. Compare content so
  # an unchanged checkout does not cause hundreds of metadata-only updates.
  rsync_args=(-rzc --itemize-changes --files-from="$manifest")
  [[ "$dry_run" == true ]] && rsync_args+=(--dry-run)

  (
    cd "$ROOT"
    rsync "${rsync_args[@]}" -e "$SSH_TRANSPORT" \
      ./ "$QUEST_HOST:$QUEST_REMOTE_DIR/"
  )
}

command="${1:-}"
case "$command" in
  status)
    check_connection
    printf '%s\n' 'LOCAL'
    git -C "$ROOT" status --short --branch
    printf '%s\n' 'QUEST'
    remote_in_project 'git status --short --branch'
    ;;
  dry-run)
    check_connection
    sync_push true
    ;;
  push)
    check_connection
    sync_push false
    ;;
  submit)
    file="${2:-}"
    [[ "$file" == setup/*.sbatch && "$file" != *..* ]] \
      || die "submit expects a path like setup/job.sbatch"
    [[ -f "$ROOT/$file" ]] || die "local file not found: $file"
    check_connection
    sync_push false
    quoted_file="$(printf '%q' "$file")"
    remote_in_project "sbatch --parsable $quoted_file"
    ;;
  queue)
    check_connection
    ssh "${SSH_OPTS[@]}" "$QUEST_HOST" 'squeue -u "$USER"'
    ;;
  exec)
    [[ -n "${2:-}" ]] || die "exec expects one quoted command"
    check_connection
    remote_in_project "$2"
    ;;
  pull-result)
    path="${2:-}"
    [[ "$path" == results/* && "$path" != *..* ]] \
      || die "pull-result only accepts a path below results/"
    check_connection
    mkdir -p "$ROOT/$(dirname "$path")"
    rsync -az --itemize-changes -e "$SSH_TRANSPORT" \
      "$QUEST_HOST:$QUEST_REMOTE_DIR/$path" "$ROOT/$path"
    ;;
  -h|--help|help|'')
    usage
    ;;
  *)
    usage >&2
    die "unknown command: $command"
    ;;
esac
