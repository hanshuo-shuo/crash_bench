#!/usr/bin/env bash
# Edit locally, publish an exact Git commit, fast-forward Quest to it, and run Slurm there.
#
# The SSH master connection is intentionally opened by the user so password/Duo
# authentication never needs to be stored in this repository:
#   ssh -M -S /tmp/quest.sock -o ControlPersist=8h \
#     -fN quest.northwestern.edu

set -euo pipefail

QUEST_HOST="${QUEST_HOST:-quest.northwestern.edu}"
QUEST_REMOTE_DIR="${QUEST_REMOTE_DIR:-crash_bench}"
QUEST_SOCKET="${QUEST_SOCKET:-/tmp/quest.sock}"
EXPECTED_GITHUB_REPO="hanshuo-shuo/crash_bench"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_OPTS=(-S "$QUEST_SOCKET" -o BatchMode=yes -o ConnectTimeout=10)
SSH_TRANSPORT="ssh -S $QUEST_SOCKET -o BatchMode=yes -o ConnectTimeout=10"

usage() {
  cat <<'EOF'
Usage: scripts/quest_sync.sh COMMAND [ARG]

Commands:
  check                  Verify the SSH connection, project identity, and Slurm.
  status                 Show local and remote Git status.
  dry-run                Preview the exact commit/files Quest would fast-forward to.
  push                   Fast-forward a clean Quest checkout to published local HEAD.
  submit FILE.sbatch     Push, then submit FILE with sbatch on Quest.
  queue                  Show this user's Slurm jobs.
  exec 'COMMAND'         Run a shell command from the remote project root.
  pull-result PATH       Pull one file or directory under results/ (no delete).
  pull-packed-result PATH
                         Pull one precompressed .tar.gz/.tar.xz/.tar.zst under results/
                         without redundant transport compression (no delete).

Environment overrides:
  QUEST_HOST, QUEST_REMOTE_DIR, QUEST_SOCKET

push never rsyncs the project tree. It requires local HEAD to be published to its
upstream and advances a clean Quest checkout through Git. Ignored envs/, third_party/,
videos, logs, model caches, and activation dumps are not touched.
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

origin_matches_project() {
  case "$1" in
    "https://github.com/$EXPECTED_GITHUB_REPO"|\
    "https://github.com/$EXPECTED_GITHUB_REPO.git"|\
    "git@github.com:$EXPECTED_GITHUB_REPO"|\
    "git@github.com:$EXPECTED_GITHUB_REPO.git"|\
    "ssh://git@github.com/$EXPECTED_GITHUB_REPO"|\
    "ssh://git@github.com/$EXPECTED_GITHUB_REPO.git")
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

remote_in_project() {
  local command="$1"
  ssh "${SSH_OPTS[@]}" "$QUEST_HOST" \
    "cd \"\$HOME/$QUEST_REMOTE_DIR\" && $command"
}

check_project_identity() {
  local local_origin
  local remote_home
  local remote_origin
  local remote_pwd
  local slurm_version

  local_origin="$(git -C "$ROOT" remote get-url origin)"
  origin_matches_project "$local_origin" \
    || die "local origin is not $EXPECTED_GITHUB_REPO: $local_origin"

  remote_home="$(ssh "${SSH_OPTS[@]}" "$QUEST_HOST" 'cd "$HOME" && pwd -P')"
  remote_pwd="$(remote_in_project 'pwd -P')"
  [[ "$remote_pwd" == "$remote_home/$QUEST_REMOTE_DIR" ]] \
    || die "unexpected Quest project path: $remote_pwd"

  remote_origin="$(remote_in_project 'git remote get-url origin')"
  origin_matches_project "$remote_origin" \
    || die "Quest origin is not $EXPECTED_GITHUB_REPO: $remote_origin"

  slurm_version="$(remote_in_project 'command -v sbatch >/dev/null && sbatch --version | head -n 1')" \
    || die "sbatch is unavailable on Quest"

  printf 'Verified CrashBench: local=%s quest=%s origin=%s slurm=%s\n' \
    "$ROOT" "$remote_pwd" "$remote_origin" "$slurm_version"
}

validate_sbatch_file() {
  local file="$1"
  local path="$ROOT/$file"

  [[ "$file" == setup/*.sbatch && "$file" != *..* ]] \
    || die "submit expects a path like setup/job.sbatch"
  [[ -f "$path" ]] || die "local file not found: $file"
  bash -n "$path" || die "sbatch script has invalid Bash syntax: $file"
  grep -Eq '^#SBATCH[[:space:]]+--account(=|[[:space:]]+)p33100([[:space:]]|$)' "$path" \
    || die "$file must declare #SBATCH --account=p33100"
  grep -Eq '^#SBATCH[[:space:]]+--partition(=|[[:space:]]+)(gengpu|short)([[:space:]]|$)' "$path" \
    || die "$file must declare #SBATCH --partition=gengpu or short"
}

sync_push() {
  local dry_run="${1:-false}"
  local branch
  local upstream
  local upstream_head
  local local_head
  local remote_head
  local remote_status
  local quoted_branch
  local quoted_head

  [[ -z "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)" ]] \
    || die "local worktree is dirty; commit the exact source before Quest sync"
  branch="$(git -C "$ROOT" symbolic-ref --quiet --short HEAD)" \
    || die "local checkout is detached; use a published branch"
  upstream="$(git -C "$ROOT" rev-parse --abbrev-ref '@{upstream}' 2>/dev/null)" \
    || die "branch $branch has no upstream; run git push -u origin HEAD"
  local_head="$(git -C "$ROOT" rev-parse HEAD)"
  upstream_head="$(git -C "$ROOT" rev-parse '@{upstream}')"
  [[ "$local_head" == "$upstream_head" ]] \
    || die "local HEAD is not published to $upstream; run git push first"

  remote_head="$(remote_in_project 'git rev-parse HEAD')"
  remote_status="$(remote_in_project 'git status --porcelain=v1 --untracked-files=all')"
  [[ -z "$remote_status" ]] || {
    printf '%s\n' "$remote_status" >&2
    die "Quest worktree is dirty; preserve/reconcile it before exact-commit sync"
  }
  git -C "$ROOT" cat-file -e "$remote_head^{commit}" 2>/dev/null \
    || die "remote commit $remote_head is not available locally; pull/fetch first"
  git -C "$ROOT" merge-base --is-ancestor "$remote_head" HEAD \
    || die "Quest history is not an ancestor of local HEAD; reconcile Git before pushing"

  if [[ "$remote_head" == "$local_head" ]]; then
    printf 'Already in sync at %s.\n' "$local_head"
    return 0
  fi

  printf 'Quest %s -> local/published %s (%s)\n' "$remote_head" "$local_head" "$branch"
  git -C "$ROOT" --no-pager diff --name-status "$remote_head" "$local_head" --
  [[ "$dry_run" == true ]] && return 0

  quoted_branch="$(printf '%q' "$branch")"
  quoted_head="$(printf '%q' "$local_head")"
  remote_in_project \
    "git fetch --quiet origin $quoted_branch && git merge --ff-only $quoted_head && test \"\$(git rev-parse HEAD)\" = $quoted_head && test -z \"\$(git status --porcelain=v1 --untracked-files=all)\""
  printf 'Quest now clean at %s.\n' "$local_head"
}

command="${1:-}"
case "$command" in
  check)
    check_connection
    check_project_identity
    ;;
  status)
    check_connection
    check_project_identity
    printf '%s\n' 'LOCAL'
    git -C "$ROOT" status --short --branch
    printf '%s\n' 'QUEST'
    remote_in_project 'git status --short --branch'
    ;;
  dry-run)
    check_connection
    check_project_identity
    sync_push true
    ;;
  push)
    check_connection
    check_project_identity
    sync_push false
    ;;
  submit)
    file="${2:-}"
    validate_sbatch_file "$file"
    check_connection
    check_project_identity
    sync_push false
    quoted_file="$(printf '%q' "$file")"
    remote_in_project "sbatch --parsable $quoted_file"
    ;;
  queue)
    check_connection
    check_project_identity
    ssh "${SSH_OPTS[@]}" "$QUEST_HOST" 'squeue -u "$USER"'
    ;;
  exec)
    [[ -n "${2:-}" ]] || die "exec expects one quoted command"
    check_connection
    check_project_identity
    remote_in_project "$2"
    ;;
  pull-result)
    path="${2:-}"
    [[ "$path" =~ ^results/[A-Za-z0-9._/-]+$ && "$path" != */ ]] \
      || die "pull-result only accepts a path below results/"
    check_connection
    check_project_identity
    quoted_path="$(printf '%q' "$path")"
    if remote_in_project "test -d $quoted_path"; then
      mkdir -p "$ROOT/$path"
      rsync -az --itemize-changes -e "$SSH_TRANSPORT" \
        "$QUEST_HOST:$QUEST_REMOTE_DIR/$path/" "$ROOT/$path/"
    else
      mkdir -p "$ROOT/$(dirname "$path")"
      rsync -az --itemize-changes -e "$SSH_TRANSPORT" \
        "$QUEST_HOST:$QUEST_REMOTE_DIR/$path" "$ROOT/$path"
    fi
    ;;
  pull-packed-result)
    path="${2:-}"
    [[ "$path" =~ ^results/[A-Za-z0-9._/-]+\.(tar\.gz|tar\.xz|tar\.zst)$ && "$path" != */ ]] \
      || die "pull-packed-result only accepts a .tar.gz/.tar.xz/.tar.zst file below results/"
    check_connection
    check_project_identity
    quoted_path="$(printf '%q' "$path")"
    remote_in_project "test -f $quoted_path" \
      || die "packed result is not a regular file: $path"
    mkdir -p "$ROOT/$(dirname "$path")"
    # macOS ships an older rsync without --append-verify.  The packed artifact
    # must carry an independently pulled SHA-256 sidecar, verified by the caller.
    rsync -a --partial --append --itemize-changes -e "$SSH_TRANSPORT" \
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
