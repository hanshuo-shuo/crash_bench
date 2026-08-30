#!/usr/bin/env python3
"""Create a deterministic content manifest for an external checkpoint tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_git_head(repo: Path) -> str:
    """Resolve HEAD without requiring a git executable on compute nodes."""

    git_path = repo.resolve() / ".git"
    if git_path.is_file():
        pointer = git_path.read_text().strip()
        if not pointer.startswith("gitdir: "):
            raise ValueError(f"invalid gitdir file: {git_path}")
        target = Path(pointer.removeprefix("gitdir: "))
        git_path = target if target.is_absolute() else (git_path.parent / target).resolve()
    head = (git_path / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    loose = git_path / ref
    if loose.is_file():
        return loose.read_text().strip()
    packed = git_path / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            commit, name = line.split(" ", 1)
            if name == ref:
                return commit
    raise ValueError(f"cannot resolve Git HEAD ref {ref!r} in {git_path}")


def tree_manifest(root: Path, label: str) -> dict:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"tree root is not a directory: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink inside identity tree is not allowed: {path}")
        if not path.is_file():
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not files:
        raise ValueError("identity tree contains no files")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "kind": "crashbench_external_tree_content_manifest",
        "label": label,
        "root": str(root),
        "file_count": len(files),
        "total_size_bytes": sum(row["size_bytes"] for row in files),
        "tree_manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "crashbench_git_commit": resolve_git_head(Path.cwd()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-git-head", type=Path)
    args = parser.parse_args()
    if args.print_git_head is not None:
        print(resolve_git_head(args.print_git_head))
        return
    if args.root is None or args.label is None or args.output is None:
        parser.error("--root, --label, and --output are required unless --print-git-head is used")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite identity manifest: {args.output}")
    payload = tree_manifest(args.root, args.label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    summary = {
        key: payload[key]
        for key in ("file_count", "total_size_bytes", "tree_manifest_sha256")
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
