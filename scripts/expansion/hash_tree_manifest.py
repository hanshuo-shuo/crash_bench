#!/usr/bin/env python3
"""Create a deterministic content manifest for an external checkpoint tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "crashbench_git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
