#!/usr/bin/env python3
"""Materialize a compact, content-pinned deployable feature cache from D5 blobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.branching.artifacts import BlobRef, ContentAddressedStore, unpack_numeric_mapping
from crashbench.models.option_outcome import pooled_anchor_features


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite feature cache: {args.output_dir}")
    anchors = read_jsonl(args.anchors)
    if not anchors or any("test" in str(row["split_role"]) for row in anchors):
        raise ValueError("feature cache requires non-test D5 anchors")
    refs = {}
    for row in anchors:
        ref = BlobRef(**row["anchor_feature_blob"])
        prior = refs.setdefault(ref.sha256, ref)
        if prior != ref:
            raise ValueError("same feature hash has inconsistent blob identity")
    store = ContentAddressedStore(args.artifact_store)
    hashes, features = [], []
    for digest, ref in sorted(refs.items()):
        store.validate(ref)
        payload = (store.root / ref.relative_path).read_bytes()
        hashes.append(digest)
        features.append(pooled_anchor_features(unpack_numeric_mapping(payload)))
    widths = {len(row) for row in features}
    if len(widths) != 1:
        raise ValueError("materialized feature width drift")
    args.output_dir.mkdir(parents=True)
    cache = args.output_dir / "features.npz"
    np.savez_compressed(
        cache,
        sha256=np.asarray(hashes, dtype="U64"),
        features=np.asarray(features, dtype=np.float32),
    )
    manifest = {
        "schema_version": 1,
        "kind": "crashbench_expansion_d5_materialized_feature_cache",
        "feature_blob_count": len(hashes),
        "feature_width": next(iter(widths)),
        "anchors_sha256": sha256_file(args.anchors),
        "features_npz_sha256": sha256_file(cache),
        "split_roles": sorted(set(row["split_role"] for row in anchors)),
        "test_rows_read": 0,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output = args.output_dir / "manifest.json"
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"output": str(args.output_dir), "blobs": len(hashes), "width": next(iter(widths))}, sort_keys=True))


if __name__ == "__main__":
    main()
