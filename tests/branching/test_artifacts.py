from __future__ import annotations

import json

import pytest

from crashbench.branching.artifacts import (
    ContentAddressedStore,
    seal_artifact_manifest,
    verify_artifact_manifest,
    pack_numeric_mapping,
    unpack_numeric_mapping,
)
import numpy as np


def test_content_addressed_put_deduplicates_and_validates(tmp_path):
    store = ContentAddressedStore(tmp_path)
    first = store.put_bytes(b"same payload")
    second = store.put_bytes(b"same payload")
    assert first == second
    assert len(list((tmp_path / "blobs/sha256").rglob("*"))) == 2
    store.validate(first)


def test_put_file_and_corruption_detection(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    store = ContentAddressedStore(tmp_path / "store")
    ref = store.put_file(source)
    (store.root / ref.relative_path).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity mismatch"):
        store.validate(ref)


def test_manifest_self_hash_and_blob_verification(tmp_path):
    store = ContentAddressedStore(tmp_path / "store")
    ref = store.put_bytes(b"branch")
    path = tmp_path / "manifest.json"
    manifest = seal_artifact_manifest(
        path,
        kind="branch_attempt",
        blobs={"trace": ref},
        metadata={"logical_key": "abc"},
    )
    verify_artifact_manifest(manifest, store)
    tampered = json.loads(path.read_text())
    tampered["metadata"]["logical_key"] = "different"
    with pytest.raises(ValueError, match="self hash"):
        verify_artifact_manifest(tampered, store)


def test_manifest_refuses_overwrite_and_recursive_self_pin(tmp_path):
    store = ContentAddressedStore(tmp_path / "store")
    ref = store.put_bytes(b"x")
    path = tmp_path / "manifest.json"
    seal_artifact_manifest(path, kind="fixture", blobs={"data": ref}, metadata={})
    with pytest.raises(FileExistsError):
        seal_artifact_manifest(path, kind="fixture", blobs={"data": ref}, metadata={})
    with pytest.raises(ValueError, match="recursively"):
        seal_artifact_manifest(
            tmp_path / "other.json", kind="fixture", blobs={"self": ref}, metadata={}
        )


def test_numeric_mapping_pack_is_deterministic_and_roundtrips():
    mapping = {
        "image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        "state": np.array([1.5, 2.5], dtype=np.float32),
    }
    first = pack_numeric_mapping(mapping)
    second = pack_numeric_mapping(dict(reversed(list(mapping.items()))))
    assert first == second
    restored = unpack_numeric_mapping(first)
    assert set(restored) == set(mapping)
    for key in mapping:
        np.testing.assert_array_equal(restored[key], mapping[key])
