"""Content-addressed branch artifacts and verifiable manifests."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class BlobRef:
    sha256: str
    size_bytes: int
    relative_path: str


def _stream_sha256(handle: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


class ContentAddressedStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.blob_root = self.root / "blobs" / "sha256"
        self.blob_root.mkdir(parents=True, exist_ok=True)

    def _path(self, sha256: str) -> Path:
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ValueError("invalid SHA-256")
        return self.blob_root / sha256[:2] / sha256[2:]

    def put_bytes(self, payload: bytes) -> BlobRef:
        digest = hashlib.sha256(payload).hexdigest()
        destination = self._path(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            ref = BlobRef(digest, len(payload), destination.relative_to(self.root).as_posix())
            self.validate(ref)
            return ref
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{digest}.", suffix=".pending", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
        ref = BlobRef(digest, len(payload), destination.relative_to(self.root).as_posix())
        self.validate(ref)
        return ref

    def put_file(self, source: str | Path) -> BlobRef:
        path = Path(source)
        with path.open("rb") as handle:
            digest, size = _stream_sha256(handle)
        destination = self._path(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{digest}.", suffix=".pending", dir=destination.parent
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as target, path.open("rb") as source_handle:
                    for chunk in iter(lambda: source_handle.read(8 * 1024 * 1024), b""):
                        target.write(chunk)
                    target.flush()
                    os.fsync(target.fileno())
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    pass
            finally:
                temporary.unlink(missing_ok=True)
        ref = BlobRef(digest, size, destination.relative_to(self.root).as_posix())
        self.validate(ref)
        return ref

    def validate(self, ref: BlobRef) -> None:
        expected = self._path(ref.sha256)
        actual = (self.root / ref.relative_path).resolve()
        if actual != expected.resolve():
            raise ValueError("blob relative path does not match its content address")
        if not expected.is_file():
            raise FileNotFoundError(f"content-addressed blob missing: {ref.sha256}")
        with expected.open("rb") as handle:
            digest, size = _stream_sha256(handle)
        if digest != ref.sha256 or size != ref.size_bytes:
            raise ValueError(f"content-addressed blob integrity mismatch: {ref.sha256}")


def pack_numeric_mapping(mapping: Mapping[str, Any]) -> bytes:
    """Serialize numeric arrays deterministically without pickle or ZIP timestamps."""

    metadata = []
    chunks = []
    offset = 0
    for key in sorted(mapping):
        array = np.ascontiguousarray(np.asarray(mapping[key]))
        if array.dtype.hasobject:
            raise TypeError(f"numeric mapping field contains object dtype: {key}")
        raw = array.tobytes()
        metadata.append(
            {
                "key": key,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "offset": offset,
                "nbytes": len(raw),
            }
        )
        chunks.append(raw)
        offset += len(raw)
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    return b"CBNM1" + struct.pack(">Q", len(header)) + header + b"".join(chunks)


def unpack_numeric_mapping(payload: bytes) -> dict[str, np.ndarray]:
    if not payload.startswith(b"CBNM1") or len(payload) < 13:
        raise ValueError("not a CrashBench numeric mapping blob")
    header_size = struct.unpack(">Q", payload[5:13])[0]
    header_end = 13 + header_size
    if header_end > len(payload):
        raise ValueError("numeric mapping header is truncated")
    metadata = json.loads(payload[13:header_end])
    raw = memoryview(payload)[header_end:]
    result = {}
    for row in metadata:
        start = int(row["offset"])
        end = start + int(row["nbytes"])
        if start < 0 or end > len(raw):
            raise ValueError(f"numeric mapping field is truncated: {row['key']}")
        array = np.frombuffer(raw[start:end], dtype=np.dtype(row["dtype"])).copy()
        array = array.reshape(tuple(row["shape"]))
        result[str(row["key"])] = array
    return result


def seal_artifact_manifest(
    output: str | Path,
    *,
    kind: str,
    blobs: Mapping[str, BlobRef],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(output)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact manifest: {path}")
    if any(name in {"manifest", "self"} for name in blobs):
        raise ValueError("artifact manifest cannot recursively pin itself")
    base: dict[str, Any] = {
        "schema_version": 1,
        "kind": kind,
        "blobs": {name: asdict(ref) for name, ref in sorted(blobs.items())},
        "metadata": dict(metadata),
        "self_pinning": "manifest_sha256 hashes canonical payload without this field",
    }
    base["manifest_sha256"] = hashlib.sha256(
        json.dumps(base, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, (json.dumps(base, indent=2, sort_keys=True, default=str) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return base


def verify_artifact_manifest(
    manifest: Mapping[str, Any], store: ContentAddressedStore
) -> None:
    payload = dict(manifest)
    declared = payload.pop("manifest_sha256", None)
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    if declared != actual:
        raise ValueError("artifact manifest self hash mismatch")
    for row in manifest.get("blobs", {}).values():
        store.validate(BlobRef(**row))
