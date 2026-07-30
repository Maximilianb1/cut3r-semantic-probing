from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible data deterministically for IDs and hashes."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def stable_id(prefix: str, value: Any, length: int = 20) -> str:
    digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()[:length]
    return f"{prefix}-{digest}"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return expand_environment(value)


def expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "$" in expanded:
            raise ValueError(
                f"Unresolved environment variable in configuration: {value}"
            )
        return expanded
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if isinstance(value, Mapping):
        return {key: expand_environment(item) for key, item in value.items()}
    return value


def reject_unknown_keys(
    mapping: Mapping[str, Any], allowed: Iterable[str], context: str
) -> None:
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {context} keys: {', '.join(unknown)}")
