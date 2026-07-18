from __future__ import annotations

import hmac
import string
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.dictconfig import DictConfig
from omegaconf.nodes import AnyNode

from src.common.io import sha256_file

CHECKPOINT_LOAD_POLICY = "pytorch-weights-only-omegaconf-v1"
TRUSTED_CHECKPOINT_GLOBALS = (
    dict,
    defaultdict,
    ContainerMetadata,
    Metadata,
    DictConfig,
    AnyNode,
    Any,
)
TRUSTED_CHECKPOINT_GLOBAL_NAMES = tuple(
    sorted(
        [
            "builtins.dict",
            "collections.defaultdict",
            "omegaconf.base.ContainerMetadata",
            "omegaconf.base.Metadata",
            "omegaconf.dictconfig.DictConfig",
            "omegaconf.nodes.AnyNode",
            "typing.Any",
        ]
    )
)


def validate_cut3r_checkpoint(
    checkpoint: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    path = Path(checkpoint).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CUT3R checkpoint does not exist: {path}")
    expected = expected_sha256.lower()
    if len(expected) != 64 or any(
        character not in string.hexdigits for character in expected
    ):
        raise ValueError(
            "Expected checkpoint SHA-256 must contain 64 hexadecimal digits"
        )
    actual = sha256_file(path)
    if not hmac.compare_digest(actual, expected):
        raise RuntimeError(
            f"CUT3R checkpoint SHA-256 mismatch: expected {expected}, got {actual}"
        )
    unsafe_globals = tuple(
        sorted(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    )
    if unsafe_globals != TRUSTED_CHECKPOINT_GLOBAL_NAMES:
        raise RuntimeError(
            "CUT3R checkpoint contains globals outside the audited allowlist: "
            f"{list(unsafe_globals)}"
        )
    return {
        "filename": path.name,
        "sha256": actual,
        "load_policy": CHECKPOINT_LOAD_POLICY,
        "allowlisted_globals": list(TRUSTED_CHECKPOINT_GLOBAL_NAMES),
    }


@contextmanager
def trusted_cut3r_checkpoint(
    checkpoint: str | Path, *, expected_sha256: str
) -> Iterator[dict[str, Any]]:
    provenance = validate_cut3r_checkpoint(checkpoint, expected_sha256=expected_sha256)
    with torch.serialization.safe_globals(list(TRUSTED_CHECKPOINT_GLOBALS)):
        yield provenance
