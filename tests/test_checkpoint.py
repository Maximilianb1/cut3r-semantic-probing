from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from src.common.io import sha256_file
from src.embeddings.checkpoint import (
    CHECKPOINT_LOAD_POLICY,
    TRUSTED_CHECKPOINT_GLOBAL_NAMES,
    TRUSTED_CHECKPOINT_GLOBALS,
    trusted_cut3r_checkpoint,
    validate_cut3r_checkpoint,
)


class UnexpectedCheckpointObject:
    pass


def _checkpoint(path: Path, *, unexpected: bool = False) -> str:
    payload = {
        "args": OmegaConf.create({"model": "fixture"}),
        "model": {},
    }
    if unexpected:
        payload["unexpected"] = UnexpectedCheckpointObject()
    torch.save(payload, path)
    return sha256_file(path)


def test_trusted_checkpoint_is_hash_bound_and_allowlist_is_scoped(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    digest = _checkpoint(checkpoint)
    safe_globals_before = list(torch.serialization.get_safe_globals())

    with trusted_cut3r_checkpoint(checkpoint, expected_sha256=digest) as provenance:
        loaded = torch.load(checkpoint)

    assert isinstance(loaded["args"], DictConfig)
    assert provenance == {
        "filename": checkpoint.name,
        "sha256": digest,
        "load_policy": CHECKPOINT_LOAD_POLICY,
        "allowlisted_globals": list(TRUSTED_CHECKPOINT_GLOBAL_NAMES),
    }
    safe_global_ids_before = {id(item) for item in safe_globals_before}
    safe_globals_after = torch.serialization.get_safe_globals()
    for trusted_global in TRUSTED_CHECKPOINT_GLOBALS:
        if id(trusted_global) not in safe_global_ids_before:
            assert all(item is not trusted_global for item in safe_globals_after)


def test_checkpoint_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    _checkpoint(checkpoint)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        validate_cut3r_checkpoint(checkpoint, expected_sha256="0" * 64)


def test_checkpoint_with_unexpected_global_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    digest = _checkpoint(checkpoint, unexpected=True)

    with pytest.raises(RuntimeError, match="outside the audited allowlist"):
        validate_cut3r_checkpoint(checkpoint, expected_sha256=digest)
