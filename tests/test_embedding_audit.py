from __future__ import annotations

from pathlib import Path

import torch

from src.embeddings.audit import compare_to_reference, load_audit_reference
from src.embeddings.cache import FeatureCacheWriter
from src.embeddings.types import FeatureTrajectory


def _trajectory() -> FeatureTrajectory:
    return FeatureTrajectory(
        image_tokens=torch.arange(6 * 1 * 4 * 3, dtype=torch.float32).reshape(
            6, 1, 4, 3
        ),
        state_tokens=torch.arange(6 * 1 * 2 * 3, dtype=torch.float32).reshape(
            6, 1, 2, 3
        ),
        frame_ids=[f"apple/sequence:{index}" for index in range(6)],
        token_grid=(2, 2),
    )


def test_exported_audit_reference_round_trips(tmp_path: Path) -> None:
    from src.embeddings.audit import export_audit_reference

    cache = tmp_path / "cache"
    hashes = {
        frame_id: {"image": "1" * 64, "mask": "2" * 64}
        for frame_id in _trajectory().frame_ids
    }
    with FeatureCacheWriter(
        cache,
        contract={"source_file_hashes": "sha256"},
        windows_per_shard=1,
    ) as writer:
        writer.add("window-a", _trajectory(), source_sha256=hashes)
    output = tmp_path / "reference"
    metadata = export_audit_reference(cache, output)
    loaded_metadata, tensors = load_audit_reference(output)
    assert loaded_metadata == metadata
    assert torch.equal(tensors["image_tokens"], _trajectory().image_tokens.half())
    assert torch.equal(tensors["state_tokens"], _trajectory().state_tokens.half())


def test_reference_comparison_requires_exact_float16_values() -> None:
    trajectory = _trajectory().cpu_float16()
    reference = {
        "image_tokens": trajectory.image_tokens,
        "state_tokens": trajectory.state_tokens,
    }
    result = compare_to_reference(
        reference,
        image_tokens=trajectory.image_tokens.float(),
        state_tokens=trajectory.state_tokens.float(),
    )
    assert result["exactly_equal"] is True
    changed = trajectory.image_tokens.float().clone()
    changed[0, 0, 0, 0] += 1
    try:
        compare_to_reference(
            reference,
            image_tokens=changed,
            state_tokens=trajectory.state_tokens,
        )
    except ValueError as error:
        assert "differs" in str(error)
    else:
        raise AssertionError("Changed features must fail exact comparison")
