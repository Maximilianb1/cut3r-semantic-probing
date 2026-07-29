from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from src.common.io import stable_id

SAMPLER_VERSION = "uniform-disjoint-v1"


def deterministic_rank(seed: int, category: str, split: str, sequence_id: str) -> str:
    payload = f"{seed}\0{category}\0{split}\0{sequence_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def choose_sequences(
    sequence_ids: Sequence[str],
    *,
    cap: int | None,
    seed: int,
    category: str,
    split: str,
) -> list[str]:
    """Choose a stable, seed-controlled subset without depending on filesystem order."""
    unique = sorted(set(sequence_ids))
    ranked = sorted(
        unique, key=lambda item: deterministic_rank(seed, category, split, item)
    )
    if cap is None:
        return ranked
    if cap < 1:
        raise ValueError("Sequence cap must be positive or null")
    return ranked[:cap]


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count < 1 or length < count:
        raise ValueError(
            f"Need length >= count >= 1, got length={length}, count={count}"
        )
    if count == 1:
        return [length // 2]
    indices = [(index * (length - 1)) // (count - 1) for index in range(count)]
    if len(set(indices)) != count:
        raise AssertionError("Evenly spaced index construction produced duplicates")
    return indices


def generate_ordered_windows(
    frames: Sequence[dict[str, Any]],
    *,
    window_length: int = 6,
    windows_per_sequence: int = 4,
) -> list[dict[str, Any]]:
    """Create non-overlapping, temporally ordered windows spread across a sequence.

    Frames are sorted by annotated frame number. We select
    ``min(windows_per_sequence, floor(n_frames / window_length)) * window_length``
    evenly spaced frames across the entire sequence, then split them into consecutive
    groups. This is deterministic, avoids duplicated frames between a sequence's
    windows, and makes the final frame in each group the supervised target.
    """
    if window_length < 2:
        raise ValueError("Window length must be at least two")
    if windows_per_sequence < 1:
        raise ValueError("windows_per_sequence must be positive")
    ordered = sorted(
        frames, key=lambda row: (int(row["frame_number"]), row["frame_id"])
    )
    frame_numbers = [int(row["frame_number"]) for row in ordered]
    if len(frame_numbers) != len(set(frame_numbers)):
        raise ValueError("A sequence contains duplicate annotated frame numbers")
    effective_windows = min(windows_per_sequence, len(ordered) // window_length)
    if effective_windows == 0:
        return []
    selected_count = effective_windows * window_length
    selected = [
        ordered[index] for index in evenly_spaced_indices(len(ordered), selected_count)
    ]
    windows: list[dict[str, Any]] = []
    for number in range(effective_windows):
        group = selected[number * window_length : (number + 1) * window_length]
        identity = {
            "sampler_version": SAMPLER_VERSION,
            "category": group[0]["category"],
            "sequence_id": group[0]["sequence_id"],
            "split": group[0]["split"],
            "frame_ids": [row["frame_id"] for row in group],
        }
        windows.append(
            {
                "window_id": stable_id("window", identity),
                "category": group[0]["category"],
                "sequence_id": group[0]["sequence_id"],
                "split": group[0]["split"],
                "frame_ids": identity["frame_ids"],
                "frame_numbers": [int(row["frame_number"]) for row in group],
                "target_frame_id": group[-1]["frame_id"],
                "target_timestep": window_length,
                "window_length": window_length,
                "sampler_version": SAMPLER_VERSION,
                "target_rgb_visible": True,
            }
        )
    return windows
