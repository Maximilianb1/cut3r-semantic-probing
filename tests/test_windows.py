from __future__ import annotations

from src.data.windows import (
    choose_sequences,
    evenly_spaced_indices,
    generate_ordered_windows,
)


def _frames(count: int) -> list[dict[str, object]]:
    return [
        {
            "frame_id": f"frame-{number}",
            "frame_number": number,
            "category": "ball",
            "sequence_id": "sequence-a",
            "split": "train",
        }
        for number in range(count)
    ]


def test_evenly_spaced_indices_cover_endpoints_without_duplicates() -> None:
    assert evenly_spaced_indices(20, 6) == [0, 3, 7, 11, 15, 19]


def test_windows_are_deterministic_ordered_and_disjoint() -> None:
    first = generate_ordered_windows(
        _frames(30), window_length=6, windows_per_sequence=4
    )
    second = generate_ordered_windows(
        list(reversed(_frames(30))), window_length=6, windows_per_sequence=4
    )
    assert first == second
    assert len(first) == 4
    used = [frame_id for window in first for frame_id in window["frame_ids"]]
    assert len(used) == len(set(used)) == 24
    assert all(window["target_frame_id"] == window["frame_ids"][-1] for window in first)
    assert all(
        window["frame_numbers"] == sorted(window["frame_numbers"]) for window in first
    )


def test_short_sequence_yields_no_window() -> None:
    assert generate_ordered_windows(_frames(5)) == []


def test_sequence_cap_is_seeded_and_independent_of_input_order() -> None:
    ids = [f"sequence-{index}" for index in range(20)]
    selected = choose_sequences(ids, cap=5, seed=11, category="ball", split="train")
    assert selected == choose_sequences(
        list(reversed(ids)), cap=5, seed=11, category="ball", split="train"
    )
    assert selected != choose_sequences(
        ids, cap=5, seed=12, category="ball", split="train"
    )
