"""Tests for the synthetic probe-cache generator's argument handling.

The generator is a smoke-test fixture, but bad arguments used to fail far from their
cause: ``windows_per_sequence=0`` as a ZeroDivisionError, ``feature_dim=0`` not at all
(it wrote a cache of zero-width features), and ``empty_fraction`` above 0.5 silently
made every window empty.
"""

from __future__ import annotations

import pytest

from scripts.make_synthetic_probe_cache import _parse_grids, _problems, build
from src.backbones.probe_cache import load_probe_index, load_target_tokens

_SANE = {
    "windows": 100,
    "grids": [(8, 10)],
    "feature_dim": 768,
    "windows_per_sequence": 4,
    "categories": 5,
    "noise": 1.5,
    "empty_fraction": 0.05,
}


def test_sane_arguments_have_no_problems() -> None:
    assert _problems(**_SANE) == []


@pytest.mark.parametrize(
    "override,expected",
    [
        ({"windows": 0}, "--windows"),
        ({"windows_per_sequence": 0}, "--windows-per-sequence"),
        ({"categories": 0}, "--categories"),
        ({"categories": 99}, "--categories"),
        ({"feature_dim": 0}, "--feature-dim"),
        ({"noise": -1.0}, "--noise"),
        ({"empty_fraction": 1.5}, "--empty-fraction"),
        ({"empty_fraction": -0.2}, "--empty-fraction"),
        ({"grids": [(1, 10)]}, "too small"),
    ],
)
def test_unusable_arguments_are_named(override, expected) -> None:
    found = _problems(**{**_SANE, **override})
    assert found, f"{override} should be rejected"
    assert any(expected in problem for problem in found), found


def test_build_reports_every_problem_at_once(tmp_path) -> None:
    with pytest.raises(ValueError) as error:
        build(tmp_path / "cache", layout="target_only", seed=1,
              **{**_SANE, "feature_dim": 0, "windows_per_sequence": 0})
    message = str(error.value)
    assert "--feature-dim" in message and "--windows-per-sequence" in message


@pytest.mark.parametrize("text", ["8-10", "8x", "x10", "eightxten", ""])
def test_grid_parsing_rejects_malformed_text(text) -> None:
    with pytest.raises(ValueError, match="expected HxW|At least one grid"):
        _parse_grids(text)


def test_grid_parsing_accepts_several_grids() -> None:
    assert _parse_grids("8x10, 6x8") == [(8, 10), (6, 8)]


@pytest.mark.parametrize("fraction,expected", [(0.0, 0), (0.25, 5), (0.6, 12), (1.0, 20)])
def test_empty_fraction_is_exact(tmp_path, fraction, expected) -> None:
    """The share of foreground-free windows must match what was asked for.

    The old rule picked "every int(1/fraction)-th window", so 0.6 became every
    window - a fixture that looked fine and taught the probe nothing.
    """
    cache = tmp_path / f"cache-{fraction}"
    build(cache, layout="target_only", seed=1,
          **{**_SANE, "windows": 20, "grids": [(4, 5)], "feature_dim": 8,
             "empty_fraction": fraction})
    rows = load_probe_index(cache)
    empty = sum(1 for row in rows if float(load_target_tokens(cache, row)[1].sum()) == 0.0)
    assert empty == expected
