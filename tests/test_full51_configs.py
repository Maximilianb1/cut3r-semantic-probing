from __future__ import annotations

from pathlib import Path

import yaml

from src.data.co3d import CO3DV2_231130_CATEGORIES


def _load_config(name: str) -> dict:
    path = Path("configs/stage0") / name
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_full51_execution_parts_are_disjoint_and_complete() -> None:
    part_a = _load_config("full51-part-a.yaml")
    part_b = _load_config("full51-part-b.yaml")
    categories_a = set(part_a["dataset"]["categories"])
    categories_b = set(part_b["dataset"]["categories"])

    assert len(categories_a) == part_a["dataset"]["expected_category_count"] == 26
    assert len(categories_b) == part_b["dataset"]["expected_category_count"] == 25
    assert categories_a.isdisjoint(categories_b)
    assert categories_a | categories_b == set(CO3DV2_231130_CATEGORIES)


def test_full51_execution_parts_share_the_scientific_contract() -> None:
    part_a = _load_config("full51-part-a.yaml")
    part_b = _load_config("full51-part-b.yaml")
    expected_caps = {"train": 30, "val": 5, "test": 5}

    for config in (part_a, part_b):
        assert config["sampling"]["sequence_caps"] == expected_caps
        assert config["sampling"]["window_length"] == 6
        assert config["sampling"]["windows_per_sequence"] == 4
        assert config["sampling"]["seed"] == 20260718
        assert config["preprocessing"] == {
            "input_size": 512,
            "patch_size": 16,
            "square_ok": False,
            "mask_threshold": 0.5,
        }

    assert part_a["cache"]["directory"] != part_b["cache"]["directory"]
    assert part_a["output"]["manifest_dir"] != part_b["output"]["manifest_dir"]
