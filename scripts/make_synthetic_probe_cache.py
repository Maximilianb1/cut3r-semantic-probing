"""Write a SYNTHETIC probe-feature cache, for smoke-testing the probe pipeline.

This produces fake embeddings and fake masks in the real cache format, so
``src.segmentation`` can be run end to end without CO3D, CUT3R weights, or a GPU.
It is a plumbing test, not an experiment: **no number produced from this cache
means anything about the research question.** The cache's ``metadata.json`` is
stamped ``synthetic: true`` so a run's ``metrics.json`` carries that marker too.

The data is deliberately *learnable but not separable*. Each window gets a blob of
foreground tokens; a token's feature vector is a fixed random direction ``w``
scaled by +1 (foreground) or -1 (background), plus Gaussian noise. A linear probe
should therefore land well above chance and well below perfect, which exercises the
IoU code on non-degenerate values. Some windows are deliberately foreground-free,
to exercise the empty-target IoU convention.

Each category also gets its own direction, added to both the grid tokens and the state
latent, so the SAME cache serves Stage 1 and Stage 2: segmentation reads the foreground
direction per token, classification reads the category direction from the pooled
window. In 768 dimensions random directions are near-orthogonal, so neither task's
signal masks the other's.

Splits are assigned at CO3D **sequence** level (as the real manifests do), so the
leakage guards see a valid cache. Run from the repository root::

    python -m scripts.make_synthetic_probe_cache --cache-dir "$CUT3R_CACHE_ROOT/probe/synthetic"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from src.backbones.base import TARGET_FRAME_INDEX, WINDOW_LENGTH
from src.backbones.probe_cache import (
    TARGET_ONLY,
    TRAJECTORY,
    EmbeddingCacheWriter,
    EmbeddingSample,
    category_index_map,
    category_vocabulary,
    verify_probe_cache,
)


def _blob_labels(grid: tuple[int, int], generator: torch.Generator, *, empty: bool) -> torch.Tensor:
    """A rectangular foreground blob of random size/position on the token grid."""
    grid_h, grid_w = grid
    labels = torch.zeros(grid_h, grid_w)
    if empty:
        return labels
    height = int(torch.randint(2, max(3, grid_h - 1), (1,), generator=generator).item())
    width = int(torch.randint(2, max(3, grid_w - 1), (1,), generator=generator).item())
    top = int(torch.randint(0, grid_h - height + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, grid_w - width + 1, (1,), generator=generator).item())
    labels[top : top + height, left : left + width] = 1.0
    return labels


def _tokens(labels: torch.Tensor, direction: torch.Tensor, noise: float,
            generator: torch.Generator) -> torch.Tensor:
    """[N, D] features: +direction for foreground, -direction for background, plus noise."""
    flat = labels.reshape(-1)
    signs = (flat * 2.0 - 1.0).unsqueeze(-1)  # 1 -> +1, 0 -> -1
    clean = signs * direction.unsqueeze(0)
    return clean + noise * torch.randn(flat.numel(), direction.numel(), generator=generator)


def _parse_grids(text: str) -> list[tuple[int, int]]:
    """"8x10,6x8" -> [(8, 10), (6, 8)]. Several grids mirror CO3D's mixed aspect ratios."""
    grids = []
    for part in text.split(","):
        height, separator, width = part.strip().lower().partition("x")
        if not separator or not height.isdigit() or not width.isdigit():
            raise ValueError(
                f"Bad grid {part.strip()!r}: expected HxW, comma-separated (e.g. '8x10,6x8')"
            )
        grids.append((int(height), int(width)))
    if not grids:
        raise ValueError("At least one grid is required")
    return grids


_MIN_GRID_SIDE = 2  # _blob_labels needs room to place a blob


def _problems(*, windows: int, grids: list[tuple[int, int]], feature_dim: int,
              windows_per_sequence: int, categories: int, noise: float,
              empty_fraction: float, category_signal: float = 1.0) -> list[str]:
    """Every unusable argument, described. Empty list means the settings are sane.

    Shared by :func:`build` and the CLI so both reject the same values: the CLI turns
    these into ``parser.error`` (exit 2 with usage), an importing caller gets a
    ``ValueError``. Without this, several values fail far from their cause -
    ``windows_per_sequence=0`` as a ZeroDivisionError, ``feature_dim=0`` not at all
    (it silently writes a cache of zero-width features).
    """
    vocabulary_size = len(category_vocabulary())
    found = []
    if windows < 1:
        found.append(f"--windows must be at least 1, got {windows}")
    if windows_per_sequence < 1:
        found.append(f"--windows-per-sequence must be at least 1, got {windows_per_sequence}")
    if not 1 <= categories <= vocabulary_size:
        found.append(f"--categories must be between 1 and {vocabulary_size}, got {categories}")
    if feature_dim < 1:
        found.append(f"--feature-dim must be at least 1, got {feature_dim}")
    if noise < 0:
        found.append(f"--noise must not be negative, got {noise}")
    if not 0.0 <= empty_fraction <= 1.0:
        found.append(f"--empty-fraction must be between 0 and 1, got {empty_fraction}")
    if category_signal < 0:
        found.append(f"--category-signal must not be negative, got {category_signal}")
    for grid in grids:
        if min(grid) < _MIN_GRID_SIDE:
            found.append(
                f"grid {grid[0]}x{grid[1]} is too small; each side must be at least "
                f"{_MIN_GRID_SIDE}"
            )
    return found


def _unit_directions(count: int, feature_dim: int, generator: torch.Generator) -> torch.Tensor:
    """``[count, feature_dim]`` unit vectors. In 768-d, random draws are near-orthogonal,
    so a per-category direction barely interferes with the foreground direction."""
    directions = torch.randn(count, feature_dim, generator=generator)
    return directions / directions.norm(dim=1, keepdim=True)


def build(cache_dir: Path, *, windows: int, layout: str, grids: list[tuple[int, int]],
          feature_dim: int, windows_per_sequence: int, categories: int, noise: float,
          empty_fraction: float, seed: int, category_signal: float = 1.0) -> dict[str, Any]:
    found = _problems(windows=windows, grids=grids, feature_dim=feature_dim,
                      windows_per_sequence=windows_per_sequence, categories=categories,
                      noise=noise, empty_fraction=empty_fraction,
                      category_signal=category_signal)
    if found:
        raise ValueError("; ".join(found))
    generator = torch.Generator().manual_seed(seed)
    direction = _unit_directions(1, feature_dim, generator)[0]
    # One direction per category, so the SAME cache trains both stages: segmentation
    # reads the foreground direction per token, classification reads the category
    # direction from the pooled window. They are near-orthogonal in 768-d, so neither
    # task's signal masks the other's.
    category_directions = _unit_directions(categories, feature_dim, generator)
    vocabulary = category_vocabulary()[:categories]
    index_of = category_index_map()

    sequences = -(-windows // windows_per_sequence)  # ceil
    # Sequence-level split, like the real manifests: 60 / 20 / 20 percent.
    train_end, val_end = int(sequences * 0.6), int(sequences * 0.8)
    # Pick the empty windows by exact count, spread evenly. An "every 1/fraction-th"
    # rule rounds badly: int(1/0.6) is 1, which would make *every* window empty.
    empty_count = round(windows * empty_fraction)
    empty_indices = {round(position * windows / empty_count) for position in range(empty_count)}

    contract = {
        "layout": layout,
        "backbone": {"backbone": "synthetic", "weights": "none",
                     "status": "SYNTHETIC FIXTURE: not a real backbone"},
        "mask_threshold": 0.5,
        "synthetic": True,
        "synthetic_note": "Fake embeddings and masks for pipeline smoke tests. "
                          "No result from this cache is scientifically meaningful.",
        "synthetic_generator": {"seed": seed, "noise": noise,
                                "category_signal": category_signal,
                                "categories": categories,
                                "grids": [list(g) for g in grids],
                                "feature_dim": feature_dim},
    }

    written = 0
    with EmbeddingCacheWriter(cache_dir, contract=contract, windows_per_shard=32) as writer:
        for index in range(windows):
            sequence = index // windows_per_sequence
            split = "train" if sequence < train_end else ("val" if sequence < val_end else "test")
            category = vocabulary[sequence % len(vocabulary)]
            # Grid varies window to window, as it does in the real caches (the crop
            # depends on the frame's aspect ratio). This is what exercises the
            # variable-token-count collation.
            grid = grids[index % len(grids)]
            labels = _blob_labels(grid, generator, empty=index in empty_indices)
            category_direction = category_directions[vocabulary.index(category)]
            spatial = _tokens(labels, direction, noise, generator)
            spatial = spatial + category_signal * category_direction.unsqueeze(0)
            # The latent carries the category signal too, so the Stage 2 comparison
            # (pooled grid tokens vs pooled state) can actually be run on this fixture.
            latent_tokens = 16
            latent = (category_signal * category_direction.expand(latent_tokens, feature_dim)
                      + noise * torch.randn(latent_tokens, feature_dim, generator=generator))
            if layout == TRAJECTORY:
                image = torch.randn(WINDOW_LENGTH, 1, grid[0] * grid[1], feature_dim,
                                    generator=generator)
                image[TARGET_FRAME_INDEX, 0] = spatial  # only the target frame is learnable
                state = torch.randn(WINDOW_LENGTH, 1, latent_tokens, feature_dim,
                                    generator=generator)
                state[TARGET_FRAME_INDEX, 0] = latent
                extra = {"image_tokens": image, "state_tokens": state}
            else:
                extra = {"spatial_tokens": spatial, "global_tokens": latent}
            writer.add(
                EmbeddingSample(
                    window_id=f"synthetic-{index:05d}",
                    category=category,
                    category_index=index_of[category],
                    sequence_id=f"synthetic-seq-{sequence:04d}",
                    split=split,
                    layout=layout,
                    token_grid=grid,
                    frame_ids=[f"synthetic-{index:05d}/f{f}" for f in range(WINDOW_LENGTH)],
                    seg_labels=labels,
                    **extra,
                ),
                source_sha256={},  # no real source files exist
            )
            written += 1
    return {"written": written, "cache_dir": str(cache_dir),
            "verification": verify_probe_cache(cache_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--layout", choices=(TARGET_ONLY, TRAJECTORY), default=TARGET_ONLY)
    parser.add_argument("--grids", default="8x10,6x8",
                        help="comma-separated token grids, cycled per window (mixed aspect ratios)")
    parser.add_argument("--feature-dim", type=int, default=768, help="768 matches the real configs")
    parser.add_argument("--windows-per-sequence", type=int, default=4)
    parser.add_argument("--categories", type=int, default=5)
    parser.add_argument("--noise", type=float, default=1.5, help="higher -> harder task")
    parser.add_argument("--empty-fraction", type=float, default=0.05,
                        help="share of foreground-free windows")
    parser.add_argument("--category-signal", type=float, default=1.0,
                        help="strength of the per-category direction (0 disables the "
                             "classification signal; segmentation is unaffected)")
    parser.add_argument("--seed", type=int, default=20260730)
    arguments = parser.parse_args()
    try:
        grids = _parse_grids(arguments.grids)
    except ValueError as error:
        parser.error(str(error))
    # Validate before building so misuse exits with usage, not a traceback from deep
    # inside the generator (or, worse, a silently useless cache).
    found = _problems(
        windows=arguments.windows, grids=grids, feature_dim=arguments.feature_dim,
        windows_per_sequence=arguments.windows_per_sequence, categories=arguments.categories,
        noise=arguments.noise, empty_fraction=arguments.empty_fraction,
        category_signal=arguments.category_signal,
    )
    if found:
        parser.error("; ".join(found))
    result = build(
        arguments.cache_dir, windows=arguments.windows, layout=arguments.layout,
        grids=grids, feature_dim=arguments.feature_dim,
        windows_per_sequence=arguments.windows_per_sequence, categories=arguments.categories,
        noise=arguments.noise, empty_fraction=arguments.empty_fraction, seed=arguments.seed,
        category_signal=arguments.category_signal,
    )
    print(f"written={result['written']} cache_dir={result['cache_dir']}")
    print(f"verification={result['verification']}")
    print("SYNTHETIC cache: for pipeline smoke tests only, results are not meaningful.")


if __name__ == "__main__":
    main()
