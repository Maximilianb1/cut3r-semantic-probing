"""CUT3R backbone (trained or randomly initialized weights).

This wraps the exact Stage 0 extraction path -- :func:`load_cut3r_model` plus
:class:`Cut3rFeatureExtractor` on ``prepare_image_window`` views -- and exposes
its target-frame outputs through the shared :class:`Backbone` contract:

- ``spatial_tokens`` <- ``image_tokens[5, 0]`` (state-conditioned per-view
  decoder tokens; ADR 0003), shape ``[grid_h * grid_w, 768]``.
- ``global_tokens`` <- ``state_tokens[5, 0]`` (persistent recurrent memory),
  shape ``[768, 768]``.

``weights="random"`` produces the random-initialized CUT3R baseline. NOTE: the
scientific meaning of that baseline is an OPEN decision. The re-initialization here is a
documented, seeded *placeholder* -- it reuses the released architecture and
resets every module that exposes ``reset_parameters`` under a fixed seed -- and
must not be treated as a settled contract until an ADR accepts it.
"""

from __future__ import annotations

from typing import Any

import torch

from src.backbones.base import (
    TARGET_FRAME_INDEX,
    WINDOW_LENGTH,
    Backbone,
    BackboneFeatures,
    TrajectoryExtraction,
    WindowExtraction,
)
from src.embeddings.cut3r_adapter import Cut3rFeatureExtractor
from src.embeddings.extract import load_cut3r_model
from src.embeddings.input import move_views_to_device, prepare_image_window

_RANDOM_INIT_STRATEGIES = {"reset_parameters"}


def _reinitialize_module(module: torch.nn.Module) -> None:
    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()


def randomize_weights(model: torch.nn.Module, *, seed: int, strategy: str) -> None:
    """Provisional random re-initialization of a CUT3R model in place.

    Kept deliberately explicit and seedable. The strategy is a config field so a
    future ADR can change it without a silent behavior shift.
    """
    if strategy not in _RANDOM_INIT_STRATEGIES:
        raise ValueError(
            f"Unknown random-init strategy {strategy!r}; "
            f"supported: {sorted(_RANDOM_INIT_STRATEGIES)}"
        )
    generator_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        model.apply(_reinitialize_module)
    finally:
        torch.random.set_rng_state(generator_state)


class Cut3rBackbone(Backbone):
    """Frozen CUT3R feature extractor behind the shared backbone contract."""

    def __init__(
        self,
        *,
        cut3r_root: str,
        checkpoint: str,
        checkpoint_sha256: str,
        expected_commit: str,
        weights: str = "trained",
        compatibility_patch: str | None = None,
        input_size: int = 512,
        patch_size: int = 16,
        square_ok: bool = False,
        device: str = "cuda",
        random_init: dict[str, Any] | None = None,
    ) -> None:
        if weights not in ("trained", "random"):
            raise ValueError("CUT3R weights must be 'trained' or 'random'")
        self.name = f"cut3r-{weights}"
        self.weights = weights
        self._input_size = int(input_size)
        self._patch_size = int(patch_size)
        self._square_ok = bool(square_ok)
        self._device = torch.device(device)
        self._checkpoint_sha256 = checkpoint_sha256
        self._expected_commit = expected_commit
        self._random_init = dict(random_init or {"strategy": "reset_parameters", "seed": 0})

        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUT3R backbone requested CUDA but it is unavailable")

        model, self._checkpoint_provenance = load_cut3r_model(
            cut3r_root,
            checkpoint,
            self._device,
            expected_checkpoint_sha256=checkpoint_sha256,
        )
        if weights == "random":
            randomize_weights(
                model,
                seed=int(self._random_init["seed"]),
                strategy=str(self._random_init["strategy"]),
            )
        self._extractor = Cut3rFeatureExtractor(model)

    def _run(self, frame_rows: list[dict[str, Any]], dataset_root: str):
        if len(frame_rows) != WINDOW_LENGTH:
            raise ValueError(f"CUT3R backbone requires {WINDOW_LENGTH} frame rows")
        views, masks, token_grid = prepare_image_window(
            frame_rows,
            dataset_root=dataset_root,
            input_size=self._input_size,
            patch_size=self._patch_size,
            square_ok=self._square_ok,
        )
        move_views_to_device(views, self._device)
        trajectory = self._extractor.extract(
            views,
            frame_ids=[row["frame_id"] for row in frame_rows],
            token_grid=token_grid,
        )
        # masks[t] is [1, 1, H, W] uint8 at the CUT3R post-transform resolution.
        target_mask = masks[TARGET_FRAME_INDEX][0, 0].to(dtype=torch.float32).cpu()
        return trajectory, token_grid, target_mask

    def extract_window(
        self,
        frame_rows: list[dict[str, Any]],
        *,
        dataset_root: str,
    ) -> WindowExtraction:
        trajectory, token_grid, target_mask = self._run(frame_rows, dataset_root)
        features = BackboneFeatures(
            spatial_tokens=trajectory.image_tokens[TARGET_FRAME_INDEX, 0].float().cpu().contiguous(),
            token_grid=token_grid,
            global_tokens=trajectory.state_tokens[TARGET_FRAME_INDEX, 0].float().cpu().contiguous(),
            frame_id=frame_rows[TARGET_FRAME_INDEX]["frame_id"],
        )
        extraction = WindowExtraction(features=features, target_mask=target_mask)
        extraction.validate()
        return extraction

    def extract_trajectory(
        self,
        frame_rows: list[dict[str, Any]],
        *,
        dataset_root: str,
    ) -> TrajectoryExtraction:
        """Full six-state trajectory (grid + latent per state) + target mask.

        Used to cache CUT3R-trained with the same structure as Stage 0.
        """
        trajectory, token_grid, target_mask = self._run(frame_rows, dataset_root)
        extraction = TrajectoryExtraction(
            image_tokens=trajectory.image_tokens.float().cpu().contiguous(),
            state_tokens=trajectory.state_tokens.float().cpu().contiguous(),
            token_grid=token_grid,
            frame_ids=list(trajectory.frame_ids),
            target_mask=target_mask,
        )
        extraction.validate()
        return extraction

    def provenance(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "backbone": self.name,
            "weights": self.weights,
            "input_size": self._input_size,
            "patch_size": self._patch_size,
            "square_ok": self._square_ok,
            "checkpoint_sha256": self._checkpoint_sha256,
            "expected_commit": self._expected_commit,
            "checkpoint_provenance": self._checkpoint_provenance,
        }
        if self.weights == "random":
            record["random_init"] = dict(self._random_init)
            record["random_init_status"] = "PROVISIONAL: pending baseline ADR"
        return record
