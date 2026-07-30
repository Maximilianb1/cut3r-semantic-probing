from __future__ import annotations

from dataclasses import dataclass

import torch

REPRESENTATION_CONTRACT_VERSION = "cut3r-trajectory-v1"


@dataclass
class FeatureTrajectory:
    """CUT3R features saved after every timestep in one ordered window."""

    image_tokens: torch.Tensor
    state_tokens: torch.Tensor
    frame_ids: list[str]
    token_grid: tuple[int, int]

    def validate(self, expected_timesteps: int = 6) -> None:
        if self.image_tokens.ndim != 4:
            raise ValueError(
                "image_tokens must have shape [T, B, image_tokens, channels], "
                f"got {tuple(self.image_tokens.shape)}"
            )
        if self.state_tokens.ndim != 4:
            raise ValueError(
                "state_tokens must have shape [T, B, state_tokens, channels], "
                f"got {tuple(self.state_tokens.shape)}"
            )
        if self.image_tokens.shape[0] != expected_timesteps:
            raise ValueError("Unexpected number of image-token timesteps")
        if self.state_tokens.shape[0] != expected_timesteps:
            raise ValueError("Unexpected number of state-token timesteps")
        if len(self.frame_ids) != expected_timesteps:
            raise ValueError("Frame ID count does not match trajectory timesteps")
        if len(set(self.frame_ids)) != len(self.frame_ids):
            raise ValueError("A trajectory contains duplicate frame IDs")
        if self.image_tokens.shape[1] != self.state_tokens.shape[1]:
            raise ValueError("Image and state feature batch sizes differ")
        if self.image_tokens.shape[1] < 1:
            raise ValueError("Feature trajectory batch dimension is empty")
        if self.image_tokens.shape[-1] != self.state_tokens.shape[-1]:
            raise ValueError("Image and state feature dimensions differ")
        if self.image_tokens.shape[-1] < 1 or self.state_tokens.shape[-2] < 1:
            raise ValueError("Feature or persistent-state dimensions are empty")
        if len(self.token_grid) != 2 or any(value < 1 for value in self.token_grid):
            raise ValueError("Token grid dimensions must be positive")
        expected_image_tokens = self.token_grid[0] * self.token_grid[1]
        if self.image_tokens.shape[-2] != expected_image_tokens:
            raise ValueError(
                f"Image token count {self.image_tokens.shape[-2]} does not match "
                f"token grid {self.token_grid}"
            )
        if not torch.isfinite(self.image_tokens).all():
            raise ValueError("image_tokens contain NaN or infinity")
        if not torch.isfinite(self.state_tokens).all():
            raise ValueError("state_tokens contain NaN or infinity")

    def cpu_float16(self) -> FeatureTrajectory:
        return FeatureTrajectory(
            image_tokens=self.image_tokens.detach()
            .to(device="cpu", dtype=torch.float16)
            .contiguous(),
            state_tokens=self.state_tokens.detach()
            .to(device="cpu", dtype=torch.float16)
            .contiguous(),
            frame_ids=list(self.frame_ids),
            token_grid=self.token_grid,
        )
