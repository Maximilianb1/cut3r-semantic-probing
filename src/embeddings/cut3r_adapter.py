from __future__ import annotations

from typing import Any

import torch

from src.embeddings.types import FeatureTrajectory


class Cut3rFeatureExtractor:
    """Read frozen recurrent features from an unmodified upstream CUT3R model.

    The implementation mirrors upstream ``ARCroco3DStereo._forward_impl`` but
    returns the final normalized image stream and committed persistent state at
    every timestep instead of running CUT3R's reconstruction head.
    """

    def __init__(self, model: Any):
        self.model = model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def extract(
        self,
        views: list[dict[str, Any]],
        *,
        frame_ids: list[str],
        token_grid: tuple[int, int],
    ) -> FeatureTrajectory:
        if len(views) != len(frame_ids):
            raise ValueError("View and frame-ID counts differ")
        if len(views) != 6:
            raise ValueError(
                "The accepted Stage 0 extraction contract requires six views"
            )
        if self.model.training:
            raise RuntimeError("CUT3R must remain in evaluation mode")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError(
                "CUT3R contains trainable parameters during frozen extraction"
            )
        with torch.inference_mode():
            trajectory = self._extract_impl(
                views, frame_ids=frame_ids, token_grid=token_grid
            )
        trajectory.validate(expected_timesteps=6)
        return trajectory

    def _extract_impl(
        self,
        views: list[dict[str, Any]],
        *,
        frame_ids: list[str],
        token_grid: tuple[int, int],
    ) -> FeatureTrajectory:
        model = self.model
        shapes, feature_layers, positions = model._encode_views(views)
        features = feature_layers[-1]
        state, state_position = model._init_state(features[0], positions[0])
        memory = model.pose_retriever.mem.expand(features[0].shape[0], -1, -1)
        initial_state = state.clone()
        initial_memory = memory.clone()
        image_trajectory: list[torch.Tensor] = []
        state_trajectory: list[torch.Tensor] = []

        for timestep, view in enumerate(views):
            feature = features[timestep]
            position = positions[timestep]
            if model.pose_head_flag:
                global_image_feature = model._get_img_level_feat(feature)
                if timestep == 0:
                    pose_feature = model.pose_token.expand(feature.shape[0], -1, -1)
                else:
                    pose_feature = model.pose_retriever.inquire(
                        global_image_feature, memory
                    )
                pose_position = -torch.ones(
                    feature.shape[0], 1, 2, device=feature.device, dtype=position.dtype
                )
            else:
                global_image_feature = None
                pose_feature = None
                pose_position = None
            new_state, decoder_images = model._recurrent_rollout(
                state,
                state_position,
                feature,
                position,
                pose_feature,
                pose_position,
                initial_state,
                img_mask=view["img_mask"],
                reset_mask=view["reset"],
                update=view.get("update"),
            )
            final_image_tokens = decoder_images[model.dec_depth]
            if model.pose_head_flag:
                final_image_tokens = final_image_tokens[:, 1:, :]
                output_pose_feature = decoder_images[-1][:, 0:1]
                new_memory = model.pose_retriever.update_mem(
                    memory, global_image_feature, output_pose_feature
                )
            else:
                new_memory = memory
            expected_tokens = token_grid[0] * token_grid[1]
            if final_image_tokens.shape[1] != expected_tokens:
                raise ValueError(
                    f"CUT3R returned {final_image_tokens.shape[1]} spatial tokens "
                    "at timestep "
                    f"{timestep + 1}, expected {expected_tokens} for grid {token_grid}"
                )
            image_mask = view["img_mask"]
            update = view.get("update")
            update_mask = image_mask & update if update is not None else image_mask
            update_weight = update_mask[:, None, None].to(dtype=new_state.dtype)
            state = new_state * update_weight + state * (1.0 - update_weight)
            memory = new_memory * update_weight + memory * (1.0 - update_weight)
            reset_mask = view.get("reset")
            if reset_mask is not None:
                reset_weight = reset_mask[:, None, None].to(dtype=state.dtype)
                state = initial_state * reset_weight + state * (1.0 - reset_weight)
                memory = initial_memory * reset_weight + memory * (1.0 - reset_weight)
            image_trajectory.append(final_image_tokens.float())
            state_trajectory.append(state.float())

        return FeatureTrajectory(
            image_tokens=torch.stack(image_trajectory, dim=0),
            state_tokens=torch.stack(state_trajectory, dim=0),
            frame_ids=list(frame_ids),
            token_grid=token_grid,
        )
