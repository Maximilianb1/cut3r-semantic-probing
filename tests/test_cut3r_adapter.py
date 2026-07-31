from __future__ import annotations

import torch

from src.embeddings.cut3r_adapter import Cut3rFeatureExtractor


class _Retriever:
    def __init__(self) -> None:
        self.mem = torch.zeros(1, 1, 3)

    def inquire(self, global_image_feature, memory):
        return global_image_feature + memory

    def update_mem(self, memory, global_image_feature, output_pose_feature):
        return memory + global_image_feature + output_pose_feature


class _FakeCut3r(torch.nn.Module):
    pose_head_flag = False
    dec_depth = 1

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.pose_retriever = _Retriever()

    def _encode_views(self, views):
        features = tuple(
            torch.full((1, 4, 3), float(index + 1), device=view["img"].device)
            for index, view in enumerate(views)
        )
        positions = tuple(
            torch.zeros(1, 4, 2, device=view["img"].device) for view in views
        )
        shapes = tuple(torch.tensor([[2, 2]]) for _ in views)
        return shapes, [features], positions

    def _init_state(self, feature, position):
        return torch.zeros(1, 2, 3, device=feature.device), torch.zeros(
            1, 2, 2, device=feature.device
        )

    def _recurrent_rollout(
        self, state, state_position, feature, position, *args, **kwargs
    ):
        new_state = state + feature.mean(dim=1, keepdim=True)
        conditioned_image = feature + state.mean(dim=1, keepdim=True)
        return new_state, (conditioned_image, conditioned_image)


def test_adapter_saves_image_and_committed_state_after_every_timestep() -> None:
    model = _FakeCut3r()
    views = [
        {
            "img": torch.zeros(1, 3, 2, 2),
            "img_mask": torch.tensor([True]),
            "update": torch.tensor([True]),
            "reset": torch.tensor([False]),
        }
        for _ in range(6)
    ]
    result = Cut3rFeatureExtractor(model).extract(
        views, frame_ids=[f"frame-{index}" for index in range(6)], token_grid=(2, 2)
    )
    assert result.image_tokens.shape == (6, 1, 4, 3)
    assert result.state_tokens.shape == (6, 1, 2, 3)
    assert torch.all(result.image_tokens[0] == 1)
    assert torch.all(result.state_tokens[0] == 1)
    assert torch.all(result.image_tokens[1] == 3)
    assert torch.all(result.state_tokens[1] == 3)
    assert not model.training
    assert not model.weight.requires_grad


class _FakePoseCut3r(_FakeCut3r):
    pose_head_flag = True

    def __init__(self) -> None:
        super().__init__()
        self.pose_token = torch.nn.Parameter(torch.ones(1, 1, 3))

    def _get_img_level_feat(self, feature):
        return feature.mean(dim=1, keepdim=True)

    def _recurrent_rollout(
        self,
        state,
        state_position,
        feature,
        position,
        pose_feature,
        pose_position,
        *args,
        **kwargs,
    ):
        assert torch.equal(pose_position, -torch.ones_like(pose_position))
        new_state = state + feature.mean(dim=1, keepdim=True)
        pose_and_image = torch.cat([pose_feature, feature], dim=1)
        return new_state, (pose_and_image, pose_and_image)


def test_adapter_matches_upstream_pose_position_and_removes_pose_token() -> None:
    model = _FakePoseCut3r()
    views = [
        {
            "img": torch.zeros(1, 3, 2, 2),
            "img_mask": torch.tensor([True]),
            "update": torch.tensor([True]),
            "reset": torch.tensor([False]),
        }
        for _ in range(6)
    ]
    result = Cut3rFeatureExtractor(model).extract(
        views, frame_ids=[f"frame-{index}" for index in range(6)], token_grid=(2, 2)
    )
    assert result.image_tokens.shape == (6, 1, 4, 3)
