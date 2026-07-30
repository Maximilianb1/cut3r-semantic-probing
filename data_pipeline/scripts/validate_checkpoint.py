from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.common.io import load_yaml
from src.embeddings.checkpoint import validate_cut3r_checkpoint
from src.embeddings.extract import load_cut3r_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the pinned CUT3R checkpoint and optionally load the model"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    model_config = config["model"]
    checkpoint = Path(model_config["checkpoint"])
    expected_sha256 = str(model_config["checkpoint_sha256"])
    result: dict[str, Any] = {}

    if args.load_model:
        requested_device = str(model_config.get("device", "cuda"))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA model loading was requested but "
                "torch.cuda.is_available() is false"
            )
        device = torch.device(requested_device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        model, provenance = load_cut3r_model(
            model_config["cut3r_root"],
            checkpoint,
            device,
            expected_checkpoint_sha256=expected_sha256,
        )
        result.update(
            {
                "checkpoint_provenance": provenance,
                "model_class": type(model).__name__,
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "device": str(next(model.parameters()).device),
                "peak_cuda_bytes": (
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else None
                ),
            }
        )
    else:
        result["checkpoint_provenance"] = validate_cut3r_checkpoint(
            checkpoint, expected_sha256=expected_sha256
        )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
