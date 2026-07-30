from __future__ import annotations

import argparse
import json
import sys

from src.data.co3d_selective import run_selective_download


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download only the deterministic CO3Dv2 RGB/mask entries required "
            "by a capped Stage 0 configuration"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Verify/extract metadata and print the deterministic plan only",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help=(
            "Plan and locate every required remote member without downloading "
            "the RGB/mask payloads"
        ),
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--initial-buffer-mib", type=int, default=16)
    args = parser.parse_args()
    print(
        json.dumps(
            run_selective_download(
                args.config,
                plan_only=args.plan_only,
                index_only=args.index_only,
                timeout=args.timeout,
                initial_buffer_size=args.initial_buffer_mib * 1024 * 1024,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
