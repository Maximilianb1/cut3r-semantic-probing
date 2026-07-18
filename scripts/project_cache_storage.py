from __future__ import annotations

import argparse
import json

from src.embeddings.projection import project_cache_storage


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Project exact Stage 0 tensor storage from manifest token grids and "
            "fail when the target filesystem lacks the configured reserve"
        )
    )
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--filesystem-path", required=True)
    parser.add_argument("--reserve-gib", type=float, default=10.0)
    parser.add_argument("--overhead-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if args.reserve_gib < 0:
        parser.error("--reserve-gib must be non-negative")
    result = project_cache_storage(
        args.manifest_dir,
        filesystem_path=args.filesystem_path,
        reserve_bytes=round(args.reserve_gib * 1024**3),
        overhead_fraction=args.overhead_fraction,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["sufficient"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
