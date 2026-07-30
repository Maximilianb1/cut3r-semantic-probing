from __future__ import annotations

import argparse
import json

from src.embeddings.extract import run_extraction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frozen CUT3R trajectory features"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-windows", type=int)
    parser.add_argument(
        "--cache-dir",
        help="Override cache.directory (useful for independent reproducibility runs)",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_extraction(
                args.config,
                limit_windows=args.limit_windows,
                cache_directory=args.cache_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
