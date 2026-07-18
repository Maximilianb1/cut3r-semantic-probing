from __future__ import annotations

import argparse
import json

from src.embeddings.cache import compare_caches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two independently extracted Stage 0 caches"
    )
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()
    print(
        json.dumps(
            compare_caches(args.left, args.right, atol=args.atol, rtol=args.rtol),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
