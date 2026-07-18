from __future__ import annotations

import argparse
import json

from src.embeddings.cache import verify_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Stage 0 CUT3R cache integrity")
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(verify_cache(args.cache_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
