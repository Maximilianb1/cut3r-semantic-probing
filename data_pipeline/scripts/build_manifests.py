from __future__ import annotations

import argparse
import json

from src.data.co3d import build_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 0 CO3Dv2 manifests")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(build_manifests(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
