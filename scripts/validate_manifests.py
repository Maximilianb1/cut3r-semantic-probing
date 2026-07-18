from __future__ import annotations

import argparse
import json

from src.data.validation import validate_manifests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Stage 0 manifests and leakage rules"
    )
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument(
        "--inspect-files",
        action="store_true",
        help="Decode every selected RGB/mask and verify its annotated dimensions",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate_manifests(
                args.manifest_dir,
                dataset_root=args.dataset_root,
                inspect_files=args.inspect_files,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
