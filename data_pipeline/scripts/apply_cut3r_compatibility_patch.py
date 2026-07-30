from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from src.embeddings.cut3r_provenance import (
    CUROPE_PATCH_ID,
    validate_cut3r_checkout,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the audited CUT3R cuRoPE/PyTorch compatibility patch"
    )
    parser.add_argument("--cut3r-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    root = Path(args.cut3r_root).resolve()
    project_root = Path(__file__).resolve().parents[1]
    patch = project_root / "patches" / "cut3r" / f"{CUROPE_PATCH_ID}.patch"
    try:
        validate_cut3r_checkout(
            root,
            expected_commit=args.expected_commit,
            expected_patch=CUROPE_PATCH_ID,
        )
    except RuntimeError:
        validate_cut3r_checkout(
            root, expected_commit=args.expected_commit, expected_patch=None
        )
        subprocess.run(
            ["git", "-C", str(root), "apply", "--check", str(patch)], check=True
        )
        subprocess.run(["git", "-C", str(root), "apply", str(patch)], check=True)
    provenance = validate_cut3r_checkout(
        root, expected_commit=args.expected_commit, expected_patch=CUROPE_PATCH_ID
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
