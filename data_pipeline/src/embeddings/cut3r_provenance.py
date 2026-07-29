from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

CUROPE_PATCH_ID = "curope-scalar-type-v1"
CUROPE_SOURCE = Path("src/croco/models/curope/kernels.cu")
CUROPE_DIRECTORY = CUROPE_SOURCE.parent
OLD_DISPATCH = 'AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.type(), "rope_2d_cuda"'
NEW_DISPATCH = (
    'AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.scalar_type(), "rope_2d_cuda"'
)


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _normalized(text: str) -> str:
    return text.replace("\r\n", "\n")


def _is_generated_curope_artifact(status_line: str) -> bool:
    if not status_line.startswith("?? "):
        return False
    relative_path = status_line[3:]
    directory = CUROPE_DIRECTORY.as_posix()
    return relative_path.startswith(f"{directory}/build/") or (
        relative_path.startswith(f"{directory}/curope")
        and relative_path.endswith(".so")
        and "/" not in relative_path[len(directory) + 1 :]
    )


def validate_cut3r_checkout(
    repository: str | Path,
    *,
    expected_commit: str,
    expected_patch: str | None,
    require_compiled_extension: bool = False,
) -> dict[str, Any]:
    root = Path(repository).resolve()
    commit = _git(root, "rev-parse", "HEAD").strip()
    if commit != expected_commit:
        raise RuntimeError(
            f"CUT3R commit mismatch: expected {expected_commit}, got {commit}"
        )
    status = _git(root, "status", "--porcelain", "--untracked-files=all").splitlines()
    source_status = f" M {CUROPE_SOURCE.as_posix()}"
    unexpected_status = [
        line
        for line in status
        if line != source_status and not _is_generated_curope_artifact(line)
    ]
    if expected_patch is None:
        if unexpected_status or source_status in status:
            raise RuntimeError(
                "CUT3R checkout contains source changes: "
                f"{unexpected_status or [source_status]}"
            )
        provenance: dict[str, Any] = {
            "commit": commit,
            "compatibility_patch": None,
        }
        return _with_compiled_extension(
            provenance, root, required=require_compiled_extension
        )
    if expected_patch != CUROPE_PATCH_ID:
        raise ValueError(f"Unsupported CUT3R compatibility patch: {expected_patch}")
    if source_status not in status or unexpected_status:
        raise RuntimeError(
            "CUT3R checkout contains changes other than the expected compatibility "
            f"patch and generated cuRoPE artifacts: {status}"
        )
    original = _normalized(_git(root, "show", f"{commit}:{CUROPE_SOURCE.as_posix()}"))
    if original.count(OLD_DISPATCH) != 1 or NEW_DISPATCH in original:
        raise RuntimeError("Pinned CUT3R source does not match the patch precondition")
    expected = original.replace(OLD_DISPATCH, NEW_DISPATCH)
    actual = _normalized((root / CUROPE_SOURCE).read_text(encoding="utf-8"))
    if actual != expected:
        raise RuntimeError("CUT3R compatibility patch content is not exact")
    provenance = {
        "commit": commit,
        "compatibility_patch": CUROPE_PATCH_ID,
        "patched_source_sha256": hashlib.sha256(expected.encode("utf-8")).hexdigest(),
    }
    return _with_compiled_extension(
        provenance, root, required=require_compiled_extension
    )


def _with_compiled_extension(
    provenance: dict[str, Any], repository: Path, *, required: bool
) -> dict[str, Any]:
    extension_directory = repository / CUROPE_DIRECTORY
    shared_objects = sorted(extension_directory.glob("curope*.so"))
    if len(shared_objects) > 1:
        names = [path.name for path in shared_objects]
        raise RuntimeError(
            f"Multiple cuRoPE extensions found; remove stale builds: {names}"
        )
    if not shared_objects:
        if required:
            raise RuntimeError(
                "Compiled cuRoPE extension is missing; run "
                "`python setup.py build_ext --inplace` in "
                f"{extension_directory}"
            )
        return provenance
    extension = shared_objects[0]
    provenance["compiled_extension"] = {
        "filename": extension.name,
        "sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
    }
    return provenance
