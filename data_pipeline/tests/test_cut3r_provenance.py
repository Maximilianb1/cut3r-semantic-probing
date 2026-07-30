from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from src.embeddings.cut3r_provenance import (
    CUROPE_PATCH_ID,
    validate_cut3r_checkout,
)

KERNEL_SNIPPET = """void rope_2d_cuda() {
    const int N_BLOCKS = B * N; // each block takes care of H*D values
    const int SHARED_MEM = sizeof(float) * (D + D/4);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.type(), "rope_2d_cuda", ([&] {
        rope_2d_cuda_kernel<scalar_t> <<<N_BLOCKS, THREADS_PER_BLOCK, SHARED_MEM>>> (
            //tokens.data_ptr<scalar_t>(),
            tokens.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            pos.data_ptr<int64_t>());
    }));
}
"""


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "CUT3R"
    source = repository / "src" / "croco" / "models" / "curope" / "kernels.cu"
    source.parent.mkdir(parents=True)
    source.write_text(KERNEL_SNIPPET, encoding="utf-8")
    _git(repository, "init")
    _git(repository, "add", ".")
    _git(
        repository,
        "-c",
        "user.name=Stage0 Test",
        "-c",
        "user.email=stage0@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    return repository, _git(repository, "rev-parse", "HEAD")


def test_compatibility_patch_is_applied_and_validated(tmp_path: Path) -> None:
    repository, commit = _repository(tmp_path)
    validate_cut3r_checkout(repository, expected_commit=commit, expected_patch=None)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.apply_cut3r_compatibility_patch",
            "--cut3r-root",
            str(repository),
            "--expected-commit",
            commit,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.apply_cut3r_compatibility_patch",
            "--cut3r-root",
            str(repository),
            "--expected-commit",
            commit,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    provenance = validate_cut3r_checkout(
        repository, expected_commit=commit, expected_patch=CUROPE_PATCH_ID
    )
    assert provenance["compatibility_patch"] == CUROPE_PATCH_ID
    assert len(provenance["patched_source_sha256"]) == 64

    with pytest.raises(RuntimeError, match="Compiled cuRoPE extension is missing"):
        validate_cut3r_checkout(
            repository,
            expected_commit=commit,
            expected_patch=CUROPE_PATCH_ID,
            require_compiled_extension=True,
        )

    extension = (
        repository
        / "src"
        / "croco"
        / "models"
        / "curope"
        / "curope.cpython-311-x86_64-linux-gnu.so"
    )
    extension.write_bytes(b"compiled-extension-fixture")
    provenance = validate_cut3r_checkout(
        repository,
        expected_commit=commit,
        expected_patch=CUROPE_PATCH_ID,
        require_compiled_extension=True,
    )
    assert provenance["compiled_extension"] == {
        "filename": extension.name,
        "sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
    }


def test_compatibility_patch_rejects_additional_changes(tmp_path: Path) -> None:
    repository, commit = _repository(tmp_path)
    source = repository / "src" / "croco" / "models" / "curope" / "kernels.cu"
    source.write_text(KERNEL_SNIPPET.replace("tokens.type()", "tokens.scalar_type()"))
    (repository / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(RuntimeError, match="other than the expected"):
        validate_cut3r_checkout(
            repository, expected_commit=commit, expected_patch=CUROPE_PATCH_ID
        )
