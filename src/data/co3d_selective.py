from __future__ import annotations

import binascii
import hashlib
import io
import json
import os
import stat
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from zipfile import ZipFile, ZipInfo

import requests
from remotezip import RemoteZip

from src.common.io import (
    atomic_write_bytes,
    atomic_write_json,
    load_yaml,
    sha256_file,
)
from src.data.co3d import load_jgzip, load_official_sequence_splits
from src.data.windows import choose_sequences, generate_ordered_windows

CO3D_TOOLING_COMMIT = "eb51d7583c56ff23dc918d9deafee50f4d8178c3"
CO3D_RAW_ROOT = (
    "https://raw.githubusercontent.com/facebookresearch/co3d/"
    f"{CO3D_TOOLING_COMMIT}/co3d"
)
DEFAULT_LINKS_URL = f"{CO3D_RAW_ROOT}/links.json"
DEFAULT_CHECKSUMS_URL = f"{CO3D_RAW_ROOT}/co3d_sha256.json"
SELECTIVE_DOWNLOAD_SCHEMA_VERSION = "co3d-selective-download-v1"
DEFAULT_INITIAL_BUFFER_SIZE = 16 * 1024 * 1024
DEFAULT_MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_METADATA_UNCOMPRESSED_BYTES = 512 * 1024 * 1024


class ArchiveReader(Protocol):
    def __enter__(self) -> ArchiveReader: ...

    def __exit__(self, *args: Any) -> None: ...

    def infolist(self) -> list[ZipInfo]: ...

    def getinfo(self, name: str) -> ZipInfo: ...

    def read(self, name: str) -> bytes: ...


ArchiveOpener = Callable[[str], ArchiveReader]
ProgressCallback = Callable[[str], None]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected {context} to be a mapping")
    return value


def _safe_relative_parts(relative: str) -> tuple[str, ...]:
    if not relative or "\\" in relative:
        raise ValueError(f"Unsafe archive member path: {relative!r}")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"Unsafe archive member path: {relative!r}")
    return pure.parts


def _safe_destination(root: Path, relative: str) -> Path:
    parts = _safe_relative_parts(relative)
    destination = (root / Path(*parts)).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Archive member escapes dataset root: {relative}") from error
    return destination


def _is_symlink(info: ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def _crc32_bytes(payload: bytes) -> int:
    return binascii.crc32(payload) & 0xFFFFFFFF


def _verify_payload(info: ZipInfo, payload: bytes, *, max_member_bytes: int) -> None:
    if info.is_dir() or _is_symlink(info):
        raise ValueError(f"Refusing non-regular ZIP member: {info.filename}")
    if info.file_size > max_member_bytes:
        raise ValueError(
            f"ZIP member exceeds {max_member_bytes} bytes: {info.filename}"
        )
    if len(payload) != info.file_size:
        raise ValueError(f"ZIP member size mismatch: {info.filename}")
    if _crc32_bytes(payload) != info.CRC:
        raise ValueError(f"ZIP member CRC mismatch: {info.filename}")


def _metadata_member_allowed(category: str, info: ZipInfo) -> bool:
    name = info.filename.rstrip("/")
    if name in {category, f"{category}/set_lists", f"{category}/eval_batches"}:
        return True
    if name in {
        f"{category}/LICENSE",
        f"{category}/frame_annotations.jgz",
        f"{category}/sequence_annotations.jgz",
    }:
        return True
    return name.startswith(f"{category}/set_lists/") or name.startswith(
        f"{category}/eval_batches/"
    )


def extract_verified_metadata_archive(
    archive_path: str | Path,
    *,
    dataset_root: str | Path,
    category: str,
) -> list[str]:
    root = Path(dataset_root)
    written: list[str] = []
    with ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if sum(info.file_size for info in infos) > MAX_METADATA_UNCOMPRESSED_BYTES:
            raise ValueError(f"Metadata archive is unexpectedly large: {archive_path}")
        for info in infos:
            if not _metadata_member_allowed(category, info):
                raise ValueError(
                    f"Unexpected member in {category} metadata archive: {info.filename}"
                )
            _safe_destination(root, info.filename)
            if _is_symlink(info):
                raise ValueError(
                    f"Metadata archive contains a symlink: {info.filename}"
                )
        for info in infos:
            if info.is_dir():
                continue
            payload = archive.read(info)
            _verify_payload(
                info, payload, max_member_bytes=MAX_METADATA_UNCOMPRESSED_BYTES
            )
            destination = _safe_destination(root, info.filename)
            atomic_write_bytes(destination, payload)
            written.append(info.filename)
    return sorted(written)


def _usable_annotation(
    annotation: Mapping[str, Any], *, require_viewpoint: bool
) -> bool:
    image = annotation.get("image")
    mask = annotation.get("mask")
    if not isinstance(image, Mapping) or not image.get("path") or not image.get("size"):
        return False
    if not isinstance(mask, Mapping) or not mask.get("path"):
        return False
    return not require_viewpoint or annotation.get("viewpoint") is not None


def plan_selective_download(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    dataset = _require_mapping(config.get("dataset"), "dataset config")
    sampling = _require_mapping(config.get("sampling"), "sampling config")
    categories = dataset.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError(
            "Selective download requires an explicit non-empty category list"
        )
    if not all(isinstance(category, str) and category for category in categories):
        raise ValueError("Selective download categories must be non-empty strings")
    if len(categories) != len(set(categories)):
        raise ValueError("Selective download categories must be unique")
    expected_count = dataset.get("expected_category_count")
    if expected_count is not None and int(expected_count) != len(categories):
        raise ValueError(
            f"Expected {expected_count} categories, found {len(categories)}"
        )
    root = Path(str(dataset["root"]))
    caps = _require_mapping(sampling.get("sequence_caps"), "sequence caps")
    for split in ("train", "val", "test"):
        if split not in caps or caps[split] is None:
            raise ValueError(
                f"Selective download requires a finite sequence cap for {split}"
            )
        if isinstance(caps[split], bool) or int(caps[split]) < 1:
            raise ValueError(f"Invalid sequence cap for {split}: {caps[split]}")
    window_length = int(sampling.get("window_length", 0))
    if window_length != 6:
        raise ValueError("Selective download requires six-frame windows")
    windows_per_sequence = int(sampling.get("windows_per_sequence", 0))
    if windows_per_sequence < 1:
        raise ValueError("windows_per_sequence must be positive")
    seed = int(sampling.get("seed", 0))
    require_viewpoint = bool(dataset.get("require_viewpoint", True))

    selected_sequences: list[dict[str, Any]] = []
    selected_windows: list[dict[str, Any]] = []
    required_files: dict[str, dict[str, Any]] = {}
    for category in categories:
        category_dir = root / category
        splits = load_official_sequence_splits(category_dir)
        split_by_sequence = {
            sequence_id: split
            for split, sequence_ids in splits.items()
            for sequence_id in sequence_ids
        }
        annotations = load_jgzip(category_dir / "frame_annotations.jgz")
        frames_by_sequence: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        paths_by_frame_id: dict[str, tuple[str, str]] = {}
        for annotation in annotations:
            if not isinstance(annotation, Mapping) or not _usable_annotation(
                annotation, require_viewpoint=require_viewpoint
            ):
                continue
            sequence_id = str(annotation["sequence_name"])
            split = split_by_sequence.get(sequence_id)
            if split is None:
                continue
            number = int(annotation["frame_number"])
            identifier = f"{category}/{sequence_id}:{number}"
            image_path = str(annotation["image"]["path"])
            mask_path = str(annotation["mask"]["path"])
            _safe_relative_parts(image_path)
            _safe_relative_parts(mask_path)
            frame = {
                "frame_id": identifier,
                "category": category,
                "sequence_id": sequence_id,
                "split": split,
                "frame_number": number,
            }
            frames_by_sequence[sequence_id].append(frame)
            if identifier in paths_by_frame_id:
                raise ValueError(f"Duplicate usable frame annotation: {identifier}")
            paths_by_frame_id[identifier] = (image_path, mask_path)

        for split in ("train", "val", "test"):
            eligible = sorted(
                sequence_id
                for sequence_id in splits[split]
                if len(frames_by_sequence.get(sequence_id, [])) >= window_length
            )
            chosen = choose_sequences(
                eligible,
                cap=int(caps[split]),
                seed=seed,
                category=category,
                split=split,
            )
            for sequence_id in chosen:
                windows = generate_ordered_windows(
                    frames_by_sequence[sequence_id],
                    window_length=window_length,
                    windows_per_sequence=windows_per_sequence,
                )
                selected_sequences.append(
                    {
                        "category": category,
                        "split": split,
                        "sequence_id": sequence_id,
                        "usable_annotation_frames": len(
                            frames_by_sequence[sequence_id]
                        ),
                        "planned_windows": len(windows),
                    }
                )
                selected_windows.extend(windows)
                for window in windows:
                    for identifier in window["frame_ids"]:
                        image_path, mask_path = paths_by_frame_id[identifier]
                        for kind, relative in (
                            ("image", image_path),
                            ("mask", mask_path),
                        ):
                            existing = required_files.get(relative)
                            record = {
                                "path": relative,
                                "kind": kind,
                                "frame_id": identifier,
                                "category": category,
                                "sequence_id": sequence_id,
                                "split": split,
                            }
                            if existing is not None and existing != record:
                                raise ValueError(
                                    "Conflicting annotations for required path: "
                                    f"{relative}"
                                )
                            required_files[relative] = record

    return {
        "schema_version": SELECTIVE_DOWNLOAD_SCHEMA_VERSION,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset_root": str(root),
        "categories": list(categories),
        "seed": seed,
        "selected_sequences": sorted(
            selected_sequences,
            key=lambda row: (row["category"], row["split"], row["sequence_id"]),
        ),
        "selected_windows": sorted(
            selected_windows,
            key=lambda row: (row["category"], row["split"], row["window_id"]),
        ),
        "required_files": [required_files[path] for path in sorted(required_files)],
        "counts": {
            "sequences": len(selected_sequences),
            "windows": len(selected_windows),
            "frames": len(required_files) // 2,
            "files": len(required_files),
        },
    }


def find_required_members(
    archive_urls: Iterable[str],
    required_paths: Iterable[str],
    *,
    archive_opener: ArchiveOpener,
    progress: ProgressCallback | None = None,
) -> dict[str, dict[str, Any]]:
    required = set(required_paths)
    remaining = set(required)
    found: dict[str, dict[str, Any]] = {}
    urls = list(archive_urls)
    for archive_number, url in enumerate(urls, start=1):
        if progress is not None:
            progress(
                f"Indexing archive {archive_number}/{len(urls)} "
                f"({len(remaining)} members remaining): {url.rsplit('/', 1)[-1]}"
            )
        with archive_opener(url) as archive:
            for info in archive.infolist():
                if info.filename not in required:
                    continue
                if info.filename in found:
                    raise ValueError(
                        f"Required member occurs in multiple archives: {info.filename}"
                    )
                if info.is_dir() or _is_symlink(info):
                    raise ValueError(f"Required member is not regular: {info.filename}")
                found[info.filename] = {
                    "url": url,
                    "archive": url.rsplit("/", 1)[-1],
                    "size": info.file_size,
                    "compressed_size": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                }
                remaining.remove(info.filename)
        if not remaining:
            break
    if remaining:
        examples = ", ".join(sorted(remaining)[:5])
        raise FileNotFoundError(
            f"Could not locate {len(remaining)} required archive members: {examples}"
        )
    return found


def materialize_required_members(
    member_sources: Mapping[str, Mapping[str, Any]],
    *,
    dataset_root: str | Path,
    archive_opener: ArchiveOpener,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    root = Path(dataset_root)
    by_url: defaultdict[str, list[str]] = defaultdict(list)
    for relative, source in member_sources.items():
        _safe_destination(root, relative)
        by_url[str(source["url"])].append(relative)
    records: list[dict[str, Any]] = []
    urls = sorted(by_url)
    for archive_number, url in enumerate(urls, start=1):
        if progress is not None:
            progress(
                f"Reading members from archive {archive_number}/{len(urls)}: "
                f"{url.rsplit('/', 1)[-1]} ({len(by_url[url])} files)"
            )
        with archive_opener(url) as archive:
            for relative in sorted(by_url[url]):
                info = archive.getinfo(relative)
                destination = _safe_destination(root, relative)
                source = member_sources[relative]
                if info.file_size != int(source["size"]) or f"{info.CRC:08x}" != str(
                    source["crc32"]
                ):
                    raise ValueError(
                        "Remote ZIP member metadata changed between indexing and "
                        f"download: {relative}"
                    )
                status = "downloaded"
                if destination.is_file():
                    payload = destination.read_bytes()
                    _verify_payload(info, payload, max_member_bytes=max_member_bytes)
                    status = "verified_existing"
                else:
                    payload = archive.read(relative)
                    _verify_payload(info, payload, max_member_bytes=max_member_bytes)
                    atomic_write_bytes(destination, payload)
                records.append(
                    {
                        "path": relative,
                        "archive": str(source["archive"]),
                        "archive_url": url,
                        "size": len(payload),
                        "crc32": f"{info.CRC:08x}",
                        "sha256": _sha256_bytes(payload),
                        "status": status,
                    }
                )
    return sorted(records, key=lambda row: row["path"])


def _fetch_json(url: str, *, timeout: float) -> tuple[dict[str, Any], str]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.content
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping from {url}")
    return value, _sha256_bytes(payload)


def _download_file(url: str, destination: Path, *, timeout: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".part"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _remote_archive_opener(
    *, timeout: float, initial_buffer_size: int
) -> ArchiveOpener:
    def open_archive(url: str) -> ArchiveReader:
        return RemoteZip(
            url,
            timeout=timeout,
            initial_buffer_size=initial_buffer_size,
        )

    return open_archive


def run_selective_download(
    config_path: str | Path,
    *,
    plan_only: bool = False,
    index_only: bool = False,
    timeout: float = 120.0,
    initial_buffer_size: int = DEFAULT_INITIAL_BUFFER_SIZE,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if plan_only and index_only:
        raise ValueError("plan_only and index_only are mutually exclusive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if initial_buffer_size < 64 * 1024:
        raise ValueError("initial_buffer_size must be at least 64 KiB")
    config = load_yaml(config_path)
    dataset = _require_mapping(config.get("dataset"), "dataset config")
    categories = dataset.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError(
            "Selective download requires an explicit non-empty category list"
        )
    root = Path(str(dataset["root"]))
    control_dir = root / ".co3d-selective"
    metadata_dir = control_dir / "metadata-archives"
    if progress is not None:
        progress("Fetching pinned official CO3D link and checksum indexes")
    links, links_sha256 = _fetch_json(DEFAULT_LINKS_URL, timeout=timeout)
    checksums, checksums_sha256 = _fetch_json(DEFAULT_CHECKSUMS_URL, timeout=timeout)
    full_links = _require_mapping(links.get("full"), "full CO3D links")
    full_checksums = _require_mapping(checksums.get("full"), "full CO3D checksums")
    metadata_records: list[dict[str, Any]] = []
    category_data_urls: dict[str, list[str]] = {}
    for category in categories:
        if progress is not None:
            progress(f"Verifying full-release metadata for category: {category}")
        urls = full_links.get(category)
        if not isinstance(urls, list) or not all(isinstance(url, str) for url in urls):
            raise ValueError(f"No official full-release links for {category}")
        metadata_name = f"{category}_000.zip"
        matches = [url for url in urls if url.rsplit("/", 1)[-1] == metadata_name]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one metadata archive for {category}, "
                f"found {len(matches)}"
            )
        expected_sha256 = full_checksums.get(metadata_name)
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError(f"Missing official SHA-256 for {metadata_name}")
        archive_path = metadata_dir / metadata_name
        if not archive_path.is_file():
            _download_file(matches[0], archive_path, timeout=timeout)
        actual_sha256 = sha256_file(archive_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Metadata SHA-256 mismatch for {metadata_name}: {actual_sha256}"
            )
        members = extract_verified_metadata_archive(
            archive_path, dataset_root=root, category=category
        )
        metadata_records.append(
            {
                "category": category,
                "archive": metadata_name,
                "url": matches[0],
                "sha256": actual_sha256,
                "members": members,
            }
        )
        category_data_urls[category] = [url for url in urls if url not in matches]

    plan = plan_selective_download(config_path)
    if progress is not None:
        counts = plan["counts"]
        progress(
            "Planned "
            f"{counts['sequences']} sequences, {counts['windows']} windows, "
            f"and {counts['files']} RGB/mask files"
        )
    if plan_only:
        return {
            "plan_only": True,
            "official_tooling_commit": CO3D_TOOLING_COMMIT,
            "links_json_sha256": links_sha256,
            "checksums_json_sha256": checksums_sha256,
            "metadata_archives": metadata_records,
            "plan": plan,
        }

    opener = _remote_archive_opener(
        timeout=timeout, initial_buffer_size=initial_buffer_size
    )
    sources: dict[str, dict[str, Any]] = {}
    required_by_category: defaultdict[str, list[str]] = defaultdict(list)
    for row in plan["required_files"]:
        required_by_category[row["category"]].append(row["path"])
    for category in categories:
        found = find_required_members(
            category_data_urls[category],
            required_by_category[category],
            archive_opener=opener,
            progress=progress,
        )
        for relative, source in found.items():
            archive_sha256 = full_checksums.get(source["archive"])
            if not isinstance(archive_sha256, str) or len(archive_sha256) != 64:
                raise ValueError(
                    f"Missing official archive SHA-256 for {source['archive']}"
                )
            sources[relative] = {**source, "official_archive_sha256": archive_sha256}
    if index_only:
        source_rows = [
            {"path": relative, **sources[relative]} for relative in sorted(sources)
        ]
        return {
            "index_only": True,
            "official_tooling_commit": CO3D_TOOLING_COMMIT,
            "links_json_sha256": links_sha256,
            "checksums_json_sha256": checksums_sha256,
            "metadata_archives": metadata_records,
            "plan": plan,
            "source_index": source_rows,
            "source_counts": {
                "files": len(source_rows),
                "archives": len({row["archive"] for row in source_rows}),
                "uncompressed_bytes": sum(int(row["size"]) for row in source_rows),
                "compressed_bytes": sum(
                    int(row["compressed_size"]) for row in source_rows
                ),
            },
        }
    files = materialize_required_members(
        sources,
        dataset_root=root,
        archive_opener=opener,
        progress=progress,
    )
    source_by_path = {row["path"]: row for row in plan["required_files"]}
    for row in files:
        row.update(
            {
                key: source_by_path[row["path"]][key]
                for key in ("kind", "frame_id", "category", "sequence_id", "split")
            }
        )
        row["official_archive_sha256"] = sources[row["path"]]["official_archive_sha256"]
    result = {
        "schema_version": SELECTIVE_DOWNLOAD_SCHEMA_VERSION,
        "official_tooling_commit": CO3D_TOOLING_COMMIT,
        "links_url": DEFAULT_LINKS_URL,
        "links_json_sha256": links_sha256,
        "checksums_url": DEFAULT_CHECKSUMS_URL,
        "checksums_json_sha256": checksums_sha256,
        "metadata_archives": metadata_records,
        "plan": plan,
        "files": files,
        "counts": {
            **plan["counts"],
            "downloaded": sum(row["status"] == "downloaded" for row in files),
            "verified_existing": sum(
                row["status"] == "verified_existing" for row in files
            ),
            "source_archives": len({row["archive"] for row in files}),
            "bytes": sum(int(row["size"]) for row in files),
        },
    }
    atomic_write_json(control_dir / "selection.json", result)
    return result


def in_memory_archive_opener(archives: Mapping[str, bytes]) -> ArchiveOpener:
    """Test helper for exercising archive selection without network access."""

    def open_archive(url: str) -> ArchiveReader:
        return ZipFile(io.BytesIO(archives[url]))

    return open_archive
