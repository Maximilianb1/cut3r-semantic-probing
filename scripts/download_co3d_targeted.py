"""Download exactly the raw CO3D image/mask files referenced by an already-built
manifest's frames.parquet -- no re-sampling, so the fetched files exactly match
whatever manifest a probe cache was already built from.

Unlike `download_co3d_selective`, which re-derives its own sequence/window
selection from a config's `sampling` block (and is not guaranteed to
reproduce an existing manifest's exact selection), this script trusts a
manifest that already exists and fetches only the files it names. Use this
when you have a trusted manifest and just need the pixels on disk -- e.g. to
extract a backbone that has no cache yet, or to build qualitative figures.

Reuses the project's own verified download primitives (pinned official links/
checksums, per-file CRC verification, atomic writes) from src.data.co3d_selective;
it only replaces *which* files get requested.

Run example:
    python -m scripts.download_co3d_targeted \
        --frames-parquet ${CUT3R_ARTIFACT_ROOT}/manifests/full51-part-a-v1/frames.parquet \
        --dataset-root ${CO3D_ROOT} \
        --limit-categories 1        # optional smoke test first
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from src.common.tables import read_parquet
from src.data.co3d_selective import (
    DEFAULT_CHECKSUMS_URL,
    DEFAULT_LINKS_URL,
    _fetch_json,
    _remote_archive_opener,
    find_required_members,
    materialize_required_members,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-parquet", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--limit-categories", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    root.mkdir(parents=True, exist_ok=True)

    rows = read_parquet(args.frames_parquet)
    required_by_category: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        required_by_category[row["category"]].add(row["image_relpath"])
        required_by_category[row["category"]].add(row["mask_relpath"])

    categories = sorted(required_by_category)
    if args.limit_categories is not None:
        categories = categories[: args.limit_categories]

    total_required = sum(len(required_by_category[c]) for c in categories)
    print(
        f"Targeting {len(categories)} categories, {total_required} exact files "
        f"(from {len(rows)} manifest frame rows). No re-sampling.",
        flush=True,
    )

    print("Fetching pinned official CO3D link and checksum indexes", flush=True)
    links, _ = _fetch_json(DEFAULT_LINKS_URL, timeout=args.timeout)
    checksums, _ = _fetch_json(DEFAULT_CHECKSUMS_URL, timeout=args.timeout)
    full_links = links["full"]
    full_checksums = checksums["full"]

    opener = _remote_archive_opener(timeout=args.timeout, initial_buffer_size=16 * 1024 * 1024)

    grand_total_bytes = 0
    for category in categories:
        required_paths = sorted(required_by_category[category])
        urls = full_links.get(category)
        if not urls:
            raise ValueError(f"No official full-release links for category: {category}")
        metadata_name = f"{category}_000.zip"
        data_urls = [u for u in urls if u.rsplit("/", 1)[-1] != metadata_name]

        print(
            f"[{category}] indexing {len(data_urls)} remote archives for "
            f"{len(required_paths)} required files",
            flush=True,
        )
        found = find_required_members(
            data_urls,
            required_paths,
            archive_opener=opener,
            progress=lambda msg: print(f"  {msg}", flush=True),
        )
        sources = {}
        for relative, source in found.items():
            archive_sha256 = full_checksums.get(source["archive"])
            if not archive_sha256:
                raise ValueError(f"Missing official archive SHA-256 for {source['archive']}")
            sources[relative] = {**source, "official_archive_sha256": archive_sha256}

        print(f"[{category}] downloading {len(sources)} verified files", flush=True)
        records = materialize_required_members(
            sources,
            dataset_root=root,
            archive_opener=opener,
            progress=lambda msg: print(f"  {msg}", flush=True),
        )
        category_bytes = sum(r["size"] for r in records)
        grand_total_bytes += category_bytes
        downloaded = sum(r["status"] == "downloaded" for r in records)
        already = sum(r["status"] == "verified_existing" for r in records)
        print(
            f"[{category}] done: {downloaded} downloaded, {already} already present, "
            f"{category_bytes / 1e6:.1f} MB",
            flush=True,
        )

    print(f"TOTAL: {grand_total_bytes / 1e9:.2f} GB across {len(categories)} categories", flush=True)


if __name__ == "__main__":
    main()
