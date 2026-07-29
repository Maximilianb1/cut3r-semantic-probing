from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet_atomic(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Refusing to write an empty table: {destination}")
    table = pa.Table.from_pylist(materialized)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def read_parquet(path: str | Path) -> list[dict[str, Any]]:
    return pq.read_table(Path(path)).to_pylist()
