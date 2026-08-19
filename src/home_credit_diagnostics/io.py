from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a parquet or CSV table without changing its schema."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source, columns=columns)
    if suffix in {".csv", ".txt"}:
        frame = pd.read_csv(source, usecols=columns)
        return frame
    raise ValueError(f"Unsupported input format {suffix!r}; use parquet or CSV.")


def restore_named_index_columns(frame: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Restore columns such as case_id when parquet preserved them as an index."""
    expected_lower = {name.lower() for name in expected}
    index_names = [name for name in frame.index.names if name is not None]
    if any(str(name).lower() in expected_lower for name in index_names):
        return frame.reset_index()
    return frame


def write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    """Write deterministic, index-free CSV output."""
    frame.to_csv(Path(path), index=False, float_format="%.10g", lineterminator="\n")
