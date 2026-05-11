"""
src/fetch_data.py
──────────────────
Hugging Face dataset indirme (opsiyonel).

Amaç: Çok büyük datasetleri RAM'e almadan `data/raw/` altına Parquet olarak yazmak.

Not:
- Bu script uygulama runtime'ında kullanılmaz (Streamlit Cloud için şart değil).
- `datasets` ve `pyarrow` gerektirir. (pyarrow zaten requirements'ta var)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _write_parquet_batches(rows: Iterable[dict], out_path: Path, batch_size: int = 50_000) -> None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    buf: list[dict] = []

    def flush() -> None:
        nonlocal writer, buf
        if not buf:
            return
        df = pd.DataFrame.from_records(buf)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)
        buf = []

    for r in rows:
        buf.append(r)
        if len(buf) >= batch_size:
            flush()
    flush()
    if writer is not None:
        writer.close()


def fetch_and_save_data(
    dataset_name: str = "AIcell/moltbook-data",
    split: str = "train",
    out_file: str = "raw_data.parquet",
    limit: int | None = None,
) -> Path:
    """
    Streaming ile dataset'i indirir ve Parquet yazar.
    """
    from datasets import load_dataset  # type: ignore

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / out_file

    ds = load_dataset(dataset_name, split=split, streaming=True)

    def row_iter():
        n = 0
        for ex in ds:
            yield dict(ex)
            n += 1
            if limit is not None and n >= limit:
                break

    _write_parquet_batches(row_iter(), out_path=out_path)
    return out_path


if __name__ == "__main__":
    p = fetch_and_save_data()
    print(f"Saved: {p}")

