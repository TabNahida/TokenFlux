#!/usr/bin/env python3
import argparse
import json
import sys


def emit(field: str, value) -> None:
    if value is None:
        return
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)
    if not text:
        return
    sys.stdout.write(json.dumps({field: text}, ensure_ascii=False))
    sys.stdout.write("\n")


def stream_with_pyarrow(path: str, field: str, batch_size: int) -> None:
    import pyarrow.parquet as pq  # type: ignore

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=[field]):
        column = batch.column(0)
        for value in column.to_pylist():
            emit(field, value)


def stream_with_pandas(path: str, field: str) -> None:
    import pandas as pd  # type: ignore

    frame = pd.read_parquet(path, columns=[field])
    for value in frame[field].tolist():
        emit(field, value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream parquet column as JSONL lines.")
    parser.add_argument("--path", required=True, help="parquet file path")
    parser.add_argument("--field", required=True, help="text field")
    parser.add_argument("--batch-size", type=int, default=4096, help="pyarrow batch size")
    args = parser.parse_args()

    try:
        stream_with_pyarrow(args.path, args.field, args.batch_size)
    except Exception as pyarrow_error:
        try:
            stream_with_pandas(args.path, args.field)
        except Exception as pandas_error:
            sys.stderr.write(
                f"failed to read parquet with pyarrow ({pyarrow_error}) and pandas ({pandas_error})\n"
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
