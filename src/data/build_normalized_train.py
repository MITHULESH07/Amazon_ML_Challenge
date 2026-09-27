"""Build normalized training-source TSV files in bounded-memory chunks."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from normalize import iter_normalized_tsv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "normalized"

SOURCE_PATHS = {
    "source1": (
        TRAIN_DIR / "train_source1.tsv",
        OUTPUT_DIR / "source1_normal.tsv",
    ),
    "source2": (
        TRAIN_DIR / "train_source2.tsv",
        OUTPUT_DIR / "source2_normal.tsv",
    ),
    "source3": (
        TRAIN_DIR / "train_source3.tsv",
        OUTPUT_DIR / "source3_normal.tsv",
    ),
}


def build_normalized_source(source: str, chunksize: int) -> None:
    """Normalize one training source and atomically publish its TSV output."""

    input_path, output_path = SOURCE_PATHS[source]
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}. Move or remove it explicitly "
            "before rebuilding."
        )
    if temporary_path.exists():
        raise FileExistsError(
            f"Temporary output already exists: {temporary_path}. It may be from an "
            "interrupted run; inspect it before removing it."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows_written = 0

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Chunk size: {chunksize:,} rows")

    try:
        for chunk_number, normalized_chunk in enumerate(
            iter_normalized_tsv(input_path, chunksize=chunksize),
            start=1,
        ):
            normalized_chunk.to_csv(
                temporary_path,
                sep="\t",
                index=False,
                encoding="utf-8",
                mode="w" if chunk_number == 1 else "a",
                header=chunk_number == 1,
                lineterminator="\n",
            )
            rows_written += len(normalized_chunk)
            elapsed = time.monotonic() - started
            print(
                f"Chunk {chunk_number}: {rows_written:,} rows written "
                f"({elapsed / 60:.1f} minutes elapsed)",
                flush=True,
            )

        if rows_written == 0:
            raise ValueError(f"Input file contains no data rows: {input_path}")

        os.replace(temporary_path, output_path)
    except Exception:
        print(
            f"Normalization stopped. Any partial data remains at: {temporary_path}"
        )
        raise

    elapsed = time.monotonic() - started
    print(f"Completed {source}: {rows_written:,} rows in {elapsed / 60:.1f} minutes")
    print(f"Created: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize one Amazon ML Challenge training source."
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=tuple(SOURCE_PATHS),
        help="Training source to normalize.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Rows processed per chunk (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be greater than zero")
    build_normalized_source(args.source, args.chunksize)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
