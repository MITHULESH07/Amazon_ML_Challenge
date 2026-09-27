"""Chunked TSV ingestion and disk-backed record storage."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pandas as pd

from .normalization import normalize_record


RECORD_COLUMNS = (
    "entity_id", "source", "country", "name", "name_core", "address",
    "numbers", "name_signature", "name_prefix", "name_anchor", "address_anchor",
)


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=120)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-262144")
    return connection


def _schema(connection: sqlite3.Connection, include_truth: bool) -> None:
    connection.executescript(
        """
        CREATE TABLE records (
            entity_id TEXT PRIMARY KEY,
            source INTEGER NOT NULL,
            country TEXT NOT NULL,
            name TEXT NOT NULL,
            name_core TEXT NOT NULL,
            address TEXT NOT NULL,
            numbers TEXT NOT NULL,
            name_signature TEXT NOT NULL,
            name_prefix TEXT NOT NULL,
            name_anchor TEXT NOT NULL,
            address_anchor TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )
    if include_truth:
        connection.execute(
            "CREATE TABLE truth (s1_id TEXT NOT NULL, target_id TEXT NOT NULL, "
            "PRIMARY KEY (s1_id, target_id)) WITHOUT ROWID"
        )


def _record_rows(chunk: pd.DataFrame, source: int):
    for row in chunk.itertuples(index=False):
        normalized = normalize_record(row.business_name, row.business_address, row.country)
        yield (row.entity_id, source, normalized[4], normalized[0], normalized[1],
               normalized[2], normalized[3], normalized[5], normalized[6],
               normalized[7], normalized[8])


def build_database(
    database_path: Path,
    source_paths: tuple[Path, Path, Path],
    ground_truth_path: Path | None = None,
    chunksize: int = 100_000,
) -> None:
    if database_path.exists():
        print(f"Reusing existing database: {database_path}")
        return

    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = database_path.with_suffix(database_path.suffix + ".tmp")
    if temporary_path.exists():
        raise FileExistsError(f"Inspect/remove interrupted database: {temporary_path}")

    connection = connect(temporary_path)
    _schema(connection, ground_truth_path is not None)
    started = time.monotonic()
    try:
        for source, path in enumerate(source_paths, start=1):
            count = 0
            for chunk in pd.read_csv(
                path, sep="\t", dtype=str, keep_default_na=False,
                usecols=["entity_id", "business_name", "business_address", "country"],
                chunksize=chunksize,
            ):
                rows = list(_record_rows(chunk, source))
                connection.executemany(
                    "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
                )
                connection.commit()
                count += len(rows)
                print(f"  source{source}: {count:,} normalized", flush=True)

        if ground_truth_path is not None:
            count = 0
            for chunk in pd.read_csv(
                ground_truth_path, sep="\t", dtype=str, keep_default_na=False,
                chunksize=chunksize,
            ):
                rows = []
                for row in chunk.itertuples(index=False):
                    if row.matched_entity_ids:
                        rows.extend(
                            (row.source1_entity_id, target.strip())
                            for target in row.matched_entity_ids.split(",") if target.strip()
                        )
                connection.executemany("INSERT OR IGNORE INTO truth VALUES (?,?)", rows)
                connection.commit()
                count += len(chunk)
                print(f"  ground truth: {count:,} S1 rows", flush=True)

        print("Creating candidate lookup indexes...", flush=True)
        connection.executescript(
            """
            CREATE INDEX idx_records_source_name
                ON records(source, country, name);
            CREATE INDEX idx_records_source_core
                ON records(source, country, name_core);
            CREATE INDEX idx_records_source_address
                ON records(source, country, address);
            CREATE INDEX idx_records_source_signature
                ON records(source, country, name_signature);
            CREATE INDEX idx_records_source_prefix
                ON records(source, country, name_prefix);
            CREATE INDEX idx_records_source_name_anchor
                ON records(source, country, name_anchor);
            CREATE INDEX idx_records_source_address_anchor
                ON records(source, country, address_anchor, numbers);
            """
        )
        connection.commit()
    except Exception:
        connection.close()
        raise
    else:
        connection.close()
        temporary_path.replace(database_path)
        print(
            f"Database built in {(time.monotonic() - started) / 60:.1f} minutes: "
            f"{database_path}"
        )


def fetch_truths(connection: sqlite3.Connection, validation_only: bool = False):
    truths: dict[str, set[str]] = {}
    for (s1_id,) in connection.execute(
        "SELECT entity_id FROM records WHERE source=1 ORDER BY entity_id"
    ):
        if not validation_only or stable_fold(s1_id) == 0:
            truths[s1_id] = set()
    for s1_id, target_id in connection.execute("SELECT s1_id, target_id FROM truth"):
        if s1_id in truths:
            truths[s1_id].add(target_id)
    return truths


def stable_fold(entity_id: str, folds: int = 5) -> int:
    # Entity IDs end in decimal digits; using them gives a deterministic split.
    try:
        return int(entity_id.rsplit("-", 1)[-1]) % folds
    except ValueError:
        import zlib
        return zlib.crc32(entity_id.encode("utf-8")) % folds

