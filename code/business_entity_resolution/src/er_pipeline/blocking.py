"""Scalable SQL blocking followed by bounded fuzzy candidate pruning."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from .features import retrieval_similarity
from .storage import connect


def _insert_block(connection: sqlite3.Connection, condition: str, label: str) -> None:
    before = connection.total_changes
    connection.execute(
        f"""
        INSERT OR IGNORE INTO candidate_raw(s1_id, target_id)
        SELECT a.entity_id, b.entity_id
        FROM records a
        JOIN records b ON b.source IN (2,3)
          AND a.country = b.country
          AND {condition}
        WHERE a.source = 1
        """
    )
    connection.commit()
    print(f"  {label}: {connection.total_changes - before:,} new pairs", flush=True)


def generate_candidates(
    database_path: Path,
    *,
    top_k_per_source: int = 10,
    max_approximate_bucket: int = 60,
    minimum_retrieval_score: float = 0.42,
) -> None:
    connection = connect(database_path)
    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='candidates'"
    ).fetchone():
        print(f"Reusing existing candidates in {database_path}")
        connection.close()
        return

    started = time.monotonic()
    connection.execute(
        "CREATE TABLE candidate_raw (s1_id TEXT NOT NULL, target_id TEXT NOT NULL, "
        "PRIMARY KEY(s1_id,target_id)) WITHOUT ROWID"
    )
    connection.executescript(
        """
        CREATE TEMP TABLE core_counts AS
            SELECT country, name_core, count(*) AS n FROM records
            WHERE source IN (2,3) AND name_core <> '' GROUP BY country,name_core;
        CREATE INDEX temp.idx_core_counts ON core_counts(country,name_core);
        CREATE TEMP TABLE signature_counts AS
            SELECT country,name_signature,count(*) AS n FROM records
            WHERE source IN (2,3) AND name_signature <> ''
            GROUP BY country,name_signature;
        CREATE INDEX temp.idx_signature_counts
            ON signature_counts(country,name_signature);
        CREATE TEMP TABLE prefix_counts AS
            SELECT country, name_prefix, count(*) AS n FROM records
            WHERE source IN (2,3) AND name_prefix <> '' GROUP BY country, name_prefix;
        CREATE INDEX temp.idx_prefix_counts ON prefix_counts(country,name_prefix);
        CREATE TEMP TABLE anchor_counts AS
            SELECT country, name_anchor, count(*) AS n FROM records
            WHERE source IN (2,3) AND name_anchor <> '' GROUP BY country,name_anchor;
        CREATE INDEX temp.idx_anchor_counts ON anchor_counts(country,name_anchor);
        CREATE TEMP TABLE address_number_counts AS
            SELECT country,address_anchor,numbers,count(*) AS n FROM records
            WHERE source IN (2,3) AND address_anchor <> '' AND numbers <> ''
            GROUP BY country,address_anchor,numbers;
        CREATE INDEX temp.idx_address_number_counts
            ON address_number_counts(country,address_anchor,numbers);
        """
    )
    print("Generating exact and deterministic candidate blocks...", flush=True)
    _insert_block(connection, "a.name <> '' AND a.name = b.name", "exact name")
    _insert_block(
        connection,
        f"a.name_core <> '' AND a.name_core = b.name_core AND EXISTS "
        f"(SELECT 1 FROM core_counts c WHERE c.country=a.country "
        f"AND c.name_core=a.name_core AND c.n<={max_approximate_bucket})",
        "bounded core name",
    )
    _insert_block(connection, "a.address <> '' AND a.address = b.address", "exact address")
    _insert_block(
        connection,
        "a.name_signature <> '' AND a.name_signature = b.name_signature AND EXISTS "
        f"(SELECT 1 FROM signature_counts c WHERE c.country=a.country "
        f"AND c.name_signature=a.name_signature AND c.n<={max_approximate_bucket})",
        "bounded name token signature",
    )

    _insert_block(
        connection,
        f"a.name_prefix <> '' AND a.name_prefix = b.name_prefix AND EXISTS "
        f"(SELECT 1 FROM prefix_counts c WHERE c.country=a.country "
        f"AND c.name_prefix=a.name_prefix AND c.n<={max_approximate_bucket})",
        "bounded name prefix",
    )
    _insert_block(
        connection,
        f"a.name_anchor <> '' AND a.name_anchor = b.name_anchor AND EXISTS "
        f"(SELECT 1 FROM anchor_counts c WHERE c.country=a.country "
        f"AND c.name_anchor=a.name_anchor AND c.n<={max_approximate_bucket})",
        "bounded longest-name token",
    )
    _insert_block(
        connection,
        "a.address_anchor <> '' AND a.address_anchor = b.address_anchor "
        "AND a.numbers <> '' AND a.numbers = b.numbers AND EXISTS "
        f"(SELECT 1 FROM address_number_counts c WHERE c.country=a.country "
        f"AND c.address_anchor=a.address_anchor AND c.numbers=a.numbers "
        f"AND c.n<={max_approximate_bucket})",
        "bounded address anchor plus numbers",
    )

    connection.execute(
        "CREATE TABLE candidates (s1_id TEXT NOT NULL, target_id TEXT NOT NULL, "
        "retrieval_score REAL NOT NULL, PRIMARY KEY(s1_id,target_id)) WITHOUT ROWID"
    )
    query = """
        SELECT c.s1_id, c.target_id,
               a.entity_id,a.source,a.country,a.name,a.name_core,a.address,a.numbers,
               a.name_signature,a.name_prefix,a.name_anchor,a.address_anchor,
               b.entity_id,b.source,b.country,b.name,b.name_core,b.address,b.numbers,
               b.name_signature,b.name_prefix,b.name_anchor,b.address_anchor
        FROM candidate_raw c
        JOIN records a ON a.entity_id=c.s1_id
        JOIN records b ON b.entity_id=c.target_id
        ORDER BY c.s1_id
    """
    print("Pruning candidates with fuzzy similarities...", flush=True)
    current_s1 = None
    group: list[tuple] = []
    written = 0

    def flush(rows: list[tuple]) -> int:
        if not rows:
            return 0
        ranked = {2: [], 3: []}
        for row in rows:
            left, right = row[2:13], row[13:24]
            score = retrieval_similarity(left, right)
            if score >= minimum_retrieval_score:
                ranked[right[1]].append((score, row[1]))
        selected = []
        for source_rows in ranked.values():
            source_rows.sort(reverse=True)
            selected.extend(source_rows[:top_k_per_source])
        connection.executemany(
            "INSERT INTO candidates VALUES (?,?,?)",
            [(rows[0][0], target, score) for score, target in selected],
        )
        return len(selected)

    for row in connection.execute(query):
        if current_s1 is not None and row[0] != current_s1:
            written += flush(group)
            group = []
            if written and written % 500_000 < top_k_per_source * 2:
                connection.commit()
                print(f"  {written:,} final candidates", flush=True)
        current_s1 = row[0]
        group.append(row)
    written += flush(group)
    connection.commit()
    connection.executescript(
        "DROP TABLE candidate_raw;"
        "CREATE INDEX idx_candidates_s1 ON candidates(s1_id);"
        "CREATE INDEX idx_candidates_target ON candidates(target_id);"
    )
    connection.commit()
    connection.close()
    print(
        f"Candidate generation finished: {written:,} pairs in "
        f"{(time.monotonic() - started) / 60:.1f} minutes"
    )


def finish_interrupted_pruning(
    database_path: Path,
    *,
    top_k_per_source: int = 10,
    minimum_retrieval_score: float = 0.42,
) -> None:
    """Rebuild final candidates from an already-complete raw blocking table.

    This is intentionally separate from blocking so an interrupted pruning pass
    can be restarted without repeating the expensive normalized-key joins.
    """

    connection = connect(database_path)
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if "candidate_raw" not in tables:
        if "candidates" in tables:
            print("Final candidates are already complete; nothing to resume.")
            connection.close()
            return
        connection.close()
        raise RuntimeError("No candidate_raw table is available to prune")

    # A candidates table alongside candidate_raw is necessarily incomplete: the
    # normal successful path drops candidate_raw only after pruning and indexing.
    connection.execute("DROP TABLE IF EXISTS candidates")
    connection.execute(
        "CREATE TABLE candidates (s1_id TEXT NOT NULL,target_id TEXT NOT NULL,"
        "retrieval_score REAL NOT NULL,PRIMARY KEY(s1_id,target_id)) WITHOUT ROWID"
    )
    connection.commit()

    query = """
        SELECT c.s1_id,c.target_id,
               a.entity_id,a.source,a.country,a.name,a.name_core,a.address,a.numbers,
               a.name_signature,a.name_prefix,a.name_anchor,a.address_anchor,
               b.entity_id,b.source,b.country,b.name,b.name_core,b.address,b.numbers,
               b.name_signature,b.name_prefix,b.name_anchor,b.address_anchor
        FROM candidate_raw c
        JOIN records a ON a.entity_id=c.s1_id
        JOIN records b ON b.entity_id=c.target_id
        ORDER BY c.s1_id
    """
    started = time.monotonic()
    current_s1 = None
    group: list[tuple] = []
    written = 0
    next_report = 500_000

    def flush(rows: list[tuple]) -> int:
        if not rows:
            return 0
        ranked = {2: [], 3: []}
        for row in rows:
            left, right = row[2:13], row[13:24]
            score = retrieval_similarity(left, right)
            if score >= minimum_retrieval_score:
                ranked[right[1]].append((score, row[1]))
        selected = []
        for source_rows in ranked.values():
            source_rows.sort(reverse=True)
            selected.extend(source_rows[:top_k_per_source])
        connection.executemany(
            "INSERT INTO candidates VALUES (?,?,?)",
            [(rows[0][0], target, score) for score, target in selected],
        )
        return len(selected)

    print("Restarting final fuzzy candidate pruning...", flush=True)
    for row in connection.execute(query):
        if current_s1 is not None and row[0] != current_s1:
            written += flush(group)
            group = []
            if written >= next_report:
                connection.commit()
                print(f"  {written:,} final candidates", flush=True)
                next_report = ((written // 500_000) + 1) * 500_000
        current_s1 = row[0]
        group.append(row)
    written += flush(group)
    connection.commit()
    connection.executescript(
        "DROP TABLE candidate_raw;"
        "CREATE INDEX idx_candidates_s1 ON candidates(s1_id);"
        "CREATE INDEX idx_candidates_target ON candidates(target_id);"
    )
    connection.commit()
    connection.close()
    print(
        f"Test candidate generation complete: {written:,} final pairs in "
        f"{(time.monotonic() - started) / 60:.1f} minutes"
    )

