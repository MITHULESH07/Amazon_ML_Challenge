"""Submission TSV construction from scored test candidates."""

from __future__ import annotations

import csv
from pathlib import Path

from .storage import connect


def _write_grouped(
    connection,
    output_path: Path,
    value_header: str,
    join_table: str,
    join_condition: str,
) -> None:
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    query = f"""
        SELECT r.entity_id, COALESCE(group_concat(x.target_id, ','), '')
        FROM records r
        LEFT JOIN {join_table} x ON x.s1_id=r.entity_id {join_condition}
        WHERE r.source=1
        GROUP BY r.entity_id
        ORDER BY r.entity_id
    """
    count = 0
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", value_header])
        for row in connection.execute(query):
            writer.writerow(row)
            count += 1
            if count % 500_000 == 0:
                print(f"  {output_path.name}: {count:,} rows", flush=True)
    temporary.replace(output_path)
    print(f"Created {output_path} ({count:,} rows)")


def write_submission(database_path: Path, output_dir: Path, threshold: float) -> None:
    connection = connect(database_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_grouped(
        connection,
        output_dir / "matching_results.tsv",
        "matched_entity_ids",
        "predictions",
        f"AND x.probability >= {float(threshold):.12g}",
    )
    _write_grouped(
        connection,
        output_dir / "candidate_pairs.tsv",
        "candidate_entity_ids",
        "candidates",
        "",
    )
    connection.close()

