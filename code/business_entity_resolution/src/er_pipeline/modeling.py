"""Model fitting, held-out threshold selection, and candidate scoring."""

from __future__ import annotations

import random
import sqlite3
import time
from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBClassifier

from .features import FEATURE_NAMES, pair_features
from .metrics import entity_f05
from .storage import connect, fetch_truths, stable_fold


PAIR_QUERY = """
    SELECT c.s1_id,c.target_id,c.retrieval_score,
           a.entity_id,a.source,a.country,a.name,a.name_core,a.address,a.numbers,
           a.name_signature,a.name_prefix,a.name_anchor,a.address_anchor,
           b.entity_id,b.source,b.country,b.name,b.name_core,b.address,b.numbers,
           b.name_signature,b.name_prefix,b.name_anchor,b.address_anchor
    FROM candidates c
    JOIN records a ON a.entity_id=c.s1_id
    JOIN records b ON b.entity_id=c.target_id
"""

TRAIN_PAIR_QUERY = """
    SELECT c.s1_id,c.target_id,c.retrieval_score,
           a.entity_id,a.source,a.country,a.name,a.name_core,a.address,a.numbers,
           a.name_signature,a.name_prefix,a.name_anchor,a.address_anchor,
           b.entity_id,b.source,b.country,b.name,b.name_core,b.address,b.numbers,
           b.name_signature,b.name_prefix,b.name_anchor,b.address_anchor,
           CASE WHEN t.target_id IS NULL THEN 0 ELSE 1 END
    FROM candidates c
    JOIN records a ON a.entity_id=c.s1_id
    JOIN records b ON b.entity_id=c.target_id
    LEFT JOIN truth t ON t.s1_id=c.s1_id AND t.target_id=c.target_id
"""


class Reservoir:
    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.random = random.Random(seed)
        self.items: list[tuple[list[float], int]] = []
        self.seen = 0

    def add(self, item: tuple[list[float], int]) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        index = self.random.randrange(self.seen)
        if index < self.capacity:
            self.items[index] = item


def _feature_row(row: tuple) -> list[float]:
    return pair_features(row[3:14], row[14:25], row[2])


def candidate_recall(connection: sqlite3.Connection) -> tuple[int, int, float]:
    total = connection.execute("SELECT count(*) FROM truth").fetchone()[0]
    found = connection.execute(
        "SELECT count(*) FROM truth t JOIN candidates c "
        "ON c.s1_id=t.s1_id AND c.target_id=t.target_id"
    ).fetchone()[0]
    return found, total, found / total if total else 0.0


def train_model(
    database_path: Path,
    model_path: Path,
    *,
    positive_sample: int = 500_000,
    negative_sample: int = 1_000_000,
) -> dict:
    if model_path.exists():
        print(f"Reusing existing model: {model_path}")
        return joblib.load(model_path)

    connection = connect(database_path)
    found, total, recall = candidate_recall(connection)
    print(f"Training candidate recall: {found:,}/{total:,} = {recall:.6f}")
    positives = Reservoir(positive_sample, 41)
    negatives = Reservoir(negative_sample, 42)
    print("Sampling training pair features...", flush=True)
    processed = 0
    for row in connection.execute(TRAIN_PAIR_QUERY):
        if stable_fold(row[0]) == 0:
            continue
        label = int(row[25])
        item = (_feature_row(row), label)
        (positives if label else negatives).add(item)
        processed += 1
        if processed % 1_000_000 == 0:
            print(f"  {processed:,} training candidates scanned", flush=True)

    sampled = positives.items + negatives.items
    if not positives.items or not negatives.items:
        raise RuntimeError("Both positive and negative candidate pairs are required")
    rng = random.Random(43)
    rng.shuffle(sampled)
    x_train = np.asarray([item[0] for item in sampled], dtype=np.float32)
    y_train = np.asarray([item[1] for item in sampled], dtype=np.uint8)
    scale = max(1.0, float((y_train == 0).sum() / (y_train == 1).sum()))
    print(
        f"Fitting XGBoost on {len(y_train):,} sampled pairs "
        f"({int(y_train.sum()):,} positive)...",
        flush=True,
    )
    model = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", n_estimators=350,
        max_depth=6, learning_rate=0.06, subsample=0.8, colsample_bytree=0.9,
        min_child_weight=3, reg_lambda=2.0, tree_method="hist",
        scale_pos_weight=scale, random_state=42, n_jobs=-1,
    )
    model.fit(x_train, y_train)
    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "candidate_recall": recall,
        "threshold": None,
        "validation_f05": None,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    connection.close()
    return artifact


def score_candidates(
    database_path: Path,
    artifact: dict,
    *,
    validation_only: bool,
    batch_size: int = 100_000,
) -> None:
    connection = connect(database_path)
    table = "validation_predictions" if validation_only else "predictions"
    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone():
        print(f"Reusing existing {table}")
        connection.close()
        return
    connection.execute(
        f"CREATE TABLE {table} (s1_id TEXT NOT NULL,target_id TEXT NOT NULL,"
        "probability REAL NOT NULL,PRIMARY KEY(s1_id,target_id)) WITHOUT ROWID"
    )
    connection.create_function("stable_fold", 1, stable_fold, deterministic=True)
    query = PAIR_QUERY
    if validation_only:
        query += " WHERE stable_fold(c.s1_id)=0"
    rows: list[tuple] = []
    ids: list[tuple[str, str]] = []
    written = 0

    def flush() -> None:
        nonlocal written
        if not rows:
            return
        matrix = np.asarray([_feature_row(row) for row in rows], dtype=np.float32)
        probabilities = artifact["model"].predict_proba(matrix)[:, 1]
        connection.executemany(
            f"INSERT INTO {table} VALUES (?,?,?)",
            [(pair[0], pair[1], float(prob)) for pair, prob in zip(ids, probabilities)],
        )
        connection.commit()
        written += len(rows)
        print(f"  {written:,} candidates scored", flush=True)
        rows.clear()
        ids.clear()

    for row in connection.execute(query):
        rows.append(row)
        ids.append((row[0], row[1]))
        if len(rows) >= batch_size:
            flush()
    flush()
    connection.execute(f"CREATE INDEX idx_{table}_s1 ON {table}(s1_id)")
    connection.commit()
    connection.close()


def select_threshold(database_path: Path, artifact: dict, model_path: Path) -> dict:
    score_candidates(database_path, artifact, validation_only=True)
    connection = connect(database_path)
    truths = fetch_truths(connection, validation_only=True)
    thresholds = np.asarray(
        [0.30, 0.40, 0.50, 0.60, 0.68, 0.74, 0.78, 0.82, 0.86,
         0.89, 0.91, 0.93, 0.95, 0.97, 0.985],
        dtype=np.float32,
    )
    sums = np.asarray(
        [sum(entity_f05(truth, set()) for truth in truths.values())] * len(thresholds),
        dtype=np.float64,
    )
    current = None
    group: list[tuple[str, float]] = []

    def apply_group(s1_id: str | None, values: list[tuple[str, float]]) -> None:
        if s1_id is None or s1_id not in truths:
            return
        empty_score = entity_f05(truths[s1_id], set())
        for index, threshold in enumerate(thresholds):
            predicted = {target for target, prob in values if prob >= threshold}
            sums[index] += entity_f05(truths[s1_id], predicted) - empty_score

    for s1_id, target_id, probability in connection.execute(
        "SELECT s1_id,target_id,probability FROM validation_predictions ORDER BY s1_id"
    ):
        if current is not None and s1_id != current:
            apply_group(current, group)
            group = []
        current = s1_id
        group.append((target_id, probability))
    apply_group(current, group)
    scores = sums / len(truths)
    best_index = int(np.argmax(scores))
    artifact["threshold"] = float(thresholds[best_index])
    artifact["validation_f05"] = float(scores[best_index])
    artifact["threshold_scores"] = {
        float(t): float(s) for t, s in zip(thresholds, scores)
    }
    joblib.dump(artifact, model_path)
    print("Threshold search:")
    for threshold, score in zip(thresholds, scores):
        print(f"  {threshold:.3f}: macro F0.5={score:.6f}")
    print(
        f"Selected threshold {artifact['threshold']:.3f}; held-out macro F0.5 "
        f"{artifact['validation_f05']:.6f}"
    )
    connection.close()
    return artifact

