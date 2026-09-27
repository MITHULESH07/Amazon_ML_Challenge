"""Command-line entry point for the complete entity-resolution pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from er_pipeline.blocking import finish_interrupted_pruning, generate_candidates
from er_pipeline.modeling import score_candidates, select_threshold, train_model
from er_pipeline.outputs import write_submission
from er_pipeline.storage import build_database


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET = PROJECT_ROOT / "dataset"
ARTIFACTS = PROJECT_ROOT / "artifacts"
MODELS = PROJECT_ROOT / "models"
OUTPUT = PROJECT_ROOT / "output"
TRAIN_DB = ARTIFACTS / "train.sqlite"
TEST_DB = ARTIFACTS / "test.sqlite"
MODEL_PATH = MODELS / "xgboost_entity_matcher.joblib"


def build_train() -> None:
    build_database(
        TRAIN_DB,
        (
            DATASET / "train" / "train_source1.tsv",
            DATASET / "train" / "train_source2.tsv",
            DATASET / "train" / "train_source3.tsv",
        ),
        DATASET / "train" / "train_ground_truth.tsv",
    )
    generate_candidates(TRAIN_DB)


def fit() -> dict:
    build_train()
    artifact = train_model(TRAIN_DB, MODEL_PATH)
    return select_threshold(TRAIN_DB, artifact, MODEL_PATH)


def build_test() -> None:
    build_database(
        TEST_DB,
        (
            DATASET / "test" / "test_source1.tsv",
            DATASET / "test" / "test_source2.tsv",
            DATASET / "test" / "test_source3.tsv",
        ),
    )
    generate_candidates(TEST_DB)


def predict() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Trained model not found; run the train stage first")
    artifact = joblib.load(MODEL_PATH)
    if artifact.get("threshold") is None:
        raise RuntimeError("Model threshold has not been selected")
    build_test()
    score_candidates(TEST_DB, artifact, validation_only=False)
    write_submission(TEST_DB, OUTPUT, artifact["threshold"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=(
            "build-train", "train", "build-test", "finish-test-candidates",
            "predict", "all",
        ),
        help="Pipeline stage to execute.",
    )
    return parser.parse_args()


def main() -> int:
    stage = parse_args().stage
    if stage == "build-train":
        build_train()
    elif stage == "train":
        fit()
    elif stage == "build-test":
        build_test()
    elif stage == "finish-test-candidates":
        finish_interrupted_pruning(TEST_DB)
    elif stage == "predict":
        predict()
    else:
        fit()
        predict()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

