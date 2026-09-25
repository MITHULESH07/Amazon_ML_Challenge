import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

files = [
    ROOT / "dataset/train/train_source1.tsv",
    ROOT / "dataset/train/train_source2.tsv",
    ROOT / "dataset/train/train_source3.tsv",
    ROOT / "dataset/train/train_ground_truth.tsv",
    ROOT / "dataset/test/test_source1.tsv",
    ROOT / "dataset/test/test_source2.tsv",
    ROOT / "dataset/test/test_source3.tsv",
]

for path in files:
    print("\n" + "=" * 70)
    print(path.relative_to(ROOT))
    print("=" * 70)

    df = pd.read_csv(path, sep="\t")

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    if "country" in df.columns:
        print("\nCountries:")
        print(df["country"].value_counts(dropna=False).to_string())

    if "entity_id" in df.columns:
        print("\nID prefix:")
        print(
            df["entity_id"]
            .astype(str)
            .str.extract(r"^(S\d-)", expand=False)
            .value_counts()
            .to_string()
        )