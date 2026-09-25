import pandas as pd
import re

files = [
    ("S1", "dataset/train/train_source1.tsv"),
    ("S2", "dataset/train/train_source2.tsv"),
    ("S3", "dataset/train/train_source3.tsv"),
]

for name, path in files:
    df = pd.read_csv(path, sep="\t", usecols=["business_name"])

    normalized = (
        df["business_name"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "", regex=True)
    )

    non_empty = normalized[normalized != ""]

    print("=" * 60)
    print(name)
    print("=" * 60)
    print("Total:", len(normalized))
    print("Non-empty:", len(non_empty))
    print("Unique normalized names:", non_empty.nunique())
    print("Duplicate normalized names:", non_empty.duplicated().sum())