import pandas as pd


def normalize_address(series):
    return (
        series
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "", regex=True)
    )


print("Loading S1...")
s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    usecols=["entity_id", "business_address"],
)

print("Loading S2...")
s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    usecols=["entity_id", "business_address"],
)

print("Loading S3...")
s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    usecols=["entity_id", "business_address"],
)

print("Loading ground truth...")
gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t",
)

print("Normalizing addresses...")

s1["norm_address"] = normalize_address(s1["business_address"])
s2["norm_address"] = normalize_address(s2["business_address"])
s3["norm_address"] = normalize_address(s3["business_address"])

s1_addresses = dict(zip(s1["entity_id"], s1["norm_address"]))
s2_addresses = dict(zip(s2["entity_id"], s2["norm_address"]))
s3_addresses = dict(zip(s3["entity_id"], s3["norm_address"]))

total_matches = 0
exact_address_matches = 0

print("Checking ground truth...")

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]
    matched_ids = str(row["matched_entity_ids"])

    if matched_ids == "nan" or not matched_ids.strip():
        continue

    s1_address = s1_addresses[s1_id]

    for matched_id in matched_ids.split(","):

        matched_id = matched_id.strip()

        if matched_id.startswith("S2-"):
            target_address = s2_addresses.get(matched_id, "")
        else:
            target_address = s3_addresses.get(matched_id, "")

        total_matches += 1

        if (
            s1_address
            and target_address
            and s1_address == target_address
        ):
            exact_address_matches += 1


print()
print("=" * 60)
print("EXACT NORMALIZED ADDRESS RECALL")
print("=" * 60)

print("Total ground-truth matched IDs:", total_matches)
print("Exact normalized-address matches:", exact_address_matches)

recall = exact_address_matches / total_matches

print("Recall:", recall)
print("Recall percentage:", f"{recall * 100:.2f}%")