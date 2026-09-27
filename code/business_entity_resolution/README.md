# Business Entity Resolution

This directory contains the complete, reproducible Amazon ML Challenge 2026
pipeline. It uses only the provided challenge data.

## Environment

From the repository root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r code\business_entity_resolution\requirements.txt
```

## End-to-end run

```powershell
python code\business_entity_resolution\src\run_pipeline.py all
```

The run creates disk-backed intermediate databases under `artifacts/`, the model
under `models/`, and these final files:

- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

Stages can also be resumed individually:

```powershell
python code\business_entity_resolution\src\run_pipeline.py build-train
python code\business_entity_resolution\src\run_pipeline.py train
python code\business_entity_resolution\src\run_pipeline.py build-test
python code\business_entity_resolution\src\run_pipeline.py predict
```

Existing completed SQLite/model stages are reused. Interrupted `.tmp` database
files are never silently accepted; inspect and remove them before restarting.

## Validation

```powershell
python utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir dataset\test --check-ids
```

The pipeline reports candidate recall and held-out entity-level macro F0.5 on a
deterministic Source-1 split. The test score cannot be calculated locally because
test ground truth is not provided.

## Method

Records are normalized with Unicode NFKC, case folding, conservative punctuation
handling, legal-suffix variants, address abbreviation normalization, and numeric
token extraction. SQLite indexes generate exact and bounded approximate blocks.
Candidates are pruned separately per target source with lexical name/address and
number evidence. XGBoost scores compact pair features. A threshold is selected on
held-out Source-1 entities using the official macro entity-level F0.5 metric.

