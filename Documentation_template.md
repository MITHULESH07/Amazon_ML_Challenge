# Amazon ML Challenge 2026 - Methodology

## Methodology

The solution is a disk-backed entity-resolution pipeline designed for bounded
memory. It normalizes records, generates a small candidate set, computes pairwise
features, trains an XGBoost binary classifier, tunes a decision threshold on a
fixed Source-1 holdout, and groups test predictions by Source-1 entity.

## Candidate generation / blocking

Candidates are the union of country-compatible exact normalized name, legal-core
name, exact normalized address, sorted name-token signature, bounded name-prefix,
bounded longest-name-token, and address-anchor-plus-number blocks. Candidate
buckets with weak keys are capped, then fuzzy name/address and numeric evidence
prunes each Source-1 record to a fixed top-K per target source. The submitted
`candidate_pairs.tsv` is this final set supplied to the classifier.

## Model and feature engineering

The classifier is XGBoost (`binary:logistic`, histogram tree method). Features
include exact normalized-name/core/address matches, RapidFuzz character and token
similarities, numeric-token Jaccard and exact agreement, country agreement,
missingness, length differences, and candidate retrieval score. Training uses
bounded reservoir samples of positive and hard-negative candidate pairs.

## Validation and decision rule

Source-1 entities are deterministically divided into five folds; fold zero is the
holdout. Candidate recall is measured independently. The final probability
threshold maximizes the official per-entity macro F0.5 on held-out Source-1
entities, including singleton credit. No match is forced when all candidates are
below the threshold.

## Reproducibility and constraints

The same normalization, blocking, and feature code is used for train and test.
Large inputs are processed in chunks and intermediate state is stored in SQLite.
The pipeline uses only supplied challenge files and performs no external lookup,
geocoding, registry access, or internet augmentation.

