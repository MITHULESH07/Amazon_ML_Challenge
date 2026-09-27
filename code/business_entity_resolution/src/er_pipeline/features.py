"""Pairwise numerical features for candidate classification."""

from __future__ import annotations

from rapidfuzz.fuzz import ratio, token_set_ratio


FEATURE_NAMES = [
    "name_exact", "name_core_exact", "name_ratio", "name_token_set",
    "address_exact", "address_ratio", "address_token_set",
    "number_jaccard", "number_exact", "country_match",
    "name_missing", "address_missing", "name_length_delta",
    "address_length_delta", "retrieval_score",
]


def _similarity(a: str, b: str) -> float:
    return ratio(a, b) / 100.0 if a and b else 0.0


def _token_similarity(a: str, b: str) -> float:
    return token_set_ratio(a, b) / 100.0 if a and b else 0.0


def _jaccard(a: str, b: str) -> float:
    left, right = set(a.split()), set(b.split())
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def pair_features(left: tuple, right: tuple, retrieval_score: float = 0.0) -> list[float]:
    # Tuple layout: id, source, country, name, core, address, numbers, ...
    _, _, lc, ln, lcore, la, lnums, *_ = left
    _, _, rc, rn, rcore, ra, rnums, *_ = right
    return [
        float(bool(ln) and ln == rn),
        float(bool(lcore) and lcore == rcore),
        _similarity(lcore, rcore),
        _token_similarity(lcore, rcore),
        float(bool(la) and la == ra),
        _similarity(la, ra),
        _token_similarity(la, ra),
        _jaccard(lnums, rnums),
        float(bool(lnums) and lnums == rnums),
        float(bool(lc) and lc == rc),
        float(not lcore or not rcore),
        float(not la or not ra),
        float(abs(len(lcore) - len(rcore))),
        float(abs(len(la) - len(ra))),
        float(retrieval_score),
    ]


def retrieval_similarity(left: tuple, right: tuple) -> float:
    features = pair_features(left, right)
    name = max(features[0], features[1], features[2], features[3])
    address = max(features[4], features[5], features[6])
    numbers = features[7]
    country = features[9]
    return 0.52 * name + 0.35 * address + 0.08 * numbers + 0.05 * country

