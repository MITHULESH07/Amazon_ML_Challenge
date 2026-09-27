"""Deterministic, Unicode-safe normalization for business entity records.

The functions in this module do not mutate their inputs.  The original business
name, address, and country columns are retained so normalized values can be
audited and alternative features can be created later.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd


_WHITESPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"(?<!\w)\d+(?!\w)")

# These are removed only from the secondary ``name_core`` representation.  The
# conservative ``name_normalized`` representation always retains them.
_LEGAL_SUFFIXES: tuple[tuple[str, ...], ...] = tuple(
    sorted(
        {
            ("private", "limited"),
            ("pvt", "limited"),
            ("pvt", "ltd"),
            ("public", "limited"),
            ("incorporated",),
            ("corporation",),
            ("company",),
            ("limited",),
            ("private",),
            ("corp",),
            ("inc",),
            ("llc",),
            ("llp",),
            ("ltd",),
            ("pvt",),
            ("plc",),
            ("co",),
        },
        key=len,
        reverse=True,
    )
)

# Address-only canonicalization.  Ambiguous abbreviations are not expanded;
# full forms are shortened so equivalent variants converge on one token.
_ADDRESS_TOKEN_MAP = {
    "apartment": "apt",
    "apartments": "apt",
    "avenue": "ave",
    "boulevard": "blvd",
    "building": "bldg",
    "circle": "cir",
    "court": "ct",
    "drive": "dr",
    "floor": "fl",
    "highway": "hwy",
    "lane": "ln",
    "mount": "mt",
    "parkway": "pkwy",
    "place": "pl",
    "road": "rd",
    "route": "rte",
    "square": "sq",
    "street": "st",
    "suite": "ste",
    "terrace": "ter",
}

_COUNTRY_ALIASES = {
    "america": "us",
    "fr": "france",
    "fra": "france",
    "french republic": "france",
    "in": "india",
    "ind": "india",
    "republique francaise": "france",
    "u s": "us",
    "u s a": "us",
    "united states": "us",
    "united states of america": "us",
    "usa": "us",
}


def _is_missing(value: Any) -> bool:
    """Return True for scalar missing values supported by pandas."""

    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_text(value: Any) -> str:
    """Return conservative Unicode-safe text suitable for comparison.

    Unicode letters, combining marks, and numbers from every script are kept.
    Punctuation, symbols, and control characters become token boundaries rather
    than being deleted, preventing accidental joins such as ``"ab-c"`` ->
    ``"abc"``.
    """

    if _is_missing(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value)).casefold()
    cleaned = "".join(
        character if unicodedata.category(character)[0] in {"L", "M", "N"} else " "
        for character in text
    )
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def remove_legal_suffixes(normalized_name: str) -> str:
    """Remove recognized trailing legal suffixes from an already-normalized name.

    Removal is repeated to handle endings such as ``"private limited company"``.
    The operation is intentionally suffix-only: a token such as ``"company"`` in
    the middle of a name is preserved.
    """

    tokens = normalized_name.split()
    changed = True
    while tokens and changed:
        changed = False
        for suffix in _LEGAL_SUFFIXES:
            suffix_length = len(suffix)
            if tuple(tokens[-suffix_length:]) == suffix:
                del tokens[-suffix_length:]
                changed = True
                break
    return " ".join(tokens)


def normalize_business_name(value: Any) -> dict[str, str]:
    """Create conservative and suffix-stripped name representations."""

    normalized = normalize_text(value)
    core = remove_legal_suffixes(normalized)
    return {
        "name_normalized": normalized,
        "name_core": core,
    }


def normalize_address(value: Any) -> dict[str, str]:
    """Create normalized address variants and extract numeric evidence.

    Numeric tokens are retained as a space-separated string for later matching
    features.  More speculative fields, such as postal-code guesses, are not
    persisted at this stage.
    """

    normalized = normalize_text(value)
    standardized = " ".join(
        _ADDRESS_TOKEN_MAP.get(token, token) for token in normalized.split()
    )
    numbers = _NUMBER_RE.findall(standardized)
    return {
        "address_normalized": normalized,
        "address_standardized": standardized,
        "address_numbers": " ".join(numbers),
    }


def normalize_country(value: Any) -> str:
    """Normalize known aliases while allowing unseen countries to pass through."""

    normalized = normalize_text(value)
    return _COUNTRY_ALIASES.get(normalized, normalized)


def normalize_dataframe(frame: pd.DataFrame, *, copy: bool = True) -> pd.DataFrame:
    """Add normalized feature columns to a source DataFrame.

    Required input columns are ``business_name``, ``business_address``, and
    ``country``.  Other columns, including ``entity_id``, are preserved unchanged.
    This function is deterministic and can be applied independently to chunks.
    """

    required = {"business_name", "business_address", "country"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    result = frame.copy() if copy else frame

    name_features = pd.DataFrame(
        result["business_name"].map(normalize_business_name).tolist(),
        index=result.index,
    )
    address_features = pd.DataFrame(
        result["business_address"].map(normalize_address).tolist(),
        index=result.index,
    )

    for column in name_features.columns:
        result[column] = name_features[column]
    for column in address_features.columns:
        result[column] = address_features[column]
    result["country_normalized"] = result["country"].map(normalize_country)

    return result


def iter_normalized_tsv(
    path: str | Path,
    *,
    chunksize: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """Yield normalized chunks from a source TSV without loading it all at once."""

    if chunksize <= 0:
        raise ValueError("chunksize must be greater than zero")

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    ):
        yield normalize_dataframe(chunk)

