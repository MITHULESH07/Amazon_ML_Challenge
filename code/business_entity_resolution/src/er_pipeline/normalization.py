"""Unicode-safe normalization shared by training and inference."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd


_SPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"(?<!\w)\d+(?!\w)")
_LEGAL_SUFFIXES = tuple(
    sorted(
        {
            ("private", "limited"), ("pvt", "limited"), ("pvt", "ltd"),
            ("public", "limited"), ("incorporated",), ("corporation",),
            ("company",), ("limited",), ("private",), ("corp",),
            ("inc",), ("llc",), ("llp",), ("ltd",), ("pvt",),
            ("plc",), ("co",),
        },
        key=len,
        reverse=True,
    )
)
_ADDRESS_MAP = {
    "apartment": "apt", "apartments": "apt", "avenue": "ave",
    "boulevard": "blvd", "building": "bldg", "circle": "cir",
    "court": "ct", "drive": "dr", "floor": "fl", "highway": "hwy",
    "lane": "ln", "mount": "mt", "parkway": "pkwy", "place": "pl",
    "road": "rd", "route": "rte", "square": "sq", "street": "st",
    "suite": "ste", "terrace": "ter",
}
_COUNTRY_ALIASES = {
    "america": "us", "u s": "us", "u s a": "us", "usa": "us",
    "united states": "us", "united states of america": "us",
    "in": "india", "ind": "india", "fr": "france", "fra": "france",
    "french republic": "france", "republique francaise": "france",
}


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_text(value: Any) -> str:
    if _missing(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = "".join(
        char if unicodedata.category(char)[0] in {"L", "M", "N"} else " "
        for char in value
    )
    return _SPACE_RE.sub(" ", value).strip()


def remove_legal_suffixes(name: str) -> str:
    tokens = name.split()
    changed = True
    while tokens and changed:
        changed = False
        for suffix in _LEGAL_SUFFIXES:
            if tuple(tokens[-len(suffix):]) == suffix:
                del tokens[-len(suffix):]
                changed = True
                break
    return " ".join(tokens)


def _anchor(text: str) -> str:
    tokens = [token for token in text.split() if len(token) >= 4 and not token.isdigit()]
    return max(tokens, key=lambda token: (len(token), token), default="")


def normalize_record(name: Any, address: Any, country: Any) -> tuple[str, ...]:
    name_norm = normalize_text(name)
    name_core = remove_legal_suffixes(name_norm)
    address_norm = normalize_text(address)
    address_std = " ".join(_ADDRESS_MAP.get(t, t) for t in address_norm.split())
    country_norm = _COUNTRY_ALIASES.get(normalize_text(country), normalize_text(country))
    numbers = " ".join(_NUMBER_RE.findall(address_std))
    compact_name = name_core.replace(" ", "")
    name_signature = " ".join(sorted(set(name_core.split())))
    return (
        name_norm,
        name_core,
        address_std,
        numbers,
        country_norm,
        name_signature,
        compact_name[:8] if len(compact_name) >= 8 else "",
        _anchor(name_core),
        _anchor(address_std),
    )

