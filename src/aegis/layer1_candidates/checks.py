"""Checksum-class labels used by identity-edge normalisation."""
from __future__ import annotations

import re

from ..checksums import iban_valid, luhn_valid

_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")


def checksum_class(text: str, typ: str) -> str:
    """A coarse verification label appended to the identity key.

    Two candidates with the same digits but different verification status are
    the same entity; two candidates that merely *look* alike but fail different
    checks should not merge, so the class enters the key.
    """
    cleaned = _NON_ALNUM.sub("", text)
    if typ == "CREDIT_CARD":
        return "luhn-ok" if luhn_valid(cleaned) else "luhn-bad"
    if typ == "IBAN":
        return "iso7064-ok" if iban_valid(cleaned) else "iso7064-bad"
    return "n/a"
