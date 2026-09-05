"""Arithmetic verification used by the Layer I rule engine.

Luhn for primary account / card numbers and ISO 7064 mod-97-10 for IBANs. Both
are also used in reverse by Layer V, which regenerates surrogates that remain
checksum-valid so downstream format validators keep passing.
"""
from __future__ import annotations

import re

_NON_DIGIT = re.compile(r"\D")
_IBAN_SHAPE = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{10,30}$")


def luhn_check_digit(payload: str) -> str:
    """Return the Luhn check digit for a digit string *without* its check digit."""
    total, double = 0, True
    for ch in reversed(payload):
        value = int(ch)
        if double:
            value *= 2
            if value > 9:
                value -= 9
        total += value
        double = not double
    return str((10 - total % 10) % 10)


def luhn_valid(text: str) -> bool:
    digits = _NON_DIGIT.sub("", text)
    if not 12 <= len(digits) <= 19:
        return False
    return luhn_check_digit(digits[:-1]) == digits[-1]


def iban_checksum(country: str, bban: str) -> str:
    """Return the two ISO 7064 mod-97-10 check digits for ``country + bban``."""
    rearranged = f"{bban}{country}00"
    numeric = "".join(str(int(c, 36)) for c in rearranged.upper())
    return f"{98 - int(numeric) % 97:02d}"


def iban_valid(text: str) -> bool:
    candidate = re.sub(r"\s", "", text).upper()
    if not _IBAN_SHAPE.match(candidate):
        return False
    rotated = candidate[4:] + candidate[:4]
    try:
        numeric = "".join(str(int(c, 36)) for c in rotated)
    except ValueError:
        return False
    return int(numeric) % 97 == 1


def make_iban(country: str, bban: str) -> str:
    """Build a checksum-valid IBAN from a country code and a BBAN body."""
    return f"{country.upper()}{iban_checksum(country, bban)}{bban.upper()}"


def ssn_plausible(text: str) -> bool:
    """SSA allocation rules: area not 000/666, not 900-999; group/serial nonzero."""
    digits = _NON_DIGIT.sub("", text)
    if len(digits) != 9:
        return False
    area, group, serial = int(digits[:3]), int(digits[3:5]), int(digits[5:])
    return area not in (0, 666) and area < 900 and group != 0 and serial != 0
