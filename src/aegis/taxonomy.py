"""Entity taxonomy and third-party label mappings.

The finance taxonomy is the nine types used in the paper. Native labels from
the external corpora are mapped by *name*; anything unmapped is dropped rather
than force-mapped, which is what the paper's kept/dropped counts refer to.
"""
from __future__ import annotations

FINANCE_TYPES: tuple[str, ...] = (
    "ACCOUNT_NUMBER",
    "ADDRESS",
    "CREDIT_CARD",
    "DOB",
    "EMAIL",
    "IBAN",
    "PERSON",
    "PHONE",
    "SSN",
)

#: Superset also covering the healthcare taxonomy, for cross-domain reuse.
CANONICAL_TYPES: tuple[str, ...] = FINANCE_TYPES + ("INSURANCE_ID", "MRN")

#: Types carrying an arithmetic check, which is what makes a strict alpha cheap.
CHECKSUMMED_TYPES: frozenset[str] = frozenset({"CREDIT_CARD", "IBAN"})

# --------------------------------------------------------------------------
# Third-party label maps
# --------------------------------------------------------------------------

GRETEL_TO_CANON: dict[str, str] = {
    "name": "PERSON",
    "first_name": "PERSON",
    "last_name": "PERSON",
    "middle_name": "PERSON",
    "customer_id": "ACCOUNT_NUMBER",
    "account_number": "ACCOUNT_NUMBER",
    "bank_account_number": "ACCOUNT_NUMBER",
    "iban": "IBAN",
    "credit_card_number": "CREDIT_CARD",
    "credit_debit_number": "CREDIT_CARD",
    "email": "EMAIL",
    "phone_number": "PHONE",
    "street_address": "ADDRESS",
    "date_of_birth": "DOB",
    "ssn": "SSN",
    "national_id": "SSN",
}

NEMOTRON_TO_CANON: dict[str, str] = {
    "PERSON": "PERSON",
    "FIRST_NAME": "PERSON",
    "LAST_NAME": "PERSON",
    "EMAIL": "EMAIL",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "PHONE": "PHONE",
    "STREET_ADDRESS": "ADDRESS",
    "ADDRESS": "ADDRESS",
    "DATE_OF_BIRTH": "DOB",
    "DOB": "DOB",
    "SSN": "SSN",
    "IBAN": "IBAN",
    "CREDIT_CARD_NUMBER": "CREDIT_CARD",
    "CREDIT_DEBIT_CARD": "CREDIT_CARD",
    "ACCOUNT_NUMBER": "ACCOUNT_NUMBER",
    "BANK_ROUTING_NUMBER": "ACCOUNT_NUMBER",
}

#: Presidio entity types -> canonical, under the recall-generous mappings of
#: the paper (DATE_TIME -> DOB and LOCATION -> ADDRESS are deliberate).
PRESIDIO_TO_CANON: dict[str, str] = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "US_SSN": "SSN",
    "CREDIT_CARD": "CREDIT_CARD",
    "IBAN_CODE": "IBAN",
    "US_BANK_NUMBER": "ACCOUNT_NUMBER",
    "DATE_TIME": "DOB",
    "LOCATION": "ADDRESS",
}


def map_label(native: str, table: dict[str, str]) -> str | None:
    """Map a native label to the canonical taxonomy, or ``None`` if unmapped.

    A label that is *already* canonical passes through unchanged, so the same
    function works on a raw Hugging Face export (native vocabulary) and on the
    normalised JSONL shipped with this release.
    """
    upper = native.upper().replace("-", "_").replace(" ", "_")
    if upper in CANONICAL_TYPES:
        return upper
    if native in table:
        return table[native]
    lower = native.lower()
    for key, value in table.items():
        if key.upper() == upper or key.lower() == lower:
            return value
    return None
