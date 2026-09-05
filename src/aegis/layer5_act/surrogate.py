"""Deterministic, format- and checksum-preserving surrogates.

Accepted spans are replaced by surrogates seeded by a keyed HMAC of the
*normalised* value, so the same account maps to the same surrogate stream-wide
with no stored mapping table, and checksummed identifiers are regenerated
Luhn- and mod-97-valid so downstream format validators keep passing.

Determinism buys referential integrity at the price of linkability under
auxiliary knowledge: an adversary who knows one (value, surrogate) pair learns
that mapping everywhere the key is in force. Key rotation per release or per
recipient isolates cohorts at the cost of cross-cohort integrity. That is a
data-governance decision with the same review weight as ``alpha`` itself, and
the audit ledger records which key id governed which decision.
"""
from __future__ import annotations

import hashlib
import hmac
import random
import re
from dataclasses import dataclass
from typing import Sequence

from ..checksums import luhn_check_digit, make_iban
from ..layer1_candidates.propagation import normalise_value
from ..types import Span

_FIRST = ("Alex Jordan Riley Casey Morgan Avery Quinn Rowan Sage Ellis Harper "
          "Reese Emerson Finley Skyler Dakota Peyton Kendall Marlow Tatum").split()
_LAST = ("Calder Whitfield Norwood Ashcombe Fenwick Larkspur Mercer Holloway "
         "Pemberton Rutledge Standish Thornbury Vexley Winslow Yardley Bexford "
         "Cranmore Dunmore Everhart Farrow").split()
_STREETS = "Maple Cedar Willow Aspen Birch Juniper Laurel Rowanberry Hazel Alder".split()
_SUFFIX = "St Ave Rd Ln Blvd Dr Ct Way".split()
_STATES = "AL AK AZ CA CO CT FL GA IL IN MA MD MI MN NC NJ NY OH PA TX VA WA WI".split()


@dataclass(frozen=True, slots=True)
class SurrogateKey:
    """An HMAC key plus the identifier the ledger records."""

    secret: bytes
    key_id: str = "k1"

    @classmethod
    def from_string(cls, secret: str, key_id: str = "k1") -> "SurrogateKey":
        return cls(secret.encode("utf-8"), key_id)

    def fingerprint(self) -> str:
        """A non-secret digest of the key, safe to publish in an audit trail."""
        return hashlib.sha256(b"aegis-key-fingerprint" + self.secret).hexdigest()[:16]


class Surrogate:
    """Generate and apply surrogates for accepted spans."""

    def __init__(self, key: SurrogateKey | str = "aegis-release-key") -> None:
        self.key = SurrogateKey.from_string(key) if isinstance(key, str) else key

    def _rng(self, span: Span) -> random.Random:
        payload = f"{span.type}\x00{normalise_value(span.text, span.type)}".encode()
        digest = hmac.new(self.key.secret, payload, hashlib.sha256).digest()
        return random.Random(int.from_bytes(digest[:16], "big"))

    def for_span(self, span: Span) -> str:
        """A type-appropriate, shape-preserving replacement for ``span``."""
        rng = self._rng(span)
        typ, text = span.type, span.text

        if typ == "EMAIL":
            domain = text.split("@")[-1] if "@" in text else "example.org"
            return f"{rng.choice(_FIRST).lower()}.{rng.choice(_LAST).lower()}@{domain}"
        if typ == "PERSON":
            return f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
        if typ == "PHONE":
            return re.sub(r"\d", lambda _: str(rng.randint(0, 9)), text)
        if typ == "SSN":
            return f"{rng.randint(100, 665):03d}-{rng.randint(10, 99):02d}-{rng.randint(1000, 9999):04d}"
        if typ == "CREDIT_CARD":
            body = "".join(str(rng.randint(0, 9)) for _ in range(15))
            digits = body + luhn_check_digit(body)
            out, i = [], 0
            for ch in text:
                if ch.isdigit():
                    out.append(digits[i % 16])
                    i += 1
                else:
                    out.append(ch)
            return "".join(out)
        if typ == "IBAN":
            country = text[:2].upper() if text[:2].isalpha() else "DE"
            bban = "".join(str(rng.randint(0, 9)) for _ in range(16))
            return make_iban(country, bban)
        if typ == "DOB":
            day, month, year = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1940, 2005)
            if re.match(r"^\d{4}-\d{2}-\d{2}$", text.strip()):
                return f"{year:04d}-{month:02d}-{day:02d}"
            return f"{month:02d}/{day:02d}/{year:04d}"
        if typ == "ADDRESS":
            base = f"{rng.randint(10, 9999)} {rng.choice(_STREETS)} {rng.choice(_SUFFIX)}"
            if re.search(r"\d{5}\s*$", text):
                return f"{base}, {rng.choice(_STREETS)}ville, {rng.choice(_STATES)} {rng.randint(10000, 99999)}"
            return base
        if typ == "ACCOUNT_NUMBER":
            return re.sub(
                r"[A-Z]", lambda _: chr(rng.randint(65, 90)),
                re.sub(r"\d", lambda _: str(rng.randint(0, 9)), text),
            )
        if typ == "__ALL__":
            return "[REDACTED]"
        return "".join(rng.choice("XZQJ0123456789") for _ in text)

    def apply(self, text: str, spans: Sequence[Span]) -> str:
        """Replace every span, right to left so earlier offsets stay valid."""
        if any(s.type == "__ALL__" for s in spans):
            return "[REDACTED]"
        out = text
        for span in sorted(spans, key=lambda s: -s.start):
            if 0 <= span.start < span.end <= len(out):
                out = out[: span.start] + self.for_span(span) + out[span.end :]
        return out
