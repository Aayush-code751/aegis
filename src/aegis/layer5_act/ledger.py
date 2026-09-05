"""A content-keyed audit ledger.

Every decision is recorded against the SHA-256 of the *canonical* document
text, so a row can be reproduced from the document alone without storing the
document, and against the surrogate key fingerprint in force, so a later
reader can tell which governance regime produced which output. The ledger is
append-only and hash-chained: each row carries the digest of the previous row,
which makes silent deletion or reordering detectable.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..layer1_candidates.minhash import content_hash


@dataclass(slots=True)
class LedgerRow:
    """One document's decision record."""

    doc_hash: str
    stage: str
    n_candidates: int = 0
    n_masked: int = 0
    n_escalated: int = 0
    leakage: float | None = None
    overmask: float | None = None
    lam_t: float | None = None
    alpha: float | None = None
    rho_star: float | None = None
    rho_hat: float | None = None
    verdict: str = ""
    key_fingerprint: str = ""
    timestamp: float = field(default_factory=time.time)
    prev_digest: str = ""
    digest: str = ""

    def compute_digest(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k != "digest"}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()


class AuditLedger:
    """Append-only, hash-chained decision log."""

    def __init__(self) -> None:
        self.rows: list[LedgerRow] = []

    @property
    def head(self) -> str:
        return self.rows[-1].digest if self.rows else ""

    def append(self, text: str | None = None, doc_hash: str | None = None, **fields: Any) -> LedgerRow:
        if doc_hash is None:
            if text is None:
                raise ValueError("append needs either text= or doc_hash=")
            doc_hash = content_hash(text)
        row = LedgerRow(doc_hash=doc_hash, prev_digest=self.head, **fields)
        row.digest = row.compute_digest()
        self.rows.append(row)
        return row

    def verify(self) -> bool:
        """Recompute the chain; ``False`` if any row was altered or reordered."""
        previous = ""
        for row in self.rows:
            if row.prev_digest != previous:
                return False
            if row.digest != row.compute_digest():
                return False
            previous = row.digest
        return True

    def to_jsonl(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "AuditLedger":
        ledger = cls()
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    ledger.rows.append(LedgerRow(**json.loads(line)))
        return ledger

    def summary(self) -> dict[str, Any]:
        return {
            "n_rows": len(self.rows),
            "head": self.head,
            "chain_valid": self.verify(),
            "stages": sorted({r.stage for r in self.rows}),
        }
