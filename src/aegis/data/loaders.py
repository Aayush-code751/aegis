"""Corpus loading, label mapping, and the calibration/test split.

Every corpus lands in the same shape: a list of :class:`aegis.types.Document`
carrying gold spans in the canonical taxonomy. Native labels are mapped by
name; anything unmapped is dropped rather than force-mapped, and the counts are
returned so a run can report them (the paper's "kept of total" figures).
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from ..taxonomy import (
    FINANCE_TYPES,
    GRETEL_TO_CANON,
    NEMOTRON_TO_CANON,
    map_label,
)
from ..types import Document, Span

#: Overridable via the ``AEGIS_DATA_ROOT`` environment variable so the Docker
#: image can mount a corpus directory read-only.
DATA_ROOT = Path(os.environ.get("AEGIS_DATA_ROOT", Path(__file__).resolve().parents[3] / "data"))

SYNPII_SEEDS: tuple[int, ...] = (7, 13, 21, 42, 77)
GRETEL_TRUNCATE_CHARS = 8_000  # applied uniformly across all systems


@dataclass(slots=True)
class LabelStats:
    """How many native labels survived the taxonomy mapping."""

    kept: int = 0
    dropped: int = 0
    dropped_labels: dict[str, int] | None = None

    @property
    def total(self) -> int:
        return self.kept + self.dropped

    def as_dict(self) -> dict[str, object]:
        return {
            "kept": self.kept,
            "dropped": self.dropped,
            "total": self.total,
            "dropped_labels": dict(sorted((self.dropped_labels or {}).items())),
        }


def load_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream a JSONL file, skipping blank lines."""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _spans_from_gold(
    rows: Iterable[dict],
    text: str,
    table: dict[str, str] | None,
    stats: LabelStats,
    allowed: frozenset[str],
) -> list[Span]:
    out: list[Span] = []
    for row in rows or ():
        native = str(row.get("type", row.get("label", "")))
        canon = native if table is None else map_label(native, table)
        if canon is None or canon not in allowed:
            stats.dropped += 1
            if stats.dropped_labels is None:
                stats.dropped_labels = {}
            stats.dropped_labels[native] = stats.dropped_labels.get(native, 0) + 1
            continue
        start, end = int(row["start"]), int(row["end"])
        if not (0 <= start < end <= len(text)):
            stats.dropped += 1
            continue
        stats.kept += 1
        out.append(Span(start, end, canon, row.get("text") or text[start:end]))
    return sorted(out, key=lambda s: (s.start, s.end))


def load_synpii(
    seed: int = 7,
    root: Path | None = None,
    allowed: Iterable[str] = FINANCE_TYPES,
) -> tuple[list[Document], LabelStats]:
    """SynPII-F for one generation seed (labels are already canonical)."""
    root = Path(root or DATA_ROOT) / "synpii"
    path = root / f"synpii_finance_seed{seed}.jsonl"
    stats = LabelStats()
    allowed_set = frozenset(allowed)
    docs: list[Document] = []
    for row in load_jsonl(path):
        text = row["text"]
        docs.append(
            Document(
                doc_id=str(row["id"]),
                text=text,
                gold=_spans_from_gold(row.get("gold") or [], text, None, stats, allowed_set),
                meta={
                    "corpus": "synpii_finance",
                    "seed": seed,
                    "is_dup": bool(row.get("is_dup", False)),
                    "dup_kind": row.get("dup_kind", ""),
                    "src_id": row.get("src_id", ""),
                },
            )
        )
    return docs, stats


def _load_external(
    filename: str,
    corpus: str,
    table: dict[str, str],
    root: Path | None,
    allowed: Iterable[str],
    truncate: int | None,
) -> tuple[list[Document], LabelStats]:
    path = Path(root or DATA_ROOT) / "external" / filename
    stats = LabelStats()
    allowed_set = frozenset(allowed)
    docs: list[Document] = []
    for row in load_jsonl(path):
        text = row["text"]
        if truncate is not None and len(text) > truncate:
            text = text[:truncate]
        gold = _spans_from_gold(row.get("gold") or [], text, table, stats, allowed_set)
        docs.append(
            Document(
                doc_id=str(row["id"]),
                text=text,
                gold=gold,
                meta={"corpus": corpus, "source": row.get("source", "")},
            )
        )
    return docs, stats


def load_gretel(root: Path | None = None, allowed: Iterable[str] = FINANCE_TYPES):
    """Gretel synthetic PII finance (English subset shipped with the release)."""
    return _load_external(
        "gretel_finance.jsonl", "gretel_finance", GRETEL_TO_CANON, root, allowed,
        GRETEL_TRUNCATE_CHARS,
    )


def load_nemotron(root: Path | None = None, allowed: Iterable[str] = FINANCE_TYPES):
    """NVIDIA Nemotron-PII, ``domain=Finance`` rows."""
    return _load_external(
        "nemotron_finance.jsonl", "nemotron_finance", NEMOTRON_TO_CANON, root, allowed,
        None,
    )


def load_corpus(name: str, seed: int = 7, root: Path | None = None):
    """Dispatch on corpus name: ``synpii`` | ``gretel`` | ``nemotron``."""
    key = name.lower().replace("-", "_")
    if key in ("synpii", "synpii_f", "synpii_finance"):
        return load_synpii(seed=seed, root=root)
    if key in ("gretel", "gretel_finance"):
        return load_gretel(root=root)
    if key in ("nemotron", "nemotron_finance"):
        return load_nemotron(root=root)
    raise ValueError(f"unknown corpus {name!r}")


def split_calibration_test(
    docs: list[Document],
    n_cal: int | None = None,
    cal_frac: float = 0.4,
    seed: int = 0,
    keep_duplicates_in_test: bool = True,
) -> tuple[list[Document], list[Document]]:
    """Split into calibration and test.

    On SynPII-F the injected duplicates ride the *test* stream, matching the
    paper: unique documents are split ``cal_frac``/``1 - cal_frac`` and every
    duplicate is appended to test, so propagation has recurring entities to
    work with exactly where the certificate is evaluated.
    """
    rng = random.Random(seed)
    dups = [d for d in docs if d.meta.get("is_dup")] if keep_duplicates_in_test else []
    uniques = [d for d in docs if not d.meta.get("is_dup")] if dups else list(docs)

    order = list(range(len(uniques)))
    rng.shuffle(order)
    k = n_cal if n_cal is not None else max(1, int(round(cal_frac * len(uniques))))
    k = min(k, len(uniques) - 1) if len(uniques) > 1 else len(uniques)

    cal = [uniques[i] for i in order[:k]]
    test = [uniques[i] for i in order[k:]] + dups
    rng.shuffle(test)
    return cal, test
