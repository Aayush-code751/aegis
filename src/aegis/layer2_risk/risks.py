"""The nested masking family and the two per-document risks.

The nesting parameter is a point on a **monotone path** through Lambda. Moving
along the path can only add masking, which is the structural hypothesis both
conformal risk control (Thm. 1) and Learn-then-Test (Thm. 2) need:

    lambda <= lambda'  =>  M_lambda(d) subseteq M_lambda'(d)   for every d

Concretely a point on the path carries one acceptance threshold per type,
derived from **unlabelled** quantiles of the propagated candidate scores, so
the path itself costs no calibration labels and LTT's guarantee stays clean.
The top of the path is *full redaction* -- mask the whole document, i.e. route
it to review. That is what makes ``L(d, lambda_max) = 0`` hold exactly even
when candidate generation is incomplete: an unproposed gold span can never be
masked by a threshold, but it is masked by redacting the document. Without
that top element the boundary condition of Thm. 1 would be false precisely
because of the candidate-miss term of eq. (1), which is the failure mode the
paper is about.
"""
from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..taxonomy import FINANCE_TYPES
from ..types import Document, Span

_WORD = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class Lambda:
    """One point on the monotone path.

    ``t`` in [0, 1] is the path position; ``thresholds`` maps a type to the
    minimum propagated score it must reach to be masked; ``full_redaction``
    marks the top element of the family.
    """

    t: float
    thresholds: dict[str, float] = field(default_factory=dict)
    full_redaction: bool = False

    def lam_y(self, typ: str) -> float:
        """The paper's ``lambda_y = 1 - threshold_y``."""
        return 1.0 - self.thresholds.get(typ, 1.0)

    def as_dict(self) -> dict[str, float]:
        out = {f"lambda_{k}": round(1.0 - v, 6) for k, v in sorted(self.thresholds.items())}
        out["t"] = round(self.t, 6)
        out["full_redaction"] = float(self.full_redaction)
        return out


def _order_statistic(sorted_values: Sequence[float], q: float) -> float:
    """The ``q``-quantile as an **exact order statistic**, never interpolated.

    Interpolating between order statistics would emit thresholds a hair above
    or below an observed score, and because rule engines emit heavily tied
    discrete scores that flips whole groups of candidates in and out of the
    mask as the path advances -- destroying the monotonicity that Thms. 1 and 2
    require. Returning an observed value keeps every ``score >= threshold``
    comparison exact and the family provably nested.
    """
    if not sorted_values:
        return 1.0 + 1e-9
    q = min(max(q, 0.0), 1.0)
    k = int(q * (len(sorted_values) - 1) + 1e-12)
    return sorted_values[min(max(k, 0), len(sorted_values) - 1)]


class MaskingFamily:
    """A monotone path of masking rules, built from unlabelled score quantiles.

    ``fit`` collects the propagated scores of every candidate, per type, over
    the pooled batch. Path position ``t`` then uses the ``1 - t`` quantile of
    each type's score distribution as its acceptance threshold, so ``t = 0``
    masks (almost) nothing, thresholds fall monotonically as ``t`` grows, and
    the final grid point is full redaction.
    """

    def __init__(self, types: Iterable[str] = FINANCE_TYPES, iou_thr: float = 0.5) -> None:
        self.types = tuple(types)
        self.iou_thr = iou_thr
        self._scores: dict[str, list[float]] = {t: [] for t in self.types}
        self._fitted = False

    def fit(self, docs: Sequence[Document]) -> "MaskingFamily":
        for doc in docs:
            for span in doc.candidates:
                if span.type in self._scores:
                    self._scores[span.type].append(span.score)
        for values in self._scores.values():
            values.sort()
        self._fitted = True
        return self

    def levels(self) -> dict[str, list[float]]:
        """Distinct observed scores per type, highest first."""
        return {t: sorted(set(self._scores[t]), reverse=True) for t in self.types}

    def path(self, n_grid: int = 2048, stagger: bool = True) -> list[Lambda]:
        """The monotone path, least masking first, full redaction last.

        Two constructions, both coordinatewise non-decreasing in the path index
        and therefore both valid nested families:

        ``stagger=True`` (default) is a **staircase**: at each step exactly one
        type's threshold drops to the next-highest score that type actually
        attained, chosen greedily by descending score across types. Because a
        threshold only ever falls, the family is nested; because only one
        coordinate moves at a time, the path is as fine as the score
        granularity allows -- roughly ``sum_y |distinct scores of y|`` points
        rather than the handful a joint quantile sweep yields on tied rule
        scores.

        ``stagger=False`` sweeps all types together through a shared quantile
        level, which is coarser but easier to describe.

        In both cases thresholds are exact observed scores, consecutive
        duplicates are collapsed, and the final element is full redaction so
        ``L(d, lambda_max) = 0`` holds.
        """
        if not self._fitted:
            raise RuntimeError("call MaskingFamily.fit(pooled_docs) first")
        if n_grid < 2:
            raise ValueError("n_grid must be at least 2")

        thresholds = {t: 1.0 + 1e-9 for t in self.types}
        raw: list[dict[str, float]] = [dict(thresholds)]

        if stagger:
            levels = self.levels()
            pointer = {t: 0 for t in self.types}
            while len(raw) < n_grid - 1:
                best_type, best_value = None, -1.0
                for typ in self.types:
                    idx = pointer[typ]
                    if idx < len(levels[typ]) and levels[typ][idx] > best_value:
                        best_type, best_value = typ, levels[typ][idx]
                if best_type is None:
                    break
                pointer[best_type] += 1
                thresholds[best_type] = best_value
                raw.append(dict(thresholds))
        else:
            interior = max(n_grid - 2, 1)
            for i in range(1, interior + 1):
                q = 1.0 - i / (interior + 1)
                raw.append(
                    {
                        typ: _order_statistic(self._scores[typ], q)
                        if self._scores[typ] else 1.0 + 1e-9
                        for typ in self.types
                    }
                )

        grid: list[Lambda] = []
        previous: tuple[tuple[str, float], ...] | None = None
        for thr in raw:
            key = tuple(sorted(thr.items()))
            if key == previous:
                continue
            previous = key
            grid.append(Lambda(t=0.0, thresholds=thr))
        # relabel t as the normalised path position, then append full redaction
        n = max(len(grid), 1)
        grid = [Lambda(t=i / n, thresholds=g.thresholds) for i, g in enumerate(grid)]
        grid.append(Lambda(t=1.0, thresholds={t: 0.0 for t in self.types},
                           full_redaction=True))
        return grid

    # -- the masking rule itself -------------------------------------------

    def mask(self, doc: Document, lam: Lambda, resolved: set[tuple[int, int, str]] | None = None
             ) -> list[Span]:
        """``M_lambda(d)``: the spans this rule masks.

        ``resolved`` optionally carries spans that Layer V escalated and the
        LLM tier confirmed; those are masked regardless of threshold, which is
        how escalation buys risk reduction.
        """
        if lam.full_redaction:
            return [Span(0, len(doc.text), "__ALL__", doc.text, 1.0, "redact-all")]
        out: list[Span] = []
        for span in doc.candidates:
            thr = lam.thresholds.get(span.type, 1.0 + 1e-9)
            if span.score >= thr or (resolved is not None and span.key() in resolved):
                out.append(span)
        return out


# ---------------------------------------------------------------------------
# Risk 1: leakage
# ---------------------------------------------------------------------------


def leakage_risk(
    doc: Document,
    masked: Sequence[Span],
    iou_thr: float = 0.5,
) -> float:
    """``L(d, lambda) = |S(d) \\ M_lambda(d)| / (|S(d)| v 1)``.

    A gold span counts as masked when some masked span of the same type
    overlaps it at IoU >= ``iou_thr``; the full-redaction element masks
    everything, so it covers every gold span by construction.
    """
    if not doc.gold:
        return 0.0
    if any(m.type == "__ALL__" for m in masked):
        return 0.0
    leaked = 0
    for gold in doc.gold:
        if not any(m.type == gold.type and gold.iou(m) >= iou_thr for m in masked):
            leaked += 1
    return leaked / max(len(doc.gold), 1)


# ---------------------------------------------------------------------------
# Risk 2: over-masking
# ---------------------------------------------------------------------------


def _token_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _WORD.finditer(text)]


def _covered(tokens: Sequence[tuple[int, int]], spans: Sequence[Span]) -> set[int]:
    """Indices of tokens overlapping any of ``spans`` (O(n log n) via bisect)."""
    if not tokens or not spans:
        return set()
    starts = [t[0] for t in tokens]
    hit: set[int] = set()
    for span in spans:
        i = max(0, bisect_left(starts, span.start) - 1)
        while i < len(tokens) and tokens[i][0] < span.end:
            if tokens[i][1] > span.start:
                hit.add(i)
            i += 1
    return hit


def overmask_risk(doc: Document, masked: Sequence[Span]) -> float:
    """``O(d, lambda)``: masked non-sensitive tokens over non-sensitive tokens."""
    tokens = _token_spans(doc.text)
    if not tokens:
        return 0.0
    pii = _covered(tokens, doc.gold)
    if any(m.type == "__ALL__" for m in masked):
        masked_idx = set(range(len(tokens)))
    else:
        masked_idx = _covered(tokens, masked)
    non_sensitive = len(tokens) - len(pii)
    if non_sensitive <= 0:
        return 0.0
    return len(masked_idx - pii) / non_sensitive


def escalation_cost(doc: Document, lam: Lambda, band_floor: float = 0.20) -> float:
    """Fraction of a document's candidates that fall in the escalation band.

    The band is ``[b, threshold_y)`` -- scored high enough to be worth a
    frontier-LLM call, not high enough for the gate to accept.
    """
    if lam.full_redaction or not doc.candidates:
        return 1.0 if lam.full_redaction else 0.0
    banded = 0
    for span in doc.candidates:
        thr = lam.thresholds.get(span.type, 1.0 + 1e-9)
        if band_floor <= span.score < thr:
            banded += 1
    return banded / max(len(doc.candidates), 1)


# ---------------------------------------------------------------------------
# Vectorised evaluation over a grid
# ---------------------------------------------------------------------------


def risk_matrix(
    docs: Sequence[Document],
    family: MaskingFamily,
    grid: Sequence[Lambda],
    band_floor: float = 0.20,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """Evaluate ``L``, ``O`` and escalation cost for every (document, lambda).

    Returns three ``n_docs x n_grid`` matrices. This is the only expensive step
    of calibration; it is ``O(|grid| * n)`` as stated in the paper, and the
    monotone structure means a caller can early-stop along the path.
    """
    n, g = len(docs), len(grid)
    L = [[0.0] * g for _ in range(n)]
    O = [[0.0] * g for _ in range(n)]
    E = [[0.0] * g for _ in range(n)]
    for i, doc in enumerate(docs):
        tokens = _token_spans(doc.text)
        pii = _covered(tokens, doc.gold)
        n_tokens = len(tokens)
        non_sensitive = max(n_tokens - len(pii), 0)
        n_gold = max(len(doc.gold), 1)
        n_cands = max(len(doc.candidates), 1)
        for j, lam in enumerate(grid):
            if lam.full_redaction:
                L[i][j] = 0.0
                O[i][j] = 1.0 if non_sensitive > 0 else 0.0
                E[i][j] = 1.0
                continue
            masked = [
                s for s in doc.candidates
                if s.score >= lam.thresholds.get(s.type, 1.0 + 1e-9)
            ]
            leaked = 0
            for gold in doc.gold:
                if not any(m.type == gold.type and gold.iou(m) >= family.iou_thr for m in masked):
                    leaked += 1
            L[i][j] = leaked / n_gold if doc.gold else 0.0
            if non_sensitive > 0:
                O[i][j] = len(_covered(tokens, masked) - pii) / non_sensitive
            banded = sum(
                1 for s in doc.candidates
                if band_floor <= s.score < lam.thresholds.get(s.type, 1.0 + 1e-9)
            )
            E[i][j] = banded / n_cands
    return L, O, E
