"""Span matching, P/R/F1, and the statistical protocol of the paper.

Matching is greedy one-to-one at ``IoU >= 0.5`` with type agreement, highest
IoU first, so a single prediction cannot claim two gold spans.

The protocol is document-resampled throughout, because the document is the
sampling unit and the exchangeable unit: BCa bootstrap intervals over
documents, paired permutation tests over documents, and Holm correction across
a stated comparison family. Span-resampled intervals would be too narrow by
exactly the within-document correlation the paper's failure mode F2 is about.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .types import Document, Span


# ---------------------------------------------------------------------------
# Matching and P/R/F1
# ---------------------------------------------------------------------------


def match_spans(
    gold: Sequence[Span],
    pred: Sequence[Span],
    iou_thr: float = 0.5,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Greedy one-to-one matching. Returns ``(pairs, matched_gold, matched_pred)``."""
    scored = sorted(
        (
            (g.iou(p), gi, pi)
            for gi, g in enumerate(gold)
            for pi, p in enumerate(pred)
            if p.type == g.type and g.iou(p) >= iou_thr
        ),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    used_g: set[int] = set()
    used_p: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, gi, pi in scored:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        pairs.append((gi, pi))
    return pairs, used_g, used_p


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


@dataclass(slots=True)
class SpanMetrics:
    """Micro and per-type P/R/F1 plus raw counts."""

    micro: dict[str, float] = field(default_factory=dict)
    per_type: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {"micro": self.micro, "per_type": self.per_type}


def evaluate_spans(
    pairs: Sequence[tuple[Sequence[Span], Sequence[Span]]],
    iou_thr: float = 0.5,
) -> SpanMetrics:
    """Evaluate a list of ``(gold, predicted)`` span lists, one entry per document."""
    tp = fp = fn = 0
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for gold, pred in pairs:
        matched, used_g, used_p = match_spans(gold, pred, iou_thr)
        tp += len(matched)
        fn += len(gold) - len(used_g)
        fp += len(pred) - len(used_p)
        for gi, _ in matched:
            per[gold[gi].type][0] += 1
        for gi, g in enumerate(gold):
            if gi not in used_g:
                per[g.type][2] += 1
        for pi, p in enumerate(pred):
            if pi not in used_p:
                per[p.type][1] += 1
    p, r, f = prf(tp, fp, fn)
    micro = {"P": p, "R": r, "F1": f, "tp": float(tp), "fp": float(fp), "fn": float(fn)}
    per_type = {}
    for typ, (t, fpo, fno) in sorted(per.items()):
        tp_, rp_, fp_ = prf(t, fpo, fno)
        per_type[typ] = {"P": tp_, "R": rp_, "F1": fp_, "support": float(t + fno)}
    return SpanMetrics(micro=micro, per_type=per_type)


def candidate_miss_rate(
    docs: Sequence[Document],
    iou_thr: float = 0.5,
    per_type: bool = True,
    min_score: float = 0.0,
) -> dict[str, float]:
    """Fraction of gold spans no engine proposed -- the term of eq. (1) that no
    threshold can reach. Computable only where gold labels exist, which is why
    Layer IV estimates it label-free in production.

    ``min_score > 0`` gives the *effective* miss rate: gold spans with no
    candidate scoring at least ``min_score``. This is the quantity graph
    propagation moves. Raw proposal is unaffected by propagation -- the
    operator re-scores candidates, it does not create them -- so the effect of
    Layer I's diffusion channel shows up only once a usable score floor is
    imposed, which is how Prop. 1(a) is stated (per-occurrence recall *at a
    threshold*).
    """
    total: dict[str, int] = defaultdict(int)
    missed: dict[str, int] = defaultdict(int)
    for doc in docs:
        for g in doc.gold:
            key = g.type if per_type else "micro"
            total[key] += 1
            if not any(
                c.type == g.type and g.iou(c) >= iou_thr and c.score >= min_score
                for c in doc.candidates
            ):
                missed[key] += 1
    out = {k: missed[k] / v for k, v in total.items() if v}
    grand = sum(total.values())
    if grand:
        out["micro"] = sum(missed.values()) / grand
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# Uncertainty: document-resampled BCa bootstrap
# ---------------------------------------------------------------------------


def _percentile_of(values: np.ndarray, x: float) -> float:
    return float(np.mean(values < x))


def _norm_ppf(p: float) -> float:
    from .layer4_monitor.chao import _normal_quantile

    return _normal_quantile(min(max(p, 1e-9), 1 - 1e-9))


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def bca_interval(
    units: Sequence[object],
    statistic: Callable[[Sequence[object]], float],
    level: float = 0.95,
    reps: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bias-corrected and accelerated bootstrap interval over resampled units.

    Returns ``(point, lo, hi)``. ``units`` are documents, so the interval
    reflects between-document variation and absorbs arbitrary within-document
    span dependence.
    """
    n = len(units)
    if n == 0:
        return (0.0, 0.0, 0.0)
    point = statistic(units)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=np.float64)
    for r in range(reps):
        idx = rng.integers(0, n, size=n)
        draws[r] = statistic([units[i] for i in idx])

    # bias correction
    prop = _percentile_of(draws, point)
    z0 = _norm_ppf(min(max(prop, 1.0 / (reps + 1)), 1 - 1.0 / (reps + 1)))
    # acceleration from jackknife
    jack = np.empty(n, dtype=np.float64)
    for i in range(n):
        jack[i] = statistic([units[j] for j in range(n) if j != i])
    jbar = jack.mean()
    num = float(((jbar - jack) ** 3).sum())
    den = 6.0 * float(((jbar - jack) ** 2).sum()) ** 1.5
    a = num / den if den != 0 else 0.0

    alpha = (1.0 - level) / 2.0
    out: list[float] = []
    for q in (alpha, 1.0 - alpha):
        z = _norm_ppf(q)
        adj = z0 + (z0 + z) / max(1e-12, 1.0 - a * (z0 + z))
        out.append(float(np.percentile(draws, 100.0 * _norm_cdf(adj))))
    lo, hi = sorted(out)
    return (point, lo, hi)


def paired_permutation_test(
    units: Sequence[object],
    statistic_a: Callable[[Sequence[object]], float],
    statistic_b: Callable[[Sequence[object]], float],
    reps: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Two-sided paired permutation test over documents.

    Returns ``(observed difference, p-value)``. Each permutation flips, per
    document, which system's prediction is attributed to which arm, which is
    the exchangeability the null asserts.
    """
    n = len(units)
    if n == 0:
        return (0.0, 1.0)
    observed = statistic_a(units) - statistic_b(units)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(reps):
        flip = rng.random(n) < 0.5
        arm_a = [units[i] for i in range(n) if not flip[i]]
        arm_b = [units[i] for i in range(n) if flip[i]]
        diff = 0.0
        if arm_a:
            diff += statistic_a(arm_a) * len(arm_a) / n - statistic_b(arm_a) * len(arm_a) / n
        if arm_b:
            diff += statistic_b(arm_b) * len(arm_b) / n - statistic_a(arm_b) * len(arm_b) / n
        if abs(diff) >= abs(observed) - 1e-15:
            count += 1
    return (float(observed), (count + 1) / (reps + 1))


def holm_correction(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values over a stated comparison family."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return dict(sorted(adjusted.items()))
