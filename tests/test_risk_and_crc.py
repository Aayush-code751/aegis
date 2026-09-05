"""Layer II: the nested family, monotone risks, and Theorem 1."""
from __future__ import annotations

import random
import statistics as st

import pytest

from aegis.layer2_risk import (
    MaskingFamily,
    crc_bound,
    crc_select,
    leakage_risk,
    overmask_risk,
    risk_matrix,
)
from aegis.types import Document, Span


def test_path_is_nested_and_ends_in_full_redaction(small_batch):
    docs, _ = small_batch
    family = MaskingFamily().fit(docs)
    grid = family.path(256)
    assert len(grid) >= 3
    assert grid[-1].full_redaction, "the top of the family must redact everything"
    for a, b in zip(grid, grid[1:]):
        if b.full_redaction:
            continue
        for typ, thr in a.thresholds.items():
            assert b.thresholds[typ] <= thr + 1e-12, "thresholds must not rise along the path"


def test_masks_are_nested_along_the_path(small_batch):
    docs, _ = small_batch
    family = MaskingFamily().fit(docs)
    grid = family.path(128)
    doc = max(docs, key=lambda d: len(d.candidates))
    previous: set[tuple[int, int, str]] = set()
    for lam in grid:
        if lam.full_redaction:
            continue
        current = {s.key() for s in family.mask(doc, lam)}
        assert previous <= current, "M_lambda must only grow along the path"
        previous = current


def test_risks_are_monotone_in_opposite_directions(small_batch):
    docs, _ = small_batch
    family = MaskingFamily().fit(docs)
    grid = family.path(256)
    leak, over, _ = risk_matrix(docs, family, grid)
    mean_leak = [st.mean(row[j] for row in leak) for j in range(len(grid))]
    mean_over = [st.mean(row[j] for row in over) for j in range(len(grid))]
    assert all(a >= b - 1e-12 for a, b in zip(mean_leak, mean_leak[1:])), "E[L] must not rise"
    assert all(a <= b + 1e-12 for a, b in zip(mean_over, mean_over[1:])), "E[O] must not fall"


def test_boundary_condition_of_theorem_1(small_batch):
    """``L(d, lambda_max) = 0`` must hold exactly, even with candidate misses."""
    docs, _ = small_batch
    family = MaskingFamily().fit(docs)
    top = family.path(64)[-1]
    for doc in docs:
        assert leakage_risk(doc, family.mask(doc, top)) == 0.0


def test_full_redaction_masks_everything_non_sensitive():
    text = "Borrower Melanie Riley owes a balance of 4200 dollars this quarter."
    gold = [Span(9, 22, "PERSON", "Melanie Riley")]
    doc = Document("d", text, gold=gold)
    family = MaskingFamily().fit([doc])
    top = family.path(8)[-1]
    assert overmask_risk(doc, family.mask(doc, top)) == pytest.approx(1.0)


def test_leakage_is_a_fraction_of_gold_spans():
    text = "a@b.com and c@d.com"
    gold = [Span(0, 7, "EMAIL", "a@b.com"), Span(12, 19, "EMAIL", "c@d.com")]
    doc = Document("d", text, gold=gold)
    assert leakage_risk(doc, []) == pytest.approx(1.0)
    assert leakage_risk(doc, [gold[0]]) == pytest.approx(0.5)
    assert leakage_risk(doc, gold) == pytest.approx(0.0)


def test_documents_without_gold_have_zero_leakage():
    doc = Document("d", "no personal data here", gold=[])
    assert leakage_risk(doc, []) == 0.0


def test_crc_bound_inflation_shrinks_with_n():
    assert crc_bound(0.05, 10) > crc_bound(0.05, 1000)
    assert crc_bound(0.05, 1000) == pytest.approx(0.05, abs=2e-3)
    assert crc_bound(0.0, 1, B=1.0) == pytest.approx(0.5)


def test_crc_selects_the_least_aggressive_feasible_point(small_batch):
    docs, _ = small_batch
    family = MaskingFamily().fit(docs)
    grid = family.path(256)
    leak, _, _ = risk_matrix(docs, family, grid)
    j, inflated = crc_select(grid, leak, alpha=0.20)
    assert j is not None
    assert inflated[j] <= 0.20
    assert all(v > 0.20 for v in inflated[:j]), "a feasible earlier point would be preferred"


def test_crc_returns_none_when_nothing_is_feasible(small_batch):
    docs, _ = small_batch
    family = MaskingFamily().fit(docs)
    grid = family.path(32)
    leak, _, _ = risk_matrix(docs, family, grid)
    # an alpha below the 1/(n+1) inflation floor cannot be met by any point
    j, _ = crc_select(grid, leak, alpha=1.0 / (len(docs) + 2))
    assert j is None


def test_crc_coverage_on_synthetic_exchangeable_data():
    """Theorem 1's guarantee, checked by simulation on a controllable loss.

    We construct a monotone loss whose population risk is known and verify the
    realised test risk respects alpha across many independent calibration
    draws. The bound is one-sided, so the check is that violations are rare,
    not that the bound is tight.
    """
    from aegis.layer2_risk.risks import Lambda

    rng = random.Random(0)
    alpha, n, trials = 0.10, 200, 300
    levels = [i / 20 for i in range(21)]
    grid = [Lambda(t=t, thresholds={}) for t in levels]

    def loss(u: float, t: float) -> float:
        # non-increasing in t, zero at t = 1, bounded in [0, 1]
        return max(0.0, min(1.0, (1.0 - t) * u))

    violations = 0
    for _ in range(trials):
        cal = [rng.random() for _ in range(n)]
        risks = [[loss(u, lam.t) for lam in grid] for u in cal]
        j, _ = crc_select(grid, risks, alpha=alpha)
        assert j is not None
        test_risk = st.mean(loss(rng.random(), grid[j].t) for _ in range(400))
        violations += test_risk > alpha
    assert violations / trials <= 0.15, f"violation rate {violations / trials}"
