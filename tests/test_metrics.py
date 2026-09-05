"""Span matching, P/R/F1, and the statistical protocol."""
from __future__ import annotations

import numpy as np
import pytest

from aegis.metrics import (
    bca_interval,
    candidate_miss_rate,
    evaluate_spans,
    holm_correction,
    match_spans,
    paired_permutation_test,
    prf,
)
from aegis.types import Document, Span


def test_matching_is_one_to_one():
    gold = [Span(0, 10, "PERSON", "Alice Adam")]
    pred = [Span(0, 10, "PERSON", "Alice Adam"), Span(1, 9, "PERSON", "lice Ada")]
    pairs, used_g, used_p = match_spans(gold, pred)
    assert len(pairs) == 1 and len(used_g) == 1 and len(used_p) == 1


def test_matching_requires_type_agreement():
    gold = [Span(0, 5, "PERSON", "Alice")]
    pred = [Span(0, 5, "ADDRESS", "Alice")]
    assert match_spans(gold, pred)[0] == []


def test_matching_respects_the_iou_threshold():
    gold = [Span(0, 10, "PERSON", "0123456789")]
    assert match_spans(gold, [Span(0, 4, "PERSON", "0123")])[0] == []
    assert len(match_spans(gold, [Span(0, 8, "PERSON", "01234567")])[0]) == 1


def test_prf_edge_cases():
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
    assert prf(5, 0, 0) == (1.0, 1.0, 1.0)
    p, r, f = prf(1, 1, 1)
    assert (p, r) == (0.5, 0.5) and f == pytest.approx(0.5)


def test_evaluate_spans_aggregates_micro_and_per_type():
    gold = [Span(0, 5, "PERSON", "Alice"), Span(10, 17, "EMAIL", "a@b.com")]
    pred = [Span(0, 5, "PERSON", "Alice"), Span(30, 35, "PHONE", "12345")]
    m = evaluate_spans([(gold, pred)])
    assert m.micro["tp"] == 1 and m.micro["fp"] == 1 and m.micro["fn"] == 1
    assert m.per_type["PERSON"]["F1"] == pytest.approx(1.0)
    assert m.per_type["EMAIL"]["F1"] == pytest.approx(0.0)
    assert m.per_type["PHONE"]["F1"] == pytest.approx(0.0)


def test_candidate_miss_rate_counts_unproposed_gold():
    text = "Alice and a@b.com"
    doc = Document("d", text, gold=[Span(0, 5, "PERSON", "Alice"),
                                    Span(10, 17, "EMAIL", "a@b.com")])
    doc.candidates = [Span(0, 5, "PERSON", "Alice", 0.9, "rule")]
    rates = candidate_miss_rate([doc])
    assert rates["PERSON"] == 0.0
    assert rates["EMAIL"] == 1.0
    assert rates["micro"] == pytest.approx(0.5)


def test_effective_miss_rate_responds_to_the_score_floor():
    """A weakly scored proposal counts as proposed, but not as usable."""
    doc = Document("d", "Alice", gold=[Span(0, 5, "PERSON", "Alice")])
    doc.candidates = [Span(0, 5, "PERSON", "Alice", 0.3, "rule")]
    assert candidate_miss_rate([doc])["micro"] == 0.0
    assert candidate_miss_rate([doc], min_score=0.6)["micro"] == 1.0


def test_bca_interval_brackets_the_point_estimate():
    values = list(np.random.default_rng(0).normal(0.8, 0.05, 300))
    point, lo, hi = bca_interval(values, lambda u: float(np.mean(u)), reps=400, seed=0)
    assert lo <= point <= hi
    assert hi - lo < 0.05


def test_bca_interval_narrows_with_more_units():
    rng = np.random.default_rng(1)
    _, lo_s, hi_s = bca_interval(list(rng.normal(0, 1, 60)),
                                 lambda u: float(np.mean(u)), reps=300, seed=0)
    _, lo_l, hi_l = bca_interval(list(rng.normal(0, 1, 400)),
                                 lambda u: float(np.mean(u)), reps=300, seed=0)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_bca_handles_an_empty_sample():
    assert bca_interval([], lambda u: 0.0) == (0.0, 0.0, 0.0)


def test_permutation_test_is_insensitive_to_identical_arms():
    units = list(np.random.default_rng(2).normal(0, 1, 120))
    stat = lambda u: float(np.mean(u))
    diff, p = paired_permutation_test(units, stat, stat, reps=400, seed=0)
    assert diff == pytest.approx(0.0)
    assert p > 0.5


def test_holm_correction_is_monotone_and_bounded():
    adjusted = holm_correction({"a": 0.001, "b": 0.02, "c": 0.3, "d": 0.9})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"] <= adjusted["d"]
    assert all(0.0 <= v <= 1.0 for v in adjusted.values())
    assert adjusted["a"] == pytest.approx(0.004)


def test_holm_correction_of_a_single_test_is_identity():
    assert holm_correction({"only": 0.03})["only"] == pytest.approx(0.03)
