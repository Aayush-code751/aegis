"""Layer II: p-value validity and Theorem 2 (simultaneous certification)."""
from __future__ import annotations

import random

import pytest

from aegis.layer2_risk import (
    MaskingFamily,
    betting_pvalue,
    bounded_mean_pvalue,
    hoeffding_bentkus_pvalue,
    learn_then_test,
    risk_matrix,
)
from aegis.layer2_risk.risks import Lambda


def test_hb_is_one_when_the_data_agree_with_the_null():
    # null is E[X] >= alpha; an empirical risk above alpha is no evidence
    assert hoeffding_bentkus_pvalue(0.20, 500, 0.10) == 1.0
    assert hoeffding_bentkus_pvalue(0.10, 500, 0.10) == 1.0


def test_hb_shrinks_as_evidence_grows():
    a = hoeffding_bentkus_pvalue(0.05, 100, 0.10)
    b = hoeffding_bentkus_pvalue(0.05, 1000, 0.10)
    c = hoeffding_bentkus_pvalue(0.01, 1000, 0.10)
    assert 1.0 > a > b > c > 0.0


@pytest.mark.parametrize("fn", [bounded_mean_pvalue, betting_pvalue])
def test_pvalues_are_super_uniform_at_the_null_boundary(fn):
    """P(p <= u) <= u at the null boundary, for the levels that matter."""
    rng = random.Random(0)
    alpha, n, trials = 0.10, 300, 1500
    draws = [
        fn([1.0 if rng.random() < alpha else 0.0 for _ in range(n)], alpha)
        for _ in range(trials)
    ]
    for level in (0.01, 0.05, 0.10):
        rate = sum(p <= level for p in draws) / trials
        assert rate <= level + 0.02, f"level {level}: empirical {rate}"


def test_betting_pvalue_has_power_against_a_true_alternative():
    rng = random.Random(1)
    detected = sum(
        betting_pvalue([1.0 if rng.random() < 0.02 else 0.0 for _ in range(300)], 0.10) <= 0.05
        for _ in range(100)
    )
    assert detected >= 90


def test_min2_combination_pays_its_multiplicity():
    values = [0.01] * 400
    single = min(
        hoeffding_bentkus_pvalue(0.01, 400, 0.10),
        betting_pvalue(values, 0.10),
    )
    assert bounded_mean_pvalue(values, 0.10) == pytest.approx(min(1.0, 2 * single))


def test_empty_sample_is_uninformative():
    assert bounded_mean_pvalue([], 0.10) == 1.0
    assert betting_pvalue([], 0.10) == 1.0


def _synthetic_grid(n: int = 400, size: int = 41, seed: int = 0):
    """A controllable pair of monotone risks with a non-empty feasible window.

    ``E[L] = (1 - t) / 2`` falls below 0.10 once ``t >= 0.8``, while
    ``E[O] = t / 20`` stays under 0.05 everywhere, so ``(alpha, gamma) =
    (0.10, 0.10)`` is genuinely feasible on the upper part of the path -- which
    is what lets the tests distinguish "correctly empty" from "wrongly empty".
    """
    rng = random.Random(seed)
    grid = [Lambda(t=i / (size - 1), thresholds={}) for i in range(size)]
    leak, over = [], []
    for _ in range(n):
        u, v = rng.random(), rng.random()
        leak.append([max(0.0, (1.0 - lam.t) * u) for lam in grid])
        over.append([min(1.0, 0.1 * lam.t * v) for lam in grid])
    return grid, leak, over


def test_certified_set_respects_both_constraints():
    grid, leak, over = _synthetic_grid()
    res = learn_then_test(grid, leak, over, alpha=0.10, gamma=0.10, delta=0.05)
    assert not res.empty
    for j in res.certified:
        assert res.mean_leak[j] <= 0.10 + 1e-9
        assert res.mean_over[j] <= 0.10 + 1e-9


def test_certified_set_is_contiguous_along_the_path():
    grid, leak, over = _synthetic_grid()
    res = learn_then_test(grid, leak, over, alpha=0.10, gamma=0.10, delta=0.05)
    assert res.certified == list(range(res.certified[0], res.certified[-1] + 1))


def test_tightening_alpha_can_only_shrink_the_certified_set():
    grid, leak, over = _synthetic_grid()
    wide = learn_then_test(grid, leak, over, alpha=0.20, gamma=0.10, delta=0.05)
    tight = learn_then_test(grid, leak, over, alpha=0.05, gamma=0.10, delta=0.05)
    assert set(tight.certified) <= set(wide.certified)


def test_empty_output_is_a_valid_answer():
    grid, leak, over = _synthetic_grid()
    res = learn_then_test(grid, leak, over, alpha=1e-4, gamma=1e-4, delta=0.05)
    assert res.empty and res.certified == []


def test_fixed_sequence_is_at_least_as_powerful_as_bonferroni():
    grid, leak, over = _synthetic_grid()
    fs = learn_then_test(grid, leak, over, 0.10, 0.10, 0.05, method="fixed-sequence")
    bf = learn_then_test(grid, leak, over, 0.10, 0.10, 0.05, method="bonferroni")
    assert len(fs.certified) >= len(bf.certified)


def test_ltt_on_real_corpus_certifies_a_loose_level(small_batch):
    docs, _ = small_batch
    family = MaskingFamily().fit(docs)
    grid = family.path(256)
    leak, over, _ = risk_matrix(docs, family, grid)
    res = learn_then_test(grid, leak, over, alpha=0.50, gamma=0.50, delta=0.05)
    assert not res.empty
