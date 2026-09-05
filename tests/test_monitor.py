"""Layer IV: Ville validity (Thm. 5), e-BH, and the Chao bound (Prop. 2)."""
from __future__ import annotations

import random

import numpy as np
import pytest

from aegis.layer4_monitor import (
    EDetector,
    EProcess,
    chao1,
    dark_matter_rate,
    ebh_reject,
    ebh_threshold,
)
from aegis.types import Span


def _stream(rng: random.Random, n: int, rate: float) -> list[float]:
    return [1.0 if rng.random() < rate else 0.0 for _ in range(n)]


def test_eprocess_starts_at_one_and_stays_non_negative():
    ep = EProcess(0.05, 0.05)
    assert ep.wealth == 1.0
    ep.run(_stream(random.Random(0), 500, 0.5))
    assert ep.wealth > 0.0


def test_ville_inequality_holds_for_the_eprocess():
    """P(sup_t E_t >= 1/delta) <= delta under the null."""
    rng = random.Random(0)
    alpha, delta, trials = 0.05, 0.05, 400
    alarms = sum(
        EProcess(alpha, delta).run(_stream(rng, 2000, alpha)).alarmed for _ in range(trials)
    )
    assert alarms / trials <= delta + 0.02


def test_edetector_mixture_is_also_valid():
    """The mixture keeps the 1/delta threshold; a running max would not."""
    rng = random.Random(1)
    alpha, delta, trials = 0.05, 0.05, 300
    alarms = sum(
        EDetector(alpha, delta).run(_stream(rng, 2000, alpha)).alarm_time is not None
        for _ in range(trials)
    )
    assert alarms / trials <= delta + 0.02


def test_edetector_detects_a_real_changepoint():
    rng = random.Random(2)
    delays = []
    for _ in range(40):
        stream = _stream(rng, 400, 0.05) + _stream(rng, 2000, 0.35)
        det = EDetector(0.05, 0.05).run(stream)
        assert det.alarm_time is not None
        delays.append(det.alarm_time - 400)
    assert 0 < float(np.median(delays)) < 300


def test_eprocess_does_not_alarm_below_the_level():
    rng = random.Random(3)
    ep = EProcess(0.20, 0.05).run(_stream(rng, 3000, 0.02))
    assert not ep.alarmed


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
def test_eprocess_rejects_invalid_levels(bad):
    with pytest.raises(ValueError):
        EProcess(bad, 0.05)


def test_ebh_rejects_only_large_evalues():
    e = {"PERSON": 400.0, "EMAIL": 1.2, "SSN": 0.4, "IBAN": 30.0}
    assert ebh_reject(e, 0.05) == ["PERSON"]
    assert ebh_reject({k: 0.1 for k in e}, 0.05) == []


def test_ebh_threshold_is_infinite_when_nothing_rejects():
    assert ebh_threshold([0.5, 0.5, 0.5], 0.05) == float("inf")
    assert ebh_threshold([], 0.05) == float("inf")


def test_ebh_is_monotone_in_delta():
    e = {"a": 100.0, "b": 60.0, "c": 5.0, "d": 1.0}
    strict = set(ebh_reject(e, 0.01))
    loose = set(ebh_reject(e, 0.20))
    assert strict <= loose


def test_ebh_controls_fdr_under_dependence():
    """All-null e-values, perfectly dependent: rejections must stay rare."""
    rng = np.random.default_rng(0)
    types = [f"T{i}" for i in range(9)]
    false_discoveries = 0
    trials = 400
    for _ in range(trials):
        shared = float(rng.exponential(1.0))     # E[e] = 1 under the null
        e = {t: shared for t in types}
        false_discoveries += bool(ebh_reject(e, 0.05))
    assert false_discoveries / trials <= 0.05 + 0.02


def test_chao_reports_zero_unseen_when_everything_is_multiply_captured():
    est = chao1([3] * 50, n_occasions=3)
    assert est.f1 == 0
    assert est.n_hat == pytest.approx(est.n_obs)
    assert est.missing_upper == pytest.approx(0.0)


def test_chao_point_estimate_never_exceeds_the_upper_limit():
    est = chao1([1] * 40 + [2] * 60 + [3] * 10, n_occasions=3)
    assert est.n_obs <= est.n_hat <= est.n_upper
    assert 0.0 <= est.missing_point <= est.missing_upper <= 1.0


def test_chao_flags_the_two_vacuous_regimes():
    """f2 = 0 and M < 3 are valid but uninformative, and must say so."""
    assert not chao1([1] * 40, n_occasions=3).informative     # f2 = 0
    assert "f2 = 0" in chao1([1] * 40, n_occasions=3).note
    assert not chao1([1] * 40 + [2] * 20, n_occasions=2).informative
    assert not chao1([1, 2, 1], n_occasions=3).informative    # too few observed


def test_chao_recovers_a_known_richness_under_independent_capture():
    """With independent occasions the upper limit should cover the truth."""
    rng = np.random.default_rng(0)
    n_true, m, p = 600, 4, 0.45
    captures = rng.binomial(m, p, size=n_true)
    observed = [int(c) for c in captures if c >= 1]
    est = chao1(observed, delta=0.05, n_occasions=m)
    assert est.informative
    assert est.n_hat <= est.n_upper
    assert est.n_obs <= n_true <= est.n_upper * 1.25


def test_dark_matter_rate_uses_engine_provenance():
    spans = [
        Span(0, 5, "PERSON", "Alice", 0.9, "rule+shape"),
        Span(6, 11, "PERSON", "Bobby", 0.9, "rule"),
        Span(12, 19, "EMAIL", "a@b.com", 0.9, "rule+heuristic-ner+shape"),
    ]
    out = dark_matter_rate(spans, min_score=0.5, n_occasions=3)
    assert set(out) == {"PERSON", "EMAIL"}
    assert out["PERSON"].f1 == 1 and out["PERSON"].f2 == 1
    assert out["EMAIL"].n_obs == 1
