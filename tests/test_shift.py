"""Layer III: Theorem 3 (weighted CRC) and Theorem 4 (the certified radius)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from aegis.layer3_shift import (
    certified_radius,
    chi2_divergence,
    chi2_dro_worst_case,
    finite_sample_certified_radius,
    finite_sample_worst_case,
    fit_density_ratio,
    tv_inflation,
    upper_confidence_mean,
    weighted_crc_weights,
)
from aegis.types import Document


# --------------------------------------------------------------------------
# Theorem 4: the chi^2 dual
# --------------------------------------------------------------------------


def test_worst_case_at_zero_radius_is_the_plain_mean():
    losses = np.random.default_rng(0).beta(2, 20, size=500)
    assert chi2_dro_worst_case(losses, 0.0) == pytest.approx(float(losses.mean()), abs=1e-12)


def test_worst_case_is_monotone_and_continuous_in_rho():
    losses = np.random.default_rng(1).beta(2, 20, size=500)
    grid = np.linspace(0.0, 3.0, 60)
    values = [chi2_dro_worst_case(losses, r) for r in grid]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))
    jumps = [abs(b - a) for a, b in zip(values, values[1:])]
    assert max(jumps) < 0.05, "the dual should be continuous, not jumpy"


def test_worst_case_matches_the_closed_form_in_the_inactive_hinge_regime():
    """Below ess-inf the hinge is inactive and the dual equals the expansion."""
    losses = np.random.default_rng(2).beta(2, 20, size=2000)
    m, v = float(losses.mean()), float(losses.var())
    for rho in (0.05, 0.1, 0.3):
        eta_star = m - math.sqrt(v / (2 * rho))
        if eta_star <= losses.min():
            assert chi2_dro_worst_case(losses, rho) == pytest.approx(
                m + math.sqrt(2 * rho * v), rel=1e-6
            )


def test_dual_upper_bounds_explicit_worst_case_reweightings():
    """No mean-one reweighting inside the ball may exceed the dual value."""
    rng = np.random.default_rng(3)
    losses = rng.beta(2, 20, size=300)
    for rho in (0.1, 0.5, 1.0):
        dual = chi2_dro_worst_case(losses, rho)
        for _ in range(400):
            w = rng.random(losses.size) * 4.0
            w /= w.mean()
            if chi2_divergence(w) <= rho:
                assert float((w * losses).mean()) <= dual + 1e-9


def test_certified_radius_is_the_crossing_point():
    losses = np.random.default_rng(4).beta(2, 20, size=800)
    alpha = 0.15
    rho = certified_radius(losses, alpha, rho_max=20.0)
    assert 0 < rho < 20.0
    assert chi2_dro_worst_case(losses, rho) <= alpha + 1e-3
    assert chi2_dro_worst_case(losses, rho * 1.2) > alpha


def test_certified_radius_is_zero_when_the_level_is_already_violated():
    losses = np.full(200, 0.3)
    assert certified_radius(losses, 0.10) == 0.0


def test_a_deterministic_loss_has_an_unbounded_radius():
    """Zero variance means no reweighting can move the mean: rho* saturates."""
    losses = np.full(500, 0.01)
    assert certified_radius(losses, 0.10, rho_max=25.0) == pytest.approx(25.0)


def test_finite_sample_radius_is_conservative():
    losses = np.random.default_rng(5).beta(2, 20, size=600)
    alpha = 0.15
    assert finite_sample_certified_radius(losses, alpha, 0.05) <= certified_radius(losses, alpha)


def test_finite_sample_worst_case_dominates_the_population_dual():
    losses = np.random.default_rng(6).beta(2, 20, size=600)
    for rho in (0.0, 0.3, 1.0):
        assert finite_sample_worst_case(losses, rho, 0.05) >= chi2_dro_worst_case(losses, rho)


def test_upper_confidence_mean_brackets_and_tightens():
    rng = np.random.default_rng(7)
    small = upper_confidence_mean(rng.random(50) * 0.1, 0.05)
    large = upper_confidence_mean(rng.random(5000) * 0.1, 0.05)
    assert small > large
    assert upper_confidence_mean([], 0.05) == 1.0


# --------------------------------------------------------------------------
# Theorem 3: weighted CRC
# --------------------------------------------------------------------------


def test_weighted_crc_weights_reserve_mass_for_the_test_point():
    w = np.ones(10)
    p = weighted_crc_weights(w)
    assert p.sum() < 1.0
    assert 1.0 - p.sum() == pytest.approx(1.0 / 11.0)
    assert list(p) == pytest.approx([1.0 / 11.0] * 10)


def test_weighted_crc_weights_reject_negatives():
    with pytest.raises(ValueError):
        weighted_crc_weights(np.array([1.0, -0.5]))


def test_tv_inflation_conversions_agree():
    assert tv_inflation(1.0, tv=0.2) == pytest.approx(0.2)
    assert tv_inflation(1.0, mean_abs_error=0.4) == pytest.approx(0.2)
    assert tv_inflation(1.0, chi2=0.16) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        tv_inflation(1.0)


def test_chi2_divergence_is_zero_for_uniform_weights():
    assert chi2_divergence(np.ones(50)) == pytest.approx(0.0)
    # mean-one weights {0, 2}: 0.5 * mean((w - 1)^2) = 0.5 * 1 = 0.5
    assert chi2_divergence(np.array([0.0, 2.0])) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# The audit: cross-fitting and the separability guard
# --------------------------------------------------------------------------


def _docs(texts: list[str]) -> list[Document]:
    return [Document(f"d{i}", t) for i, t in enumerate(texts)]


def test_no_shift_is_measured_as_small():
    rng = np.random.default_rng(8)
    words = ["wire", "memo", "account", "balance", "transfer", "branch", "ledger"]
    def sample() -> str:
        return " ".join(rng.choice(words, size=30))
    ratio = fit_density_ratio(_docs([sample() for _ in range(120)]),
                              _docs([sample() for _ in range(120)]), epochs=120)
    assert not ratio.separable
    assert ratio.rho_hat < 0.5
    assert 0.3 < ratio.auc < 0.7


def test_disjoint_support_reports_an_infinite_radius():
    """A perfectly separable deployment must not be reported as "no shift".

    This is the failure mode the guard exists for: the plug-in ratio collapses
    to ~0 exactly when the true chi^2 divergence is unbounded.
    """
    source = _docs([f"KYC onboarding summary {i}: borrower Melanie Riley, IBAN DE89 3704." for i in range(80)])
    target = _docs([f"Safety data sheet {i}: toxicity assessment of substance X, no personal data." for i in range(80)])
    ratio = fit_density_ratio(source, target, epochs=200)
    assert ratio.separable
    assert math.isinf(ratio.rho_hat)
    assert ratio.auc > 0.95


def test_weights_align_with_the_calibration_sample():
    source = _docs([f"wire memo {i} account balance transfer" for i in range(60)])
    target = _docs([f"loan note {i} interest schedule amortisation" for i in range(60)])
    ratio = fit_density_ratio(source, target, epochs=80)
    assert ratio.weights.size == len(source)
    assert float(ratio.weights.mean()) == pytest.approx(1.0, abs=1e-6)
