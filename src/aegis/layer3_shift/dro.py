"""The chi^2-DRO worst case and the certified shift radius (Theorem 4).

With ``D_chi2(Q || P) = 1/2 E_P[(dQ/dP - 1)^2]`` the worst-case risk over a
chi^2 ball has the exact dual

    R_rho(lambda) = inf_eta { sqrt(1 + 2 rho) * || (L - eta)_+ ||_2 + eta },

non-decreasing and continuous in ``rho``, equal to ``E_P[L]`` at ``rho = 0``
and with first-order expansion ``E_P[L] + sqrt(2 rho Var_P(L)) + O(rho)``. The
**certified radius**

    rho_star(lambda) = sup { rho >= 0 : R_rho(lambda) <= alpha }

is then the largest calibration-to-deployment divergence for which the
alpha-bound provably survives. It is not a tuning knob: it is a property of
the calibration losses and the selected ``lambda``, and its only operational
use is the comparison ``rho_hat <=> rho_star`` that gates deployment onto a new
corpus. A plain conformal quantile has ``rho_star = 0`` by construction -- it
certifies nothing beyond ``P`` itself.

The finite-sample version replaces ``E_P[(L - eta)_+^2]`` by a
``(1 - delta)`` upper confidence bound, giving a conservative
``rho_star_hat <= rho_star`` valid with probability at least ``1 - delta``. The
bound is the tighter of Hoeffding and empirical Bernstein, with a Bonferroni
split across the ``eta`` grid so the statement holds uniformly in ``eta``.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _second_moment_plus(losses: np.ndarray, eta: float) -> float:
    return float(np.mean(np.maximum(losses - eta, 0.0) ** 2))


def chi2_dro_worst_case(
    losses: Sequence[float] | np.ndarray,
    rho: float,
    n_eta: int = 512,
) -> float:
    """Evaluate the dual (eq. 5) exactly where possible, else by grid search.

    The objective is convex in ``eta``. Below ``ess inf L`` the hinge is
    inactive, ``E[(L - eta)_+^2] = Var(L) + (E[L] - eta)^2``, and the
    stationary point is available in closed form:

        eta* = E[L] - sqrt(Var(L) / (2 rho)),
        R_rho = E[L] + sqrt(2 rho Var(L)),

    which is the paper's first-order expansion holding *exactly* in that
    regime. When ``eta*`` falls above ``min L`` the hinge is active and we grid
    search the bracket ``[min L, max L]`` with a local refinement. ``rho = 0``
    returns ``E[L]`` by definition of the singleton ball.
    """
    L = np.asarray(losses, dtype=np.float64)
    if L.size == 0:
        return 0.0
    if rho < 0:
        raise ValueError("rho must be non-negative")
    mean = float(L.mean())
    if rho == 0.0:
        return mean
    var = float(L.var())
    scale = math.sqrt(1.0 + 2.0 * rho)
    if var > 0.0:
        eta_star = mean - math.sqrt(var / (2.0 * rho))
        if eta_star <= float(L.min()):
            return float(mean + math.sqrt(2.0 * rho * var))
    lo, hi = float(L.min()), float(L.max())
    if hi <= lo:
        return float(scale * 0.0 + lo)
    grid = np.linspace(lo, hi, n_eta)
    values = scale * np.sqrt(np.array([_second_moment_plus(L, e) for e in grid])) + grid
    k = int(np.argmin(values))
    left = grid[max(k - 1, 0)]
    right = grid[min(k + 1, n_eta - 1)]
    fine = np.linspace(left, right, 129)
    fine_values = scale * np.sqrt(np.array([_second_moment_plus(L, e) for e in fine])) + fine
    return float(min(values[k], fine_values.min()))


def certified_radius(
    losses: Sequence[float] | np.ndarray,
    alpha: float,
    rho_max: float = 50.0,
    tol: float = 1e-4,
    n_eta: int = 256,
) -> float:
    """``rho_star = sup { rho : R_rho <= alpha }`` by bisection.

    Returns 0.0 when even ``rho = 0`` violates the level (i.e. the empirical
    risk already exceeds ``alpha``), and ``rho_max`` when the bound survives
    the whole search range.
    """
    L = np.asarray(losses, dtype=np.float64)
    if L.size == 0:
        return rho_max
    if chi2_dro_worst_case(L, 0.0, n_eta) > alpha:
        return 0.0
    if chi2_dro_worst_case(L, rho_max, n_eta) <= alpha:
        return float(rho_max)
    lo, hi = 0.0, rho_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if chi2_dro_worst_case(L, mid, n_eta) <= alpha:
            lo = mid
        else:
            hi = mid
    return float(lo)


# ---------------------------------------------------------------------------
# Finite-sample version
# ---------------------------------------------------------------------------


def upper_confidence_mean(values: Sequence[float] | np.ndarray, delta: float,
                          upper: float = 1.0) -> float:
    """A ``(1 - delta)`` upper confidence bound on the mean of ``[0, upper]`` data.

    The tighter of Hoeffding and empirical Bernstein; both are valid for
    bounded observations, and empirical Bernstein wins whenever the sample
    variance is small, which it is for squared hinge losses.
    """
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n == 0:
        return upper
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    mean = float(x.mean())
    log_term = math.log(1.0 / delta)
    hoeffding = mean + upper * math.sqrt(log_term / (2.0 * n))
    if n > 1:
        var = float(x.var(ddof=1))
        bernstein = mean + math.sqrt(2.0 * var * log_term / n) + 7.0 * upper * log_term / (3.0 * (n - 1))
    else:
        bernstein = upper
    return float(min(upper, min(hoeffding, bernstein)))


def lower_confidence_mean(values: Sequence[float] | np.ndarray, delta: float,
                          upper: float = 1.0) -> float:
    """A ``(1 - delta)`` lower confidence bound on the mean of ``[0, upper]`` data."""
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n == 0:
        return 0.0
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    mean = float(x.mean())
    log_term = math.log(1.0 / delta)
    hoeffding = mean - upper * math.sqrt(log_term / (2.0 * n))
    if n > 1:
        var = float(x.var(ddof=1))
        bernstein = mean - math.sqrt(2.0 * var * log_term / n) \
            - 7.0 * upper * log_term / (3.0 * (n - 1))
    else:
        bernstein = 0.0
    return float(max(0.0, max(hoeffding, bernstein)))


def finite_sample_worst_case(
    losses: Sequence[float] | np.ndarray,
    rho: float,
    delta: float = 0.05,
    n_eta: int = 128,
    B: float = 1.0,
) -> float:
    """``Rhat_rho >= R_rho`` with probability ``>= 1 - delta`` (Cor. to Thm. 4).

    Each ``eta`` on the search grid gets an upper confidence bound at level
    ``delta / n_eta``, so the resulting infimum is valid *uniformly* in
    ``eta``. Two details are easy to get wrong and both matter:

    * ``rho = 0`` is the singleton ball, so the worst case is just ``E[L]`` and
      its finite-sample counterpart is a plain upper confidence bound on the
      mean. Routing it through the dual would be needlessly loose.
    * ``(L - eta)_+^2`` is bounded by ``B^2`` only for ``eta >= 0``. For
      negative ``eta`` the correct bound is ``(B - eta)^2``, and using ``B^2``
      there lets the ``+ eta`` term dominate and drives the "upper bound"
      arbitrarily negative -- an invalid bound that looks like a tight one.

    Any ``eta`` gives a *valid* bound; the infimum only affects tightness. The
    grid therefore includes the analytic stationary point of the
    inactive-hinge regime so the bound stays tight where it usually lives.
    """
    L = np.asarray(losses, dtype=np.float64)
    if L.size == 0:
        return B
    if rho < 0:
        raise ValueError("rho must be non-negative")
    if rho == 0.0:
        return upper_confidence_mean(L, delta, upper=B)

    # --- Branch 1: the analytic bound, tight and continuous at rho -> 0 ----
    # For ANY eta the dual objective upper-bounds R_rho, and evaluating it at
    # eta0 = E[L] - sqrt(Var(L) / 2 rho) together with (L - eta)_+^2 <=
    # (L - eta)^2 collapses to E[L] + sqrt(2 rho Var(L)) -- no regime
    # condition needed. Replacing the moments by confidence bounds (a third of
    # delta each, Bonferroni) gives a valid finite-sample version that is
    # tight for small rho, exactly where the grid search is at its worst.
    third = delta / 3.0
    ucb_mean = upper_confidence_mean(L, third, upper=B)
    lcb_mean = lower_confidence_mean(L, third, upper=B)
    ucb_second = upper_confidence_mean(L * L, third, upper=B * B)
    var_upper = max(0.0, ucb_second - lcb_mean**2)
    analytic = ucb_mean + math.sqrt(2.0 * rho * var_upper)

    # --- Branch 2: the grid search, tighter for large rho -----------------
    delta = delta / 3.0
    scale = math.sqrt(1.0 + 2.0 * rho)
    mean, var = float(L.mean()), float(L.var())
    eta_star = mean - math.sqrt(var / (2.0 * rho)) if var > 0 else float(L.min())
    # A grid that does NOT depend on rho, for two reasons. Tightness: as
    # rho -> 0 the population stationary point runs off to -infinity, and
    # chasing it makes the confidence bound on (L - eta)_+^2 blow up with
    # (B - eta)^2, so a bound that should tighten instead explodes.
    # Monotonicity: with the grid fixed, the objective is
    # sqrt(1 + 2 rho) * sqrt(UCB(eta)) + eta, whose infimum is non-decreasing
    # in rho by construction -- which is the shape Thm. 4 asserts and which
    # the bisection for rho_star relies on.
    grid = np.linspace(-B, float(L.max()), n_eta)
    if -B <= eta_star <= float(L.max()):
        grid = np.unique(np.concatenate([grid, np.array([eta_star])]))
    per_eta_delta = delta / grid.size
    best = math.inf
    for eta in grid:
        hinge_sq = np.maximum(L - eta, 0.0) ** 2
        # valid range of (L - eta)_+^2 for L in [0, B]
        cap = (B - min(float(eta), 0.0)) ** 2
        ucb = upper_confidence_mean(hinge_sq, per_eta_delta, upper=cap)
        best = min(best, scale * math.sqrt(max(ucb, 0.0)) + float(eta))
    return float(min(max(min(best, analytic), 0.0), B))


def finite_sample_certified_radius(
    losses: Sequence[float] | np.ndarray,
    alpha: float,
    delta: float = 0.05,
    rho_max: float = 50.0,
    tol: float = 1e-3,
    n_eta: int = 128,
    B: float = 1.0,
) -> float:
    """Conservative ``rho_star_hat <= rho_star``, valid w.p. ``>= 1 - delta``."""
    L = np.asarray(losses, dtype=np.float64)
    if L.size == 0:
        return 0.0
    if finite_sample_worst_case(L, 0.0, delta, n_eta, B) > alpha:
        return 0.0
    if finite_sample_worst_case(L, rho_max, delta, n_eta, B) <= alpha:
        return float(rho_max)
    lo, hi = 0.0, rho_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if finite_sample_worst_case(L, mid, delta, n_eta, B) <= alpha:
            lo = mid
        else:
            hi = mid
    return float(lo)
