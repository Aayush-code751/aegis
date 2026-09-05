"""Super-uniform p-values for a bounded risk.

Learn-then-Test needs, for each grid point, a valid p-value for the *risk*
null

    H_0 : E[X] >= alpha        with X in [0, 1],

so that a **small** p-value is evidence the risk is genuinely below the
declared level. (Note the direction: this is the opposite of the drift null
that Layer IV monitors, ``H_0 : E[L] <= alpha``, where an alarm fires when
leakage *exceeds* the level. Same machinery, mirrored.)

Two families are computed and the smaller is taken, as in the paper:

* **Hoeffding--Bentkus** -- the fixed-sample bound of the LTT paper: the
  minimum of a Hoeffding tail and a Bentkus binomial tail, evaluated at
  ``min(Rhat, alpha)`` so it is exactly 1 when the data do not undercut the
  null at all.
* **Betting / capital process** -- wealth from a bet that ``X`` sits below
  ``alpha``, with predictable tuning-free ONS stakes; ``1 / sup_t K_t`` is a
  p-value by Ville's inequality, valid at every ``t`` and hence at a
  data-dependent stopping time. This is the same construction Layer IV
  monitors with, which is deliberate: the statistic the certificate is built
  from is the statistic deployment watches.

Taking a plain ``min`` of two valid p-values is not itself valid, so the
reported value is ``min(1, 2 * min(p_HB, p_bet))``. The factor two is cheap
and keeps the guarantee exact; ``combine="hb"`` / ``"betting"`` select a single
family for ablations.
"""
from __future__ import annotations

import math
from typing import Literal, Sequence

_EPS = 1e-12


def _kl(a: float, b: float) -> float:
    """Binary KL divergence ``h_1(a, b) = a log(a/b) + (1-a) log((1-a)/(1-b))``."""
    a = min(max(a, _EPS), 1.0 - _EPS)
    b = min(max(b, _EPS), 1.0 - _EPS)
    return a * math.log(a / b) + (1.0 - a) * math.log((1.0 - a) / (1.0 - b))


def _binom_cdf(k: int, n: int, p: float) -> float:
    """``P(Bin(n, p) <= k)`` by a stable log-space forward sum."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    p = min(max(p, _EPS), 1.0 - _EPS)
    log_p, log_q = math.log(p), math.log1p(-p)
    log_term = n * log_q
    total = math.exp(log_term)
    for i in range(1, k + 1):
        log_term += math.log((n - i + 1) / i) + log_p - log_q
        total += math.exp(log_term)
    return min(1.0, total)


def hoeffding_bentkus_pvalue(mean: float, n: int, alpha: float) -> float:
    """HB p-value for the risk null ``H_0 : E[X] >= alpha``, ``X in [0, 1]``.

    ``mean`` is the empirical risk over ``n`` observations. Returns 1.0 when
    the empirical risk is at or above ``alpha``: the data give no evidence
    against a null they agree with.
    """
    if n <= 0:
        return 1.0
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    mean = min(max(mean, 0.0), 1.0)
    if mean >= alpha:
        return 1.0
    hoeffding = math.exp(-n * _kl(mean, alpha))
    bentkus = math.e * _binom_cdf(int(math.ceil(n * mean)), n, alpha)
    return float(min(1.0, hoeffding, bentkus))


def betting_pvalue(
    values: Sequence[float],
    alpha: float,
    max_stake_fraction: float = 0.5,
) -> float:
    """``1 / sup_t K_t`` for a capital process betting against ``E[X] >= alpha``.

    ``K_t = prod_{i<=t} (1 + kappa_i (alpha - X_i))`` with predictable
    ``kappa_i`` in ``[0, max_stake_fraction / (1 - alpha))``, which keeps the
    wealth strictly positive because ``alpha - X_i >= -(1 - alpha)``. Under the
    null ``E[alpha - X] <= 0``, so ``K`` is a non-negative supermartingale with
    ``K_0 = 1`` and Ville's inequality makes ``1 / sup_t K_t`` super-uniform.

    Stakes follow the tuning-free online-Newton-step rule: the running plug-in
    estimate of the optimal bet, using only observations already seen.
    """
    if not values:
        return 1.0
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    cap = max_stake_fraction / (1.0 - alpha)
    wealth = 1.0
    best = 1.0
    mu_hat, var_hat, count = 0.5, 0.25, 0
    for raw in values:
        x = min(max(float(raw), 0.0), 1.0)
        kappa = 0.0
        if count > 0:
            edge = alpha - mu_hat          # positive when the risk looks small
            if edge > 0.0:
                kappa = min(cap, edge / max(var_hat, 1e-4))
        wealth *= 1.0 + kappa * (alpha - x)
        if wealth <= 0.0:
            return 1.0
        best = max(best, wealth)
        count += 1
        delta = x - mu_hat
        mu_hat += delta / count
        var_hat += (delta * (x - mu_hat) - var_hat) / count
    return float(min(1.0, 1.0 / best))


def bounded_mean_pvalue(
    values: Sequence[float],
    alpha: float,
    combine: Literal["min2", "hb", "betting"] = "min2",
) -> float:
    """The p-value LTT consumes for one grid point (null: ``E[X] >= alpha``)."""
    n = len(values)
    if n == 0:
        return 1.0
    mean = sum(values) / n
    if combine == "hb":
        return hoeffding_bentkus_pvalue(mean, n, alpha)
    if combine == "betting":
        return betting_pvalue(values, alpha)
    p_hb = hoeffding_bentkus_pvalue(mean, n, alpha)
    p_bet = betting_pvalue(values, alpha)
    return float(min(1.0, 2.0 * min(p_hb, p_bet)))
