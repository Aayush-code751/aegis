"""Weighted conformal risk control with *estimated* weights (Theorem 3).

Weighted CRC replaces the uniform calibration average of eq. (4) by

    sum_i ptilde_i * L(D_i, lambda),
    ptilde_i = w_hat(D_i) / (sum_{j<=n} w_hat(D_j) + sup w_hat),

which is exact when ``w_hat`` really is ``dQ/dP``. The leftover mass
``sup w_hat / (sum_j w_hat(D_j) + sup w_hat)`` plays the role of the
``B / (n + 1)`` inflation in the unweighted bound: it is the probability the
test point receives, so setting the loss there to its supremum ``B`` recovers
a conservative statement.

The case that actually occurs -- an *estimated* ratio -- is what Thm. 3
prices:

    E_Q[L] <= alpha + B * d_TV(Q, Qhat) = alpha + (B/2) * E_P|w_hat - w|,

so the whole cost of not knowing the true ratio is one total-variation term.
``tv_inflation`` computes it from a bound on the ratio-estimation error, in
either total-variation or chi^2 form (Cauchy--Schwarz gives
``2 d_TV <= sqrt(chi^2)``).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..layer2_risk.risks import Lambda


def weighted_crc_weights(w_hat: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return ``ptilde`` of Thm. 3, including the test point's leftover mass.

    The returned array has length ``n``; the missing mass ``1 - sum(ptilde)``
    is exactly the test point's share and is what makes the bound conservative
    without any further inflation term.
    """
    w = np.asarray(w_hat, dtype=np.float64)
    if w.size == 0:
        return w
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    denom = w.sum() + w.max()
    if denom <= 0:
        return np.full(w.size, 1.0 / w.size)
    return w / denom


def tv_inflation(
    B: float = 1.0,
    tv: float | None = None,
    chi2: float | None = None,
    mean_abs_error: float | None = None,
) -> float:
    """The ``B * d_TV(Q, Qhat)`` term of Thm. 3.

    Supply exactly one of a total-variation distance, a chi^2 divergence
    (converted by ``d_TV <= sqrt(chi2)/2``), or ``E_P|w_hat - w|`` (which is
    ``2 d_TV``).
    """
    provided = [x for x in (tv, chi2, mean_abs_error) if x is not None]
    if len(provided) != 1:
        raise ValueError("supply exactly one of tv=, chi2=, mean_abs_error=")
    if tv is not None:
        distance = tv
    elif chi2 is not None:
        distance = 0.5 * math.sqrt(max(chi2, 0.0))
    else:
        distance = 0.5 * max(mean_abs_error, 0.0)  # type: ignore[arg-type]
    return float(B * min(max(distance, 0.0), 1.0))


def weighted_crc_select(
    grid: Sequence[Lambda],
    risks: Sequence[Sequence[float]],
    alpha: float,
    w_hat: Sequence[float] | np.ndarray | None = None,
    B: float = 1.0,
    ratio_error_chi2: float | None = None,
) -> tuple[int | None, list[float], float]:
    """Weighted CRC selection.

    Returns ``(index of lambda_hat, weighted risk per grid point, budget)``,
    where ``budget = alpha - tv_inflation(...)`` is the level the weighted
    average must respect so that the *deployment* risk still respects
    ``alpha``. When ``ratio_error_chi2`` is ``None`` the inflation is omitted
    and the selection is exact only under a correct ratio -- which is the gap
    Thm. 3 exists to close, so callers auditing a real deployment should pass
    a bound.
    """
    n = len(risks)
    g = len(grid)
    if g == 0:
        raise ValueError("empty grid")
    inflation = 0.0 if ratio_error_chi2 is None else tv_inflation(B, chi2=ratio_error_chi2)
    budget = alpha - inflation
    if n == 0:
        return None, [B] * g, budget

    if w_hat is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(w_hat, dtype=np.float64)
        if w.size != n:
            raise ValueError("w_hat length must match the number of calibration documents")
    p = weighted_crc_weights(w)
    leftover = max(0.0, 1.0 - float(p.sum()))

    R = np.asarray(risks, dtype=np.float64)
    weighted = (p[:, None] * R).sum(axis=0) + leftover * B
    values = [float(v) for v in weighted]
    if budget <= 0:
        return None, values, budget
    for j in range(g):
        if values[j] <= budget:
            return j, values, budget
    return None, values, budget
