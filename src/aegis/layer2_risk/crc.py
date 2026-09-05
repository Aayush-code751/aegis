"""Conformal risk control at the document level (Theorem 1).

For a monotone, right-continuous, ``[0, B]``-valued per-document loss with
``L(d, lambda_max) = 0``, the selection

    lambda_hat = inf { lambda : (n / (n+1)) * Rhat_n(lambda) + B / (n+1) <= alpha }

satisfies ``E[L(D_{n+1}, lambda_hat)] <= alpha`` for an exchangeable
calibration set plus test point. Spans inside a document may be arbitrarily
dependent: they enter ``L`` only through a per-document average, and the
exchangeability the theorem needs is over *documents*, which is how the stream
is sampled. Score ties are harmless for the same reason -- the statement is
about an expectation of a bounded average, not about the rank of a scalar.
"""
from __future__ import annotations

from typing import Sequence

from .risks import Lambda


def crc_bound(mean_risk: float, n: int, B: float = 1.0) -> float:
    """The finite-sample inflated risk ``n/(n+1) * Rhat + B/(n+1)``."""
    if n <= 0:
        return B
    return (n / (n + 1.0)) * mean_risk + B / (n + 1.0)


def crc_select(
    grid: Sequence[Lambda],
    risks: Sequence[Sequence[float]],
    alpha: float,
    B: float = 1.0,
    weights: Sequence[float] | None = None,
) -> tuple[int | None, list[float]]:
    """Return ``(index of lambda_hat in grid, inflated risk per grid point)``.

    ``risks[i][j]`` is ``L(D_i, grid[j])``. ``weights`` optionally supplies the
    normalised weighted-CRC probabilities of Layer III, in which case the
    calibration average is replaced by the weighted average and the leftover
    mass already sits in the caller's weights (see
    :func:`aegis.layer3_shift.weighted_crc.weighted_crc_select`).

    Because ``L`` is non-increasing along the path, the feasible set is an
    upper set and its infimum is the *least aggressive* masking that still
    certifies -- which is the point: nothing is masked that the level does not
    require.
    """
    n = len(risks)
    g = len(grid)
    if g == 0:
        raise ValueError("empty grid")
    inflated: list[float] = []
    for j in range(g):
        if n == 0:
            inflated.append(B)
            continue
        if weights is None:
            mean = sum(risks[i][j] for i in range(n)) / n
            inflated.append(crc_bound(mean, n, B))
        else:
            # weights already include the (1/(sum w + sup w)) normalisation and
            # the leftover mass acts as the +B/(n+1) inflation
            inflated.append(sum(weights[i] * risks[i][j] for i in range(n)))
    for j in range(g):
        if inflated[j] <= alpha:
            return j, inflated
    return None, inflated
