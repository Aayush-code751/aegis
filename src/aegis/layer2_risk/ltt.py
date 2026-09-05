"""Learn-then-Test with simultaneous leakage and utility control (Theorem 2).

For each grid point we form the intersection--union null

    H_lambda :  { E L(.,lambda) > alpha }  union  { E O(.,lambda) > gamma }

and take ``p_lambda = max(p^L_lambda, p^O_lambda)``, which is valid for
``H_lambda`` by the intersection--union principle: rejecting a union null
requires rejecting both components, so the maximum of two valid p-values is
valid.

Multiplicity is handled by **two fixed sequences**, which is the concrete form
the monotone path of Sec. III gives Theorem 2:

* ``E[L]`` is non-increasing along the path, so the certifiable set for
  leakage is an *upper* set. We test from the most-masking end downwards and
  stop at the first non-rejection.
* ``E[O]`` is non-decreasing along the path, so its certifiable set is a
  *lower* set. We test from the least-masking end upwards and stop likewise.

Each sequence gets level ``delta / 2``; a fixed sequence controls FWER at its
level for any pre-specified order, and the order here is fixed by the path, not
by the data. The intersection of the two certified sets is ``Lambda_hat``. This
is uniformly more powerful than Bonferroni over the whole grid, which would
spend ``delta / |grid|`` per point; ``method="bonferroni"`` is kept for the
ablation that shows it.

``Lambda_hat = {}`` is a valid and informative output: it means *no operating
point certifies ``(alpha, gamma)`` on this data*, and AEGIS then reports no
certificate and routes the stream to review rather than emitting a threshold
it cannot support.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from .pvalues import bounded_mean_pvalue
from .risks import Lambda


@dataclass(slots=True)
class LTTResult:
    """What the calibrator certified."""

    certified: list[int] = field(default_factory=list)
    p_leak: list[float] = field(default_factory=list)
    p_over: list[float] = field(default_factory=list)
    mean_leak: list[float] = field(default_factory=list)
    mean_over: list[float] = field(default_factory=list)
    t_leak_index: int | None = None
    t_over_index: int | None = None
    alpha: float = 0.0
    gamma: float = 0.0
    delta: float = 0.05
    method: str = "fixed-sequence"

    @property
    def empty(self) -> bool:
        return not self.certified

    def summary(self) -> dict[str, object]:
        return {
            "n_certified": len(self.certified),
            "empty": self.empty,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "delta": self.delta,
            "method": self.method,
            "leak_frontier_index": self.t_leak_index,
            "over_frontier_index": self.t_over_index,
        }


def _column(matrix: Sequence[Sequence[float]], j: int) -> list[float]:
    return [row[j] for row in matrix]


def learn_then_test(
    grid: Sequence[Lambda],
    leak: Sequence[Sequence[float]],
    over: Sequence[Sequence[float]],
    alpha: float,
    gamma: float,
    delta: float = 0.05,
    method: Literal["fixed-sequence", "bonferroni"] = "fixed-sequence",
    combine: str = "min2",
) -> LTTResult:
    """Certify leakage and over-masking simultaneously over the grid.

    ``leak[i][j]`` and ``over[i][j]`` are the two risks of document ``i`` at
    grid point ``j``; the grid must be ordered least-masking first.
    """
    n, g = len(leak), len(grid)
    if g == 0:
        raise ValueError("empty grid")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")

    p_leak = [1.0] * g
    p_over = [1.0] * g
    mean_leak = [0.0] * g
    mean_over = [0.0] * g
    for j in range(g):
        lj = _column(leak, j) if n else []
        oj = _column(over, j) if n else []
        mean_leak[j] = sum(lj) / n if n else 1.0
        mean_over[j] = sum(oj) / n if n else 1.0
        p_leak[j] = bounded_mean_pvalue(lj, alpha, combine=combine) if n else 1.0
        p_over[j] = bounded_mean_pvalue(oj, gamma, combine=combine) if n else 1.0

    result = LTTResult(
        p_leak=p_leak, p_over=p_over, mean_leak=mean_leak, mean_over=mean_over,
        alpha=alpha, gamma=gamma, delta=delta, method=method,
    )

    if method == "bonferroni":
        level = delta / (2 * g)
        certified = [j for j in range(g) if p_leak[j] <= level and p_over[j] <= level]
        result.certified = certified
        result.t_leak_index = min(certified) if certified else None
        result.t_over_index = max(certified) if certified else None
        return result

    half = delta / 2.0
    # leakage: walk down from the most-masking end, stop at first failure
    leak_ok: list[int] = []
    for j in range(g - 1, -1, -1):
        if p_leak[j] <= half:
            leak_ok.append(j)
        else:
            break
    # over-masking: walk up from the least-masking end, stop at first failure
    over_ok: list[int] = []
    for j in range(g):
        if p_over[j] <= half:
            over_ok.append(j)
        else:
            break

    leak_set, over_set = set(leak_ok), set(over_ok)
    result.certified = sorted(leak_set & over_set)
    result.t_leak_index = min(leak_ok) if leak_ok else None
    result.t_over_index = max(over_ok) if over_ok else None
    return result
