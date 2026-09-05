"""e-BH: FDR control across types under arbitrary dependence.

Per-type drift alarms are strongly positively dependent -- a template change
moves many types at once -- so a procedure that assumes independence or PRDS is
the wrong tool. The e-BH procedure of Wang and Ramdas takes e-values (not
p-values) and controls FDR at level ``delta`` under *arbitrary* dependence:
sort the e-values decreasingly, find

    k* = max { k : e_(k) >= m / (delta * k) },

and reject the ``k*`` largest. Benjamini--Hochberg on p-values would need a
Benjamini--Yekutieli ``log m`` correction to make the same claim, and would
lose the anytime validity that made the e-values available in the first place.
"""
from __future__ import annotations

from typing import Mapping, Sequence


def ebh_threshold(e_values: Sequence[float], delta: float = 0.05) -> float:
    """The rejection cut-off ``m / (delta * k*)``, or ``inf`` if nothing rejects."""
    m = len(e_values)
    if m == 0:
        return float("inf")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    ordered = sorted((float(e) for e in e_values), reverse=True)
    k_star = 0
    for k in range(1, m + 1):
        if ordered[k - 1] >= m / (delta * k):
            k_star = k
    if k_star == 0:
        return float("inf")
    return m / (delta * k_star)


def ebh_reject(e_values: Mapping[str, float], delta: float = 0.05) -> list[str]:
    """Return the keys whose e-values are rejected at FDR level ``delta``."""
    cut = ebh_threshold(list(e_values.values()), delta)
    if cut == float("inf"):
        return []
    return sorted(k for k, v in e_values.items() if float(v) >= cut)
