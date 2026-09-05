"""Estimating what nobody proposed (Proposition 2).

``E[L]`` is observable only with gold labels, and in production there are
none, while the dominant term of eq. (1) -- spans *no* engine proposed -- is by
definition unobserved. AEGIS therefore treats the ``M`` heterogeneous engines
as capture occasions on the population of true spans and applies a
species-richness estimator.

With ``f_k`` the number of proposed-and-verified spans captured by exactly
``k`` engines and ``N_obs = sum_{k>=1} f_k``, the bias-corrected Chao estimator

    N_hat = N_obs + f_1 (f_1 - 1) / (2 (f_2 + 1))

lower-bounds true richness under arbitrary heterogeneity of capture
probabilities, so ``1 - N_obs / N_hat`` *under-states* the candidate-miss rate.
The reported quantity is therefore the conservative

    p_hat^U_y = 1 - N_obs_y / N_hat^U_{y, 1-delta}

built from the log-normal upper confidence limit.

**Honest scope.** Chao's bound assumes independence across occasions and
engines are not independent: a rule engine and a lexical extractor fail on the
same obfuscated surface forms. Positive dependence deflates ``f_1`` relative to
``f_2``, which makes ``N_hat`` conservative as a richness estimate and hence
*optimistic* as a miss estimate. That is why only the upper limit is reported,
why it is validated against gold labels where they exist before being trusted
where they do not, and why it should trigger a labelled audit rather than
replace one.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..types import Span


@dataclass(slots=True)
class ChaoEstimate:
    """Richness point estimate, upper confidence limit, and the frequency counts."""

    n_obs: int
    f1: int
    f2: int
    n_hat: float
    n_upper: float
    delta: float
    n_occasions: int = 0
    informative: bool = True
    note: str = ""

    @property
    def missing_point(self) -> float:
        """``1 - N_obs / N_hat`` -- optimistic, reported only for diagnostics."""
        return 0.0 if self.n_hat <= 0 else max(0.0, 1.0 - self.n_obs / self.n_hat)

    @property
    def missing_upper(self) -> float:
        """``p_hat^U`` -- the conservative production monitor."""
        return 0.0 if self.n_upper <= 0 else max(0.0, 1.0 - self.n_obs / self.n_upper)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_obs": float(self.n_obs),
            "f1": float(self.f1),
            "f2": float(self.f2),
            "n_occasions": float(self.n_occasions),
            "n_hat": round(self.n_hat, 4),
            "n_upper": round(self.n_upper, 4),
            "missing_point": round(self.missing_point, 6),
            "missing_upper": round(self.missing_upper, 6),
            "informative": self.informative,
            "note": self.note,
        }


def _capture_counts(capture_multiplicities: Iterable[int]) -> tuple[int, int, int]:
    counts = Counter(int(k) for k in capture_multiplicities if int(k) >= 1)
    n_obs = sum(counts.values())
    return n_obs, counts.get(1, 0), counts.get(2, 0)


def chao1(
    capture_multiplicities: Iterable[int],
    delta: float = 0.05,
    n_occasions: int = 0,
    min_observed: int = 30,
) -> ChaoEstimate:
    """Bias-corrected Chao1 with the log-normal upper confidence limit.

    ``capture_multiplicities`` is one integer per *observed* span: how many
    engines proposed it.

    The returned estimate carries an ``informative`` flag, because Chao1 has
    two regimes where it is technically valid but practically vacuous and
    silently reporting a huge number would be worse than saying so:

    * ``f_2 = 0``. The correction collapses to ``f_1 (f_1 - 1) / 2``, which is
      quadratic in the singleton count and can exceed the observed population
      by orders of magnitude.
    * ``M < 3`` capture occasions. With two engines almost every span is a
      singleton or a doubleton, so ``f_1``/``f_2`` is dominated by which of two
      detectors happened to fire, not by capture heterogeneity. Three
      architecturally distinct engines is the practical minimum, which is why
      the released default ensemble has three.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    n_obs, f1, f2 = _capture_counts(capture_multiplicities)
    if n_obs == 0:
        return ChaoEstimate(0, 0, 0, 0.0, 0.0, delta, n_occasions, False,
                            "no observed spans")

    extra = f1 * (f1 - 1) / (2.0 * (f2 + 1))
    n_hat = n_obs + extra

    # Variance of the bias-corrected estimator (Chao 1987, eq. 6)
    if f1 > 0:
        a = f1 * (f1 - 1) / (2.0 * (f2 + 1))
        b = f1 * (2 * f1 - 1) ** 2 / (4.0 * (f2 + 1) ** 2)
        c = f1**2 * f2 * (f1 - 1) ** 2 / (4.0 * (f2 + 1) ** 4)
        var = max(a + b + c, 1e-12)
    else:
        var = 1e-12

    # Log-normal limit on the *unseen* count (Chao & Lee 1992 style), which
    # keeps the limit above N_obs by construction.
    if extra > 0:
        ratio = math.sqrt(math.log(1.0 + var / (extra**2)))
        z = _normal_quantile(1.0 - delta)
        n_upper = n_obs + extra * math.exp(z * ratio)
    else:
        n_upper = float(n_obs)

    notes: list[str] = []
    if f2 == 0 and f1 > 1:
        notes.append("f2 = 0: correction is quadratic in f1 and effectively vacuous")
    if 0 < n_occasions < 3:
        notes.append(f"only {n_occasions} capture occasions: need >= 3 to be usable")
    if n_obs < min_observed:
        notes.append(f"only {n_obs} observed spans (< {min_observed})")
    return ChaoEstimate(
        n_obs, f1, f2, n_hat, n_upper, delta, n_occasions,
        informative=not notes, note="; ".join(notes),
    )


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def dark_matter_rate(
    candidates: Sequence[Span],
    verified_only: bool = True,
    min_score: float = 0.5,
    delta: float = 0.05,
    per_type: bool = True,
    n_occasions: int = 0,
) -> dict[str, ChaoEstimate]:
    """Per-type candidate-miss estimates from engine capture multiplicities.

    A candidate's ``engine`` field lists every engine that proposed it (the
    ensemble merges overlapping same-type proposals and joins the names with
    ``+``), so its capture multiplicity is that list's length. ``verified_only``
    restricts the population to candidates the pipeline would act on, which is
    the population whose richness we care about.
    """
    buckets: dict[str, list[int]] = {}
    if n_occasions == 0:
        seen: set[str] = set()
        for span in candidates:
            seen.update(e for e in span.engine.split("+") if e)
        n_occasions = len(seen)
    for span in candidates:
        if verified_only and span.score < min_score:
            continue
        key = span.type if per_type else "__ALL__"
        multiplicity = len([e for e in span.engine.split("+") if e])
        buckets.setdefault(key, []).append(max(multiplicity, 1))
    return {
        k: chao1(v, delta=delta, n_occasions=n_occasions)
        for k, v in sorted(buckets.items())
    }
