"""Anytime-valid drift alarms (Theorem 5).

A certificate that is only checked at recalibration time fails silently
between checks; a fixed-sample test applied repeatedly is invalid. The fix is
an e-process. For type ``y`` with realised per-document leakage
``L^y_t in [0, 1]`` and any predictable ``kappa_t in [0, 1/alpha)``,

    E^y_T = prod_{t <= T} (1 + kappa_t (L^y_t - alpha))

is a non-negative supermartingale with ``E^y_0 = 1`` under
``H^y_0 : E[L^y_t | F_{t-1}] <= alpha``. Ville's inequality then gives

    P( exists T >= 1 : E^y_T >= 1/delta ) <= delta,

so the threshold may be watched *continuously* with no multiplicity penalty --
which is exactly what a repeated fixed-window test cannot claim, and why its
empirical false-alarm rate runs several times its nominal level.

Stakes come from the tuning-free online-Newton-step rule: a running plug-in
estimate of the optimal bet, computed from observations already seen so it
stays predictable. An :class:`EDetector` sums changepoint-anchored e-processes,
which is what buys a detection-delay guarantee rather than mere validity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(slots=True)
class ONSBettor:
    """Online-Newton-step stake selection, truncated to keep wealth positive.

    ``kappa_t`` is chosen from the running mean and variance of the observed
    ``L`` values only, so it is ``F_{t-1}``-measurable. The truncation to
    ``max_fraction / alpha`` keeps ``1 + kappa (L - alpha) > 0`` because
    ``L - alpha >= -alpha``.
    """

    alpha: float
    max_fraction: float = 0.5
    mu: float = 0.0
    var: float = 0.25
    count: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if self.mu == 0.0:
            self.mu = self.alpha

    @property
    def cap(self) -> float:
        return self.max_fraction / self.alpha

    def stake(self) -> float:
        if self.count == 0:
            return 0.0
        edge = self.mu - self.alpha
        if edge <= 0.0:
            return 0.0
        return min(self.cap, edge / max(self.var, 1e-4))

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mu
        self.mu += delta / self.count
        self.var += (delta * (x - self.mu) - self.var) / self.count


@dataclass(slots=True)
class EProcess:
    """One type's capital process, with a running maximum and an alarm flag."""

    alpha: float
    delta: float = 0.05
    max_fraction: float = 0.5
    wealth: float = 1.0
    peak: float = 1.0
    t: int = 0
    alarm_time: int | None = None
    log_wealth_trace: list[float] = field(default_factory=list)
    _bettor: ONSBettor | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie in (0, 1)")
        self._bettor = ONSBettor(self.alpha, self.max_fraction)

    @property
    def threshold(self) -> float:
        """``1 / delta`` -- Ville's uniform level."""
        return 1.0 / self.delta

    @property
    def e_value(self) -> float:
        """The running supremum, which is itself a valid e-value at any time."""
        return self.peak

    @property
    def alarmed(self) -> bool:
        return self.alarm_time is not None

    def update(self, loss: float) -> float:
        """Feed one document's realised leakage; return the current wealth."""
        assert self._bettor is not None
        x = min(max(float(loss), 0.0), 1.0)
        kappa = self._bettor.stake()
        self.wealth *= 1.0 + kappa * (x - self.alpha)
        self.wealth = max(self.wealth, 1e-300)
        self.t += 1
        self.peak = max(self.peak, self.wealth)
        self.log_wealth_trace.append(math.log(self.wealth))
        if self.alarm_time is None and self.wealth >= self.threshold:
            self.alarm_time = self.t
        self._bettor.update(x)
        return self.wealth

    def run(self, losses: Iterable[float]) -> "EProcess":
        for loss in losses:
            self.update(loss)
        return self


@dataclass(slots=True)
class EDetector:
    """A changepoint-anchored e-detector built as a *mixture*, not a maximum.

    Restarting an e-process at candidate changepoints is what buys a detection
    -delay guarantee: a single process started at ``t = 0`` is valid but slow
    to react to a late change, because early null evidence has already shrunk
    its wealth. Combining the family correctly matters, though. Taking the
    running **maximum** over ``K`` restarts and keeping the ``1/delta``
    threshold inflates the level by up to a factor ``K`` -- each process is
    individually valid, but the max of ``K`` supermartingales is not one.

    We therefore combine them as a mixture with a **fixed prior** over
    changepoints,

        E_t = sum_k w_k * E^(k)_t ,   sum_k w_k = 1 ,

    where a process that has not started yet contributes ``E^(k)_t = 1``. Then
    ``E_0 = 1``, the mixture is itself a non-negative supermartingale under the
    null, and Ville's inequality applies to it at the same ``1/delta``
    threshold -- so continuous watching remains free. ``prior="geometric"``
    concentrates mass on early changepoints and reacts faster to them;
    ``"uniform"`` spreads it evenly over the allowed restarts.
    """

    alpha: float
    delta: float = 0.05
    restart_every: int = 25
    max_processes: int = 64
    prior: str = "geometric"
    prior_decay: float = 0.85
    t: int = 0
    alarm_time: int | None = None
    processes: list[EProcess] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    trace: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_processes < 1:
            raise ValueError("max_processes must be >= 1")
        if self.prior == "uniform":
            raw = [1.0] * self.max_processes
        elif self.prior == "geometric":
            raw = [self.prior_decay**k for k in range(self.max_processes)]
        else:
            raise ValueError("prior must be 'uniform' or 'geometric'")
        total = sum(raw)
        self.weights = [w / total for w in raw]

    @property
    def threshold(self) -> float:
        return 1.0 / self.delta

    def _mixture(self) -> float:
        value = 0.0
        for k, w in enumerate(self.weights):
            if k < len(self.processes):
                value += w * self.processes[k].wealth
            else:
                value += w  # not started yet: contributes E = 1
        return value

    def update(self, loss: float) -> float:
        if self.t % self.restart_every == 0 and len(self.processes) < self.max_processes:
            self.processes.append(EProcess(self.alpha, self.delta))
        for proc in self.processes:
            proc.update(loss)
        self.t += 1
        value = self._mixture()
        self.trace.append(value)
        if self.alarm_time is None and value >= self.threshold:
            self.alarm_time = self.t
        return value

    def run(self, losses: Iterable[float]) -> "EDetector":
        for loss in losses:
            self.update(loss)
        return self

    @property
    def e_value(self) -> float:
        """The mixture value, a valid e-value at any stopping time."""
        return self._mixture() if self.processes else 1.0
