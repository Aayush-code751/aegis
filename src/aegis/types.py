"""Core data types shared by every layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Span:
    """A typed character interval, optionally scored by an engine.

    ``start``/``end`` are Python slice bounds into the owning document text.
    ``score`` is the engine confidence in [0, 1]; ``engine`` records which
    member of the ensemble proposed it (used as a capture occasion by the
    Chao estimator of Layer IV).
    """

    start: int
    end: int
    type: str
    text: str = ""
    score: float = 0.0
    engine: str = ""

    def iou(self, other: "Span") -> float:
        """Intersection-over-union of the two character intervals."""
        inter = max(0, min(self.end, other.end) - max(self.start, other.start))
        if inter == 0:
            return 0.0
        union = (self.end - self.start) + (other.end - other.start) - inter
        return inter / union if union else 0.0

    def matches(self, other: "Span", iou_thr: float = 0.5) -> bool:
        """Paper's match predicate: same type and IoU >= iou_thr."""
        return self.type == other.type and self.iou(other) >= iou_thr

    def key(self) -> tuple[int, int, str]:
        return (self.start, self.end, self.type)


@dataclass(slots=True)
class Document:
    """One stream item, with gold spans when available."""

    doc_id: str
    text: str
    gold: list[Span] = field(default_factory=list)
    candidates: list[Span] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def n_chars(self) -> int:
        return len(self.text)


@dataclass(slots=True)
class Certificate:
    """What Layer III emits, or ``None`` when nothing is certifiable.

    ``lam`` is the selected nesting parameter (per-type acceptance offsets plus
    the escalation budget coordinate); ``alpha``/``gamma`` are the declared
    leakage and over-masking levels; ``rho_star`` is the certified chi^2 shift
    radius (Thm. 4) and ``rho_hat`` the measured shift, so a deployment is
    in-scope exactly when ``rho_hat <= rho_star``.
    """

    lam: dict[str, float]
    alpha: float
    gamma: float
    delta: float
    rho_star: float
    rho_hat: float | None = None
    n_cal: int = 0
    empty: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def in_scope(self) -> bool | None:
        """True when the measured shift is inside the certified radius."""
        if self.rho_hat is None:
            return None
        return self.rho_hat <= self.rho_star

    @property
    def verdict(self) -> str:
        if self.empty:
            return "no-certificate"
        if self.rho_hat is None:
            return "valid-at-P"
        return "valid" if self.in_scope else "void"

    def to_dict(self) -> dict[str, Any]:
        import math

        rho_hat: Any = self.rho_hat
        if rho_hat is not None and math.isinf(rho_hat):
            rho_hat = "inf"
        return {
            "lam": self.lam,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "delta": self.delta,
            "rho_star": self.rho_star,
            "rho_hat": rho_hat,
            "n_cal": self.n_cal,
            "empty": self.empty,
            "verdict": self.verdict,
            "diagnostics": self.diagnostics,
        }
