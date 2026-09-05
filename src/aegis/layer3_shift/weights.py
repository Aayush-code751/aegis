"""Estimating the shift: a domain discriminator and its density ratio.

Given unlabelled deployment documents we fit a discriminator ``g`` separating
calibration from deployment and set

    w_hat(d)  proportional to  g(d) / (1 - g(d)),

normalised so that ``E_Phat_n[w_hat] = 1``. Two quantities come out:

* the weights themselves, which Layer III's weighted CRC uses to *correct* for
  the part of the shift the discriminator can see, and
* ``rho_hat = 1/2 * Ehat_P[(w_hat - 1)^2]``, a plug-in chi^2 divergence, which
  is the *measured* shift compared against the certified radius ``rho_star``.

Features are hashed character 4-grams plus a few document-shape statistics --
no embedding model and no network, so the whole audit runs offline and
deterministically. Where the discriminator separates nearly perfectly the
ratio is unstable in the tails, so it is clipped at a high percentile; the
clip biases the correction towards conservatism and is reported in the
diagnostics because it voids the exactness claim of Thm. 3.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..types import Document

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _hash_index(token: bytes, dim: int) -> int:
    return int.from_bytes(hashlib.blake2b(token, digest_size=4).digest(), "little") % dim


def featurise(text: str, dim: int = 512, ngram: int = 4) -> np.ndarray:
    """Hashed character n-gram features plus document-shape statistics."""
    vec = np.zeros(dim + 6, dtype=np.float64)
    lowered = text.lower()
    data = lowered.encode("utf-8", "ignore")
    if len(data) >= ngram:
        for i in range(0, len(data) - ngram + 1):
            vec[_hash_index(data[i : i + ngram], dim)] += 1.0
    norm = np.linalg.norm(vec[:dim])
    if norm > 0:
        vec[:dim] /= norm
    n_chars = max(len(text), 1)
    tokens = _TOKEN.findall(text)
    vec[dim + 0] = math.log1p(n_chars) / 10.0
    vec[dim + 1] = len(tokens) / n_chars
    vec[dim + 2] = sum(c.isdigit() for c in text) / n_chars
    vec[dim + 3] = sum(c.isupper() for c in text) / n_chars
    vec[dim + 4] = text.count("\n") / n_chars
    vec[dim + 5] = 1.0  # bias
    return vec


@dataclass(slots=True)
class DomainDiscriminator:
    """L2-regularised logistic regression trained with full-batch gradient descent.

    Deliberately simple and dependency-light: the estimand is a density ratio,
    not a state-of-the-art classifier, and a low-capacity model is *safer*
    here because an over-confident discriminator produces exactly the unstable
    tail ratios Sec. VI flags as a limitation.
    """

    dim: int = 512
    ngram: int = 4
    l2: float = 1e-3
    lr: float = 0.5
    epochs: int = 300
    seed: int = 0
    weights_: np.ndarray | None = field(default=None, repr=False)

    def fit(self, source: Sequence[str], target: Sequence[str]) -> "DomainDiscriminator":
        X = np.vstack(
            [featurise(t, self.dim, self.ngram) for t in list(source) + list(target)]
        )
        y = np.concatenate([np.zeros(len(source)), np.ones(len(target))])
        rng = np.random.default_rng(self.seed)
        w = rng.normal(scale=0.01, size=X.shape[1])
        n = X.shape[0]
        # class-balanced weights so an unequal split does not bias the ratio
        pos = max(y.sum(), 1.0)
        neg = max(n - pos, 1.0)
        sample_w = np.where(y > 0.5, n / (2 * pos), n / (2 * neg))
        for _ in range(self.epochs):
            z = X @ w
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            grad = X.T @ (sample_w * (p - y)) / n + self.l2 * w
            w -= self.lr * grad
        self.weights_ = w
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("fit the discriminator first")
        X = np.vstack([featurise(t, self.dim, self.ngram) for t in texts])
        z = np.clip(X @ self.weights_, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def auc(self, source: Sequence[str], target: Sequence[str]) -> float:
        """Rank AUC of source-vs-target separation (0.5 = no measurable shift)."""
        ps = self.predict_proba(list(source))
        pt = self.predict_proba(list(target))
        scores = np.concatenate([ps, pt])
        labels = np.concatenate([np.zeros(len(ps)), np.ones(len(pt))])
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1)
        n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
        if n_pos == 0 or n_neg == 0:
            return 0.5
        return float((ranks[labels > 0.5].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass(slots=True)
class DensityRatio:
    """The fitted ratio and everything a shift audit needs to report."""

    weights: np.ndarray          # w_hat on the calibration sample, mean 1
    rho_hat: float               # plug-in chi^2 divergence (may be math.inf)
    auc: float                   # held-out separation AUC
    clip_at: float
    clipped_fraction: float
    overlap: float               # score-support overlap in [0, 1]
    separable: bool              # True => effectively disjoint support
    discriminator: DomainDiscriminator

    def as_dict(self) -> dict[str, object]:
        return {
            "rho_hat": "inf" if math.isinf(self.rho_hat) else round(self.rho_hat, 6),
            "auc": round(self.auc, 6),
            "overlap": round(self.overlap, 6),
            "separable": self.separable,
            "clip_at": round(self.clip_at, 6),
            "clipped_fraction": round(self.clipped_fraction, 6),
            "w_min": round(float(self.weights.min()), 6),
            "w_max": round(float(self.weights.max()), 6),
            "w_mean": round(float(self.weights.mean()), 6),
        }


def chi2_divergence(weights: Sequence[float] | np.ndarray) -> float:
    """``rho_hat = 1/2 * E[(w - 1)^2]`` for mean-one weights."""
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return 0.0
    return float(0.5 * np.mean((w - 1.0) ** 2))


def _auc_from_scores(source_scores: np.ndarray, target_scores: np.ndarray) -> float:
    """Rank AUC from pre-computed scores (target treated as the positive class)."""
    scores = np.concatenate([source_scores, target_scores])
    labels = np.concatenate([np.zeros(source_scores.size), np.ones(target_scores.size)])
    if source_scores.size == 0 or target_scores.size == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    n_pos = labels.sum()
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[labels > 0.5].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _overlap(source_scores: np.ndarray, target_scores: np.ndarray) -> float:
    """A cheap support-overlap statistic in [0, 1].

    The fraction of source documents whose discriminator score exceeds the
    target's 5th percentile, times the fraction of target documents below the
    source's 95th percentile. It is 1 when the two score distributions sit on
    top of each other and 0 when they are cleanly separated.
    """
    if source_scores.size == 0 or target_scores.size == 0:
        return 0.0
    t_lo = float(np.percentile(target_scores, 5))
    s_hi = float(np.percentile(source_scores, 95))
    return float(np.mean(source_scores >= t_lo) * np.mean(target_scores <= s_hi))


def fit_density_ratio(
    calibration: Sequence[Document],
    deployment: Sequence[Document],
    clip_percentile: float = 99.0,
    dim: int = 512,
    seed: int = 0,
    epochs: int = 300,
    separability_auc: float = 0.99,
    separability_overlap: float = 0.02,
    folds: int = 2,
) -> DensityRatio:
    """Fit ``g`` by K-fold cross-fitting and measure ``rho_hat``.

    Three details matter for this to be an *honest* audit rather than a
    reassuring one.

    **Cross-fitting.** Each calibration document's weight is predicted by a
    discriminator that never saw it. In-sample ratios are optimistically small
    because the model has memorised the calibration set, which would
    understate the very shift the audit exists to surface. Because every fold
    contributes out-of-fold predictions, the returned ``weights`` array is
    aligned one-to-one with ``calibration`` and can be handed straight to
    weighted CRC.

    **A separability guard.** If the discriminator separates the two corpora
    (essentially) perfectly then their supports are effectively disjoint, the
    true ``dQ/dP`` is unbounded, and ``chi^2(Q || P) = infinity``: *no finite
    radius can cover the deployment*. The plug-in estimate does the opposite --
    with ``p -> 0`` on every source document the raw ratios collapse and, after
    mean-normalisation, ``rho_hat -> 0``, reporting "no shift" in precisely the
    case of total shift. We therefore return ``rho_hat = inf`` whenever the
    out-of-fold AUC or the score-support overlap crosses a threshold, so the
    comparison ``rho_hat <= rho_star`` fails as it must and the certificate is
    declared void rather than quietly held.

    **Tail clipping.** Where the discriminator separates nearly perfectly the
    ratio is unstable in the tails, so it is clipped at ``clip_percentile``.
    The clip biases the correction towards conservatism and is reported in
    ``clipped_fraction`` because it voids the exactness claim of Thm. 3.
    """
    src = [d.text for d in calibration]
    tgt = [d.text for d in deployment]
    if not src or not tgt:
        raise ValueError("need both calibration and deployment documents")

    folds = max(2, min(folds, len(src), len(tgt)))
    rng = np.random.default_rng(seed)
    src_fold = rng.integers(0, folds, size=len(src))
    tgt_fold = rng.integers(0, folds, size=len(tgt))

    oof_p_src = np.full(len(src), np.nan, dtype=np.float64)
    oof_p_tgt: list[float] = []
    last: DomainDiscriminator | None = None
    for k in range(folds):
        s_fit = [src[i] for i in range(len(src)) if src_fold[i] != k]
        t_fit = [tgt[i] for i in range(len(tgt)) if tgt_fold[i] != k]
        s_out = [i for i in range(len(src)) if src_fold[i] == k]
        t_out = [i for i in range(len(tgt)) if tgt_fold[i] == k]
        if not s_fit or not t_fit or not s_out:
            continue
        disc = DomainDiscriminator(dim=dim, seed=seed + k, epochs=epochs).fit(s_fit, t_fit)
        last = disc
        oof_p_src[s_out] = disc.predict_proba([src[i] for i in s_out])
        if t_out:
            oof_p_tgt.extend(float(v) for v in disc.predict_proba([tgt[i] for i in t_out]))

    if last is None:
        raise RuntimeError("cross-fitting produced no usable fold")
    missing = np.isnan(oof_p_src)
    if missing.any():
        oof_p_src[missing] = last.predict_proba([src[i] for i in np.flatnonzero(missing)])

    p_src = np.clip(oof_p_src, 1e-6, 1 - 1e-6)
    p_tgt = np.clip(np.asarray(oof_p_tgt or [0.5], dtype=np.float64), 1e-6, 1 - 1e-6)
    auc = _auc_from_scores(p_src, p_tgt)
    overlap = _overlap(p_src, p_tgt)

    raw = p_src / (1.0 - p_src)
    clip_at = float(np.percentile(raw, clip_percentile))
    clipped = raw > clip_at
    raw = np.minimum(raw, clip_at)
    mean = raw.mean()
    weights = raw / mean if mean > 0 else np.ones_like(raw)

    separable = bool(auc >= separability_auc or overlap <= separability_overlap)
    rho_hat = math.inf if separable else chi2_divergence(weights)
    return DensityRatio(
        weights=weights,
        rho_hat=rho_hat,
        auc=auc,
        clip_at=clip_at,
        clipped_fraction=float(clipped.mean()),
        overlap=overlap,
        separable=separable,
        discriminator=last,
    )


def bootstrap_rho_ci(
    ratio: DensityRatio,
    reps: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for ``rho_hat`` (document resampling)."""
    if ratio.separable:
        return (math.inf, math.inf)
    w = ratio.weights
    n = w.size
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=np.float64)
    for r in range(reps):
        idx = rng.integers(0, n, size=n)
        resampled = w[idx]
        mean = resampled.mean()
        draws[r] = chi2_divergence(resampled / mean if mean > 0 else resampled)
    lo = float(np.percentile(draws, 100 * (1 - level) / 2))
    hi = float(np.percentile(draws, 100 * (1 + level) / 2))
    return (lo, hi)
