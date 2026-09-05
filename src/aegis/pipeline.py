"""The AEGIS pipeline: Algorithm 1 of the paper, end to end.

    calibrate(cal_docs, deploy_docs) -> Certificate | None
    deploy(docs, certificate)        -> DeploymentReport

Calibration order matters and mirrors the paper:

1. **Layer I** -- run the ensemble over the *pooled* calibration + deployment
   batch, build the entity graph from the MinHash index, propagate. Pooling is
   not an optimisation: Lemma 1's equivariance holds for the pooled multiset,
   not for an asymmetric train/apply split, so propagating separately would
   void the downstream certificate.
2. **Layer III (measure)** -- fit the domain discriminator on unlabelled
   deployment text and obtain ``w_hat`` and ``rho_hat``.
3. **Layer II** -- build the monotone path, evaluate both risks on
   calibration, and run Learn-then-Test. An empty rejection set is a valid
   answer and short-circuits to ``no certificate``.
4. **Layer III (certify)** -- among certified operating points, select the one
   maximising the finite-sample certified radius ``rho_star_hat`` subject to
   the escalation budget, and flag the certificate void when
   ``rho_hat > rho_star_hat``.
5. **Layer V / IV** -- deploy under the selected point: gate, escalate under
   budget, mask with keyed surrogates, append to the ledger, and feed realised
   outcomes to the e-processes.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .layer1_candidates import Ensemble, build_ensemble
from .layer1_candidates.minhash import MinHashIndex
from .layer1_candidates.propagation import (
    EntityGraph,
    PropagationConfig,
    build_entity_graph,
    propagate,
)
from .layer2_risk import (
    Lambda,
    MaskingFamily,
    escalation_cost,
    learn_then_test,
    leakage_risk,
    overmask_risk,
    risk_matrix,
)
from .layer3_shift import (
    fit_density_ratio,
    bootstrap_rho_ci,
    finite_sample_certified_radius,
    weighted_crc_weights,
)
from .layer4_monitor import EDetector, dark_matter_rate, ebh_reject
from .layer5_act import AuditLedger, Surrogate, SurrogateKey
from .layer5_act.escalate import lazy_greedy_escalate, resolve_escalated
from .metrics import candidate_miss_rate, evaluate_spans
from .taxonomy import FINANCE_TYPES
from .types import Certificate, Document, Span


@dataclass(slots=True)
class AegisConfig:
    """Every knob, with the paper's defaults."""

    engines: str = "rule,heuristic-ner,shape"
    types: tuple[str, ...] = FINANCE_TYPES
    alpha: float = 0.10                 # declared leakage level
    gamma: float = 0.10                 # declared over-masking level
    delta: float = 0.05                 # certificate confidence
    band_floor: float = 0.20            # escalation band lower edge
    effective_floor: float = 0.60       # score floor for the effective-miss diagnostic
    grid_size: int = 2048
    escalation_budget: float = 0.30     # max expected escalated fraction
    beta1: float = 0.95
    beta2: float = 0.60
    power_iterations: int = 3
    lsh_threshold: float = 0.60
    context_cosine: float = 0.35
    iou_threshold: float = 0.5
    surrogate_key: str = "aegis-release-key"
    ltt_method: str = "fixed-sequence"
    # "rho_star"        widest certified radius (the deployment-safe choice)
    # "min_escalation"   cheapest certified point
    # "min_overmask"     most useful certified point
    # "max_f1"           *uncertified*: maximise calibration span F1, bypassing
    #                    LTT entirely. This is the paper's AEGIS-U point and
    #                    exists only to make the head-to-head accuracy
    #                    comparison against uncontrolled baselines fair; it
    #                    carries no guarantee and reports none.
    select_by: str = "rho_star"
    weighted_crc: bool = True
    seed: int = 0

    def propagation(self) -> PropagationConfig:
        return PropagationConfig(
            beta1=self.beta1,
            beta2=self.beta2,
            power_iterations=self.power_iterations,
            lsh_threshold=self.lsh_threshold,
            context_cosine=self.context_cosine,
        )

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["types"] = list(self.types)
        return d


@dataclass(slots=True)
class DeploymentReport:
    """What a deployment run produced, and what the monitor thinks of it."""

    n_docs: int = 0
    realised_leakage: float | None = None
    realised_overmask: float | None = None
    escalated_fraction: float = 0.0
    n_escalated: int = 0
    span_metrics: dict[str, Any] = field(default_factory=dict)
    candidate_miss: dict[str, float] = field(default_factory=dict)
    effective_miss: dict[str, float] = field(default_factory=dict)
    dark_matter: dict[str, dict[str, float]] = field(default_factory=dict)
    alarms: list[str] = field(default_factory=list)
    alarm_times: dict[str, int | None] = field(default_factory=dict)
    certificate: dict[str, Any] = field(default_factory=dict)
    ledger: dict[str, Any] = field(default_factory=dict)
    anonymised: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class AegisPipeline:
    """The five layers wired together."""

    def __init__(self, config: AegisConfig | None = None, ensemble: Ensemble | None = None) -> None:
        self.config = config or AegisConfig()
        self.ensemble = ensemble or build_ensemble(self.config.engines, types=self.config.types)
        self.family: MaskingFamily | None = None
        self._pooled: list[Document] = []
        self.grid: list[Lambda] = []
        self.graph: EntityGraph | None = None
        self.certificate: Certificate | None = None
        self.surrogate = Surrogate(SurrogateKey.from_string(self.config.surrogate_key))
        self.ledger = AuditLedger()
        self.diagnostics: dict[str, Any] = {}

    # -- Layer I ------------------------------------------------------------

    def form_candidates(self, docs: Sequence[Document]) -> EntityGraph:
        """Detect, build the graph over the pooled batch, and propagate."""
        for doc in docs:
            doc.candidates = self.ensemble.detect(doc.text)
        index = MinHashIndex(threshold=self.config.lsh_threshold).build(
            [(str(i), d.text) for i, d in enumerate(docs)]
        )
        graph = build_entity_graph(docs, config=self.config.propagation(), index=index)
        for node, score in zip(graph.nodes, propagate(graph, config=self.config.propagation())):
            node.score = score
        self.graph = graph
        # the graph's doc_of indexes into the *pooled* batch, so anything that
        # maps a node back to its document must use the same list
        self._pooled = list(docs)
        self.diagnostics["graph"] = graph.stats()
        self.diagnostics["n_candidates"] = float(len(graph.nodes))
        return graph

    # -- Layers II + III ---------------------------------------------------

    def calibrate(
        self,
        calibration: Sequence[Document],
        deployment: Sequence[Document] | None = None,
    ) -> Certificate:
        """Run steps 1-4 and return a certificate (possibly an empty one)."""
        cfg = self.config
        pooled = list(calibration) + list(deployment or [])
        self.form_candidates(pooled)

        # Layer III (measure): unlabelled shift estimate
        rho_hat: float | None = None
        rho_ci: tuple[float, float] | None = None
        weights: np.ndarray | None = None
        if deployment:
            ratio = fit_density_ratio(calibration, deployment, seed=cfg.seed)
            rho_hat = ratio.rho_hat
            rho_ci = bootstrap_rho_ci(ratio, reps=1000, seed=cfg.seed)
            # Under (near) perfect separation the estimated ratio carries no
            # usable information -- correcting with it would be worse than not
            # correcting -- so we skip the reweighting and let the separability
            # verdict void the certificate instead.
            if cfg.weighted_crc and not ratio.separable:
                weights = ratio.weights
            self.diagnostics["density_ratio"] = ratio.as_dict()
            import math as _math

            self.diagnostics["rho_hat_ci"] = [
                "inf" if _math.isinf(v) else round(v, 6) for v in rho_ci
            ]

        # Layer II: monotone path + risks + LTT
        self.family = MaskingFamily(types=cfg.types, iou_thr=cfg.iou_threshold).fit(pooled)
        self.grid = self.family.path(cfg.grid_size)
        leak, over, esc = risk_matrix(
            list(calibration), self.family, self.grid, band_floor=cfg.band_floor
        )
        ltt = learn_then_test(
            self.grid, leak, over, alpha=cfg.alpha, gamma=cfg.gamma,
            delta=cfg.delta, method=cfg.ltt_method,
        )
        if cfg.select_by == "max_f1":
            # Uncertified operating point: pick the grid index maximising span
            # F1 on calibration. No certificate is claimed, and rho_star is
            # reported as 0 so no reader can mistake it for a guarantee.
            best_j, best_f1 = 0, -1.0
            for j in range(len(self.grid)):
                pairs = [
                    (doc.gold, self.family.mask(doc, self.grid[j]))
                    for doc in calibration
                ]
                f1 = evaluate_spans(pairs, cfg.iou_threshold).micro["F1"]
                if f1 > best_f1:
                    best_j, best_f1 = j, f1
            self._selected_index = best_j
            self.diagnostics["uncertified_f1"] = round(best_f1, 6)
            self.certificate = Certificate(
                lam=self.grid[best_j].as_dict(), alpha=cfg.alpha, gamma=cfg.gamma,
                delta=cfg.delta, rho_star=0.0, rho_hat=rho_hat,
                n_cal=len(calibration), empty=False,
                diagnostics=dict(self.diagnostics) | {
                    "selected_index": best_j, "uncertified": True,
                },
            )
            return self.certificate
        self.diagnostics["ltt"] = ltt.summary()
        self.diagnostics["grid_size"] = float(len(self.grid))
        self.diagnostics["mean_leak_path"] = [round(v, 6) for v in ltt.mean_leak]
        self.diagnostics["mean_over_path"] = [round(v, 6) for v in ltt.mean_over]

        if ltt.empty:
            self.certificate = Certificate(
                lam={}, alpha=cfg.alpha, gamma=cfg.gamma, delta=cfg.delta,
                rho_star=0.0, rho_hat=rho_hat, n_cal=len(calibration), empty=True,
                diagnostics=dict(self.diagnostics),
            )
            return self.certificate

        # Layer III (certify): pick the certified point with the widest radius
        L = np.asarray(leak, dtype=np.float64)
        E = np.asarray(esc, dtype=np.float64)
        best_j, best_rho, rows = None, -1.0, []
        for j in ltt.certified:
            losses = L[:, j]
            if weights is not None:
                p = weighted_crc_weights(weights)
                # importance-weighted losses keep the DRO radius on the target
                losses = losses * (p * len(losses) / max(p.sum(), 1e-12))
            rho_star = finite_sample_certified_radius(
                losses, cfg.alpha, cfg.delta, n_eta=64
            )
            mean_esc = float(E[:, j].mean()) if E.size else 0.0
            rows.append({"j": j, "t": self.grid[j].t, "rho_star": rho_star, "esc": mean_esc})
            if mean_esc > cfg.escalation_budget:
                continue
            key = {
                "rho_star": rho_star,
                "min_escalation": -mean_esc,
                "min_overmask": -ltt.mean_over[j],
            }[cfg.select_by]
            if key > best_rho:
                best_j, best_rho = j, key
        if best_j is None:  # budget infeasible: fall back to the widest radius
            widest = max(rows, key=lambda r: r["rho_star"])
            best_j = int(widest["j"])
        chosen = next(r for r in rows if r["j"] == best_j)
        self.diagnostics["certified_points"] = rows

        self.certificate = Certificate(
            lam=self.grid[best_j].as_dict(),
            alpha=cfg.alpha,
            gamma=cfg.gamma,
            delta=cfg.delta,
            rho_star=float(chosen["rho_star"]),
            rho_hat=rho_hat,
            n_cal=len(calibration),
            empty=False,
            diagnostics=dict(self.diagnostics) | {"selected_index": best_j},
        )
        self._selected_index = best_j
        return self.certificate

    # -- Layers V + IV -----------------------------------------------------

    def deploy(
        self,
        docs: Sequence[Document],
        certificate: Certificate | None = None,
        resolver: Callable[[Document, Span], bool] | None = None,
        emit_text: bool = False,
        score_baseline: bool = False,
    ) -> DeploymentReport:
        """Gate, escalate, mask, log, and monitor a deployment stream."""
        cfg = self.config
        cert = certificate or self.certificate
        if cert is None:
            raise RuntimeError("calibrate() before deploy()")
        report = DeploymentReport(n_docs=len(docs), certificate=cert.to_dict())

        if cert.empty:
            # No operating point certifies (alpha, gamma): route everything to
            # review. Reporting this is the whole point of Thm. 2's empty set.
            # Layer IV diagnostics are still computed -- they are the label-free
            # quantities an operator needs precisely when the gate has refused.
            report.realised_leakage = 0.0
            report.realised_overmask = 1.0
            report.escalated_fraction = 1.0
            report.n_escalated = sum(len(d.candidates) for d in docs)
            report.candidate_miss = candidate_miss_rate(docs, cfg.iou_threshold)
            report.effective_miss = candidate_miss_rate(
                docs, cfg.iou_threshold, min_score=cfg.effective_floor
            )
            all_candidates = [s for d in docs for s in d.candidates]
            report.dark_matter = {
                k: v.as_dict() for k, v in dark_matter_rate(
                    all_candidates, delta=cfg.delta, min_score=cfg.band_floor
                ).items()
            }
            for doc in docs:
                self.ledger.append(
                    text=doc.text, stage="route-to-review",
                    n_candidates=len(doc.candidates), verdict="no-certificate",
                    alpha=cfg.alpha, key_fingerprint=self.surrogate.key.fingerprint(),
                )
            report.ledger = self.ledger.summary()
            return report

        if self.family is None or not self.grid:
            raise RuntimeError("calibrate() must run before deploy()")
        lam = self.grid[getattr(self, "_selected_index", 0)]

        # ---- Layer V: escalate under budget, then mask -------------------
        graph = self.graph
        resolved: set[tuple[int, int, str]] = set()
        n_escalated = 0
        if graph is not None and cfg.escalation_budget > 0:
            node_pos = {id(node): i for i, node in enumerate(graph.nodes)}
            eligible: list[int] = []
            for i, doc in enumerate(docs):
                for span in doc.candidates:
                    thr = lam.thresholds.get(span.type, 1.0 + 1e-9)
                    if cfg.band_floor <= span.score < thr:
                        pos = node_pos.get(id(span))
                        if pos is not None:
                            eligible.append(pos)
            total_candidates = sum(len(d.candidates) for d in docs)
            budget = int(cfg.escalation_budget * max(total_candidates, 1))
            pi = [1.0 if n.score >= cfg.band_floor else 0.0 for n in graph.nodes]
            plan = lazy_greedy_escalate(graph, pi, budget, eligible=eligible)
            resolved = resolve_escalated(
                self._pooled or list(docs), graph, plan, resolver=resolver
            )
            n_escalated = len(plan.chosen)
            report.escalated_fraction = n_escalated / max(total_candidates, 1)
            self.diagnostics["escalation"] = plan.as_dict()
        report.n_escalated = n_escalated

        # ---- masking, metrics, ledger ------------------------------------
        detectors: dict[str, EDetector] = {
            t: EDetector(cfg.alpha, cfg.delta) for t in cfg.types
        }
        leak_sum = over_sum = 0.0
        eval_pairs: list[tuple[Sequence[Span], Sequence[Span]]] = []
        for doc in docs:
            masked = self.family.mask(doc, lam, resolved=resolved)
            leak = leakage_risk(doc, masked, iou_thr=cfg.iou_threshold)
            over = overmask_risk(doc, masked)
            leak_sum += leak
            over_sum += over
            eval_pairs.append((doc.gold, masked))
            if emit_text:
                report.anonymised.append(self.surrogate.apply(doc.text, masked))
            for typ, det in detectors.items():
                gold_y = [g for g in doc.gold if g.type == typ]
                if not gold_y:
                    continue
                missed = sum(
                    1 for g in gold_y
                    if not any(m.type == typ and g.iou(m) >= cfg.iou_threshold for m in masked)
                )
                det.update(missed / len(gold_y))
            self.ledger.append(
                text=doc.text, stage="mask",
                n_candidates=len(doc.candidates), n_masked=len(masked),
                n_escalated=n_escalated, leakage=leak, overmask=over,
                lam_t=lam.t, alpha=cfg.alpha, rho_star=cert.rho_star,
                rho_hat=cert.rho_hat, verdict=cert.verdict,
                key_fingerprint=self.surrogate.key.fingerprint(),
            )

        n = max(len(docs), 1)
        report.realised_leakage = leak_sum / n
        report.realised_overmask = over_sum / n
        report.span_metrics = evaluate_spans(eval_pairs, cfg.iou_threshold).as_dict()
        report.candidate_miss = candidate_miss_rate(docs, cfg.iou_threshold)
        report.effective_miss = candidate_miss_rate(
            docs, cfg.iou_threshold, min_score=cfg.effective_floor
        )
        all_candidates = [s for d in docs for s in d.candidates]
        report.dark_matter = {
            k: v.as_dict() for k, v in dark_matter_rate(
                all_candidates, delta=cfg.delta, min_score=cfg.band_floor
            ).items()
        }
        e_values = {t: det.e_value for t, det in detectors.items() if det.t > 0}
        report.alarms = ebh_reject(e_values, cfg.delta)
        report.alarm_times = {t: det.alarm_time for t, det in detectors.items() if det.t > 0}
        report.ledger = self.ledger.summary()
        return report

    # -- convenience -------------------------------------------------------

    def run(
        self,
        calibration: Sequence[Document],
        deployment: Sequence[Document],
        **deploy_kwargs: Any,
    ) -> tuple[Certificate, DeploymentReport]:
        cert = self.calibrate(calibration, deployment)
        return cert, self.deploy(deployment, cert, **deploy_kwargs)

    def save_diagnostics(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {"config": self.config.as_dict(), "diagnostics": self.diagnostics},
                handle, indent=2, sort_keys=True, default=float,
            )
