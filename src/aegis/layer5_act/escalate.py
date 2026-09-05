"""Submodular escalation with a ``(1 - 1/e)`` guarantee (Proposition 3).

Spans whose propagated score lands in the band ``[b, threshold_y)`` are
candidates for escalation to the LLM tier under a budget. Taking the largest
individual expected risk reductions would be a fractional knapsack -- exactly
solvable and uninteresting. Propagation changes the structure: resolving one
occurrence of a recurring entity resolves its whole identity component through
the damped-maximum channel, so the escalation value

    F(A) = sum over { z in union of C(v) for v in A } of pi_z

is a **weighted coverage function**, hence monotone and submodular. Greedy
selection under a cardinality budget therefore attains ``(1 - 1/e) F(A*)``,
and lazy evaluation with a priority queue makes it near-linear in the number
of banded spans.

The economically relevant consequence is that greedy spends the budget on
*distinct entities* rather than on repeated instances of the same one --
precisely where a modular per-span rule wastes it, and the reason the measured
cost frontier is steep early and flat late.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..layer1_candidates.propagation import EntityGraph
from ..layer2_risk.risks import Lambda
from ..types import Document, Span


@dataclass(slots=True)
class EscalationPlan:
    """Which spans to escalate, what they cover, and what it is worth."""

    chosen: list[int] = field(default_factory=list)
    covered: set[int] = field(default_factory=set)
    value: float = 0.0
    budget: int = 0
    n_candidates: int = 0
    evaluations: int = 0

    def as_dict(self) -> dict[str, float]:
        return {
            "n_chosen": float(len(self.chosen)),
            "n_covered": float(len(self.covered)),
            "value": round(self.value, 6),
            "budget": float(self.budget),
            "n_candidates": float(self.n_candidates),
            "evaluations": float(self.evaluations),
        }


def banded_spans(
    docs: Sequence[Document],
    lam: Lambda,
    band_floor: float = 0.20,
) -> list[tuple[int, int]]:
    """Indices ``(doc_index, span_index)`` of spans inside the escalation band."""
    if lam.full_redaction:
        return []
    out: list[tuple[int, int]] = []
    for i, doc in enumerate(docs):
        for j, span in enumerate(doc.candidates):
            thr = lam.thresholds.get(span.type, 1.0 + 1e-9)
            if band_floor <= span.score < thr:
                out.append((i, j))
    return out


def _component_index(graph: EntityGraph) -> dict[int, list[int]]:
    """Node index -> its identity component (list of node indices)."""
    out: dict[int, list[int]] = {}
    for indices in graph.components.values():
        for i in indices:
            out[i] = indices
    return out


def coverage_value(
    graph: EntityGraph,
    chosen: Sequence[int],
    pi: Sequence[float],
) -> float:
    """``F(A)`` -- the weighted coverage of the chosen nodes' components."""
    comp = _component_index(graph)
    covered: set[int] = set()
    for v in chosen:
        covered.update(comp.get(v, [v]))
    return float(sum(pi[z] for z in covered))


def lazy_greedy_escalate(
    graph: EntityGraph,
    pi: Sequence[float],
    budget: int,
    eligible: Sequence[int] | None = None,
) -> EscalationPlan:
    """Lazy-greedy maximisation of ``F`` under a cardinality budget.

    Minoux's lazy evaluation keeps an upper bound on each element's marginal
    gain in a max-heap and re-evaluates only when an element reaches the top;
    submodularity guarantees stale bounds are never optimistic in the wrong
    direction, so the result is bit-identical to eager greedy while touching
    far fewer elements. ``evaluations`` records how many marginal gains were
    actually computed.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    comp = _component_index(graph)
    pool = list(range(len(graph.nodes))) if eligible is None else list(eligible)
    plan = EscalationPlan(budget=budget, n_candidates=len(pool))
    if budget == 0 or not pool:
        return plan

    covered: set[int] = set()
    heap: list[tuple[float, int]] = []
    for v in pool:
        gain = sum(pi[z] for z in comp.get(v, [v]))
        heapq.heappush(heap, (-gain, v))
        plan.evaluations += 1

    while heap and len(plan.chosen) < budget:
        neg_gain, v = heapq.heappop(heap)
        fresh = sum(pi[z] for z in comp.get(v, [v]) if z not in covered)
        plan.evaluations += 1
        if not heap or fresh >= -heap[0][0] - 1e-12:
            if fresh <= 0.0:
                break
            plan.chosen.append(v)
            covered.update(comp.get(v, [v]))
            plan.value += fresh
        else:
            heapq.heappush(heap, (-fresh, v))
    plan.covered = covered
    return plan


def brute_force_optimum(
    graph: EntityGraph,
    pi: Sequence[float],
    budget: int,
    eligible: Sequence[int],
) -> float:
    """``F(A*)`` by exhaustive search -- for tests of the ``(1 - 1/e)`` bound only."""
    from itertools import combinations

    best = 0.0
    for size in range(min(budget, len(eligible)) + 1):
        for subset in combinations(eligible, size):
            best = max(best, coverage_value(graph, subset, pi))
    return best


def oracle_resolver(iou_thr: float = 0.5):
    """A *gold-label* resolver, for upper-bound experiments only.

    Returns a callable that answers "is this span really PII?" by consulting
    the document's gold spans. It stands in for a perfect LLM tier and so
    traces the **best case** escalation frontier -- the ceiling any real
    resolver approaches from below. It is never used in a certificate path and
    must never be reported as a system result; the honest offline default is
    the conservative review-queue policy in :func:`resolve_escalated`.
    """

    def _resolve(doc: Document, span: Span) -> bool:
        return any(
            g.type == span.type and span.iou(g) >= iou_thr for g in doc.gold
        )

    return _resolve


def resolve_escalated(
    docs: Sequence[Document],
    graph: EntityGraph,
    plan: EscalationPlan,
    resolver: Callable[[Document, Span], bool] | None = None,
) -> set[tuple[int, int, str]]:
    """Turn a plan into the set of span keys the mask should include.

    ``resolver`` is the LLM tier (or a human queue): given a document and a
    banded span it returns whether the span really is PII. When it is ``None``
    the conservative *review-queue* policy applies -- every escalated span and
    every clone in its identity component is masked -- which is what a review
    queue that approves everything does, and is the offline default so the
    pipeline runs with no network.
    """
    comp = _component_index(graph)
    doc_of = graph.doc_of
    keys: set[tuple[int, int, str]] = set()
    for v in plan.chosen:
        node = graph.nodes[v]
        keep = True
        if resolver is not None:
            keep = resolver(docs[doc_of[v]], node)
        if not keep:
            continue
        for z in comp.get(v, [v]):
            keys.add(graph.nodes[z].key())
    return keys
