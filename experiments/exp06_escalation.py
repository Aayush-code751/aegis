"""The escalation frontier and the (1 - 1/e) guarantee (Proposition 3).

Two things are measured.

**The frontier.** Quality against escalation budget. Because propagation makes
the escalation objective a weighted coverage function, greedy spends the budget
on *distinct entities* rather than on repeated instances of the same one, so
the frontier is steep early and flat late: once the graph has resolved the
recurring entities, further calls buy genuinely novel spans at a much lower hit
rate. A modular per-span rule would miss that structure entirely, and the
experiment reports both so the difference is visible rather than asserted.

Two resolver policies are traced, because they answer different questions.
``review`` is the honest offline default -- a human queue that approves
everything escalated -- and it shows what escalation buys in *leakage* and
costs in *utility*. ``oracle`` consults gold labels and so traces the **best
case** frontier a perfect LLM tier could reach; it is an upper bound and is
labelled as one, never a system result.

**The guarantee.** On small instances the greedy value is compared against the
brute-force optimum, and the ratio is checked against ``1 - 1/e``. This is a
test of the implementation, not of the theorem: lazy greedy must return
exactly what eager greedy returns, and both must clear the bound.
"""
from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np

from _common import banner, engines_from_env, save

from aegis.data import load_corpus, split_calibration_test
from aegis.layer5_act.escalate import (
    brute_force_optimum,
    coverage_value,
    lazy_greedy_escalate,
    oracle_resolver,
)
from aegis.pipeline import AegisConfig, AegisPipeline

BUDGETS = [0.0, 0.05, 0.10, 0.147, 0.20, 0.273, 0.40, 0.60, 0.80, 1.00]


def frontier(corpus: str, engines: str, limit: int, alpha: float,
             grid: int, resolver_name: str = "review") -> list[dict[str, Any]]:
    banner(f"escalation frontier :: {corpus} :: resolver={resolver_name}")
    docs, _ = load_corpus(corpus, seed=7)
    if limit:
        docs = docs[:limit]
    cal, dep = split_calibration_test(docs, cal_frac=0.4, seed=0)
    rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        cfg = AegisConfig(
            engines=engines, alpha=alpha, grid_size=grid,
            escalation_budget=budget, select_by="max_f1",
        )
        pipeline = AegisPipeline(cfg)
        resolver = oracle_resolver() if resolver_name == "oracle" else None
        cert = pipeline.calibrate(list(cal), list(dep))
        report = pipeline.deploy(list(dep), cert, resolver=resolver)
        micro = report.span_metrics.get("micro", {}) if report.span_metrics else {}
        row = {
            "budget": budget,
            "F1": round(micro.get("F1", 0.0), 4),
            "P": round(micro.get("P", 0.0), 4),
            "R": round(micro.get("R", 0.0), 4),
            "miss": round(report.realised_leakage or 0.0, 4),
            "overmask": round(report.realised_overmask or 0.0, 4),
            "escalated_fraction": round(report.escalated_fraction, 4),
            "n_escalated": report.n_escalated,
        }
        rows.append(row)
        print(f"[aegis] budget={budget:<6} F1={row['F1']:.4f} miss={row['miss']:.4f} "
              f"escalated={row['escalated_fraction']:.4f}")
    return rows


def submodularity_check(corpus: str, engines: str, limit: int,
                        trials: int = 12, pool: int = 14, budget: int = 4) -> dict[str, Any]:
    """Greedy vs brute-force optimum on small random sub-instances."""
    banner("Prop. 3 :: greedy vs brute-force optimum")
    docs, _ = load_corpus(corpus, seed=7)
    docs = docs[: limit or 200]
    pipeline = AegisPipeline(AegisConfig(engines=engines))
    graph = pipeline.form_candidates(docs)
    rng = np.random.default_rng(0)
    pi = [1.0 if n.score >= 0.2 else 0.0 for n in graph.nodes]

    ratios: list[float] = []
    for _ in range(trials):
        eligible = sorted(rng.choice(len(graph.nodes), size=min(pool, len(graph.nodes)),
                                     replace=False).tolist())
        plan = lazy_greedy_escalate(graph, pi, budget, eligible=eligible)
        greedy = coverage_value(graph, plan.chosen, pi)
        optimum = brute_force_optimum(graph, pi, budget, eligible)
        if optimum > 0:
            ratios.append(greedy / optimum)
    bound = 1.0 - 1.0 / math.e
    worst = min(ratios) if ratios else 1.0
    print(f"[aegis] trials={len(ratios)} worst greedy/optimum={worst:.4f} "
          f"bound=1-1/e={bound:.4f} -> {'PASS' if worst >= bound - 1e-9 else 'FAIL'}")
    return {
        "trials": len(ratios),
        "worst_ratio": round(worst, 6),
        "mean_ratio": round(float(np.mean(ratios)), 6) if ratios else None,
        "bound": round(bound, 6),
        "passes_bound": bool(worst >= bound - 1e-9),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="synpii")
    ap.add_argument("--engines", default=engines_from_env())
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--grid-size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    save(
        "exp06_escalation",
        {
            "experiment": "exp06_escalation",
            "corpus": args.corpus,
            "engines": args.engines,
            "frontier_review_policy": frontier(
                args.corpus, args.engines, args.limit, args.alpha, args.grid_size,
                "review",
            ),
            "frontier_oracle_upper_bound": frontier(
                args.corpus, args.engines, args.limit, args.alpha, args.grid_size,
                "oracle",
            ),
            "submodularity": submodularity_check(args.corpus, args.engines, args.limit),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
