"""Validating the label-free dark-matter bound (Proposition 2).

The estimator is meant for production, where there are no gold labels. So it
has to be validated *where labels do exist* before it is trusted where they do
not. For each corpus we compare, per type:

* the true candidate-miss count (from gold),
* the Chao1 point estimate, which should sit systematically *below* the truth
  because Chao's bound is a lower bound on richness and inter-engine
  dependence deflates ``f_1`` relative to ``f_2``, and
* the ``1 - delta`` log-normal upper confidence limit, which is the quantity
  actually reported in production and which should *cover* the truth.

The pass/fail criterion is coverage of the upper limit, not accuracy of the
point estimate. A point estimate below the truth is the estimator behaving as
advertised; an upper limit below the truth would be a real failure.
"""
from __future__ import annotations

import argparse
from typing import Any

from _common import banner, engines_from_env, save

from aegis.data import load_corpus
from aegis.layer4_monitor import dark_matter_rate
from aegis.pipeline import AegisConfig, AegisPipeline


def audit_corpus(corpus: str, engines: str, limit: int, delta: float,
                 min_score: float) -> dict[str, Any]:
    banner(f"Prop. 2 :: {corpus}")
    docs, _ = load_corpus(corpus, seed=7)
    if limit:
        docs = docs[:limit]
    pipeline = AegisPipeline(AegisConfig(engines=engines, band_floor=min_score))
    pipeline.form_candidates(docs)

    # ground truth: how many gold spans of each type no engine proposed
    true_missed: dict[str, int] = {}
    true_total: dict[str, int] = {}
    for doc in docs:
        for gold in doc.gold:
            true_total[gold.type] = true_total.get(gold.type, 0) + 1
            if not any(c.type == gold.type and gold.iou(c) >= 0.5 for c in doc.candidates):
                true_missed[gold.type] = true_missed.get(gold.type, 0) + 1

    estimates = dark_matter_rate(
        [s for d in docs for s in d.candidates], delta=delta, min_score=min_score
    )

    rows: list[dict[str, Any]] = []
    covered = comparable = 0
    inf_covered = inf_comparable = 0
    for typ in sorted(true_total):
        est = estimates.get(typ)
        if est is None:
            continue
        truth = true_missed.get(typ, 0)
        # the estimator speaks about the observed population it can see, so we
        # compare unseen *counts*: N_true - N_obs against N_hat - N_obs
        est_point = est.n_hat - est.n_obs
        est_upper = est.n_upper - est.n_obs
        ok = truth <= est_upper + 1e-9
        comparable += 1
        covered += int(ok)
        if est.informative:
            inf_comparable += 1
            inf_covered += int(ok)
        rows.append({
            "informative": est.informative,
            "note": est.note,
            "type": typ,
            "gold_total": true_total[typ],
            "true_unproposed": truth,
            "n_obs": est.n_obs,
            "f1": est.f1,
            "f2": est.f2,
            "chao_point_unseen": round(est_point, 3),
            "chao_upper_unseen": round(est_upper, 3),
            "covered_by_upper_limit": ok,
            "missing_upper_rate": round(est.missing_upper, 5),
        })
        tag = "" if est.informative else f"  [uninformative: {est.note}]"
        print(f"[aegis] {typ:16s} true={truth:5d}  point={est_point:8.2f}  "
              f"upper={est_upper:8.2f}  covered={'yes' if ok else 'NO'}{tag}")
    print(f"[aegis] coverage: {covered}/{comparable} types overall, "
          f"{inf_covered}/{inf_comparable} on informative types")
    below = [r for r in rows if r["chao_point_unseen"] <= r["true_unproposed"]]
    return {
        "corpus": corpus,
        "n_documents": len(docs),
        "delta": delta,
        "min_score": min_score,
        "rows": rows,
        "coverage": f"{covered}/{comparable}",
        "coverage_fraction": round(covered / max(comparable, 1), 4),
        "coverage_informative": f"{inf_covered}/{inf_comparable}",
        "coverage_informative_fraction": round(inf_covered / max(inf_comparable, 1), 4),
        "point_estimate_below_truth": f"{len(below)}/{comparable}",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="*", default=["synpii", "gretel", "nemotron"])
    ap.add_argument("--engines", default=engines_from_env())
    ap.add_argument("--limit", type=int, default=1200)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--min-score", type=float, default=0.20)
    args = ap.parse_args()
    save(
        "exp05_chao",
        {
            "experiment": "exp05_chao",
            "engines": args.engines,
            "corpora": {
                c: audit_corpus(c, args.engines, args.limit, args.delta, args.min_score)
                for c in args.corpora
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
