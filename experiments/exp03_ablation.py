"""Leave-one-out ablation: what each layer buys, and what it does not.

The point of the table is to separate two things a reader might conflate.
*Accuracy* comes overwhelmingly from Layer I -- candidate formation and graph
propagation -- while *guarantee quality* comes from Layers II-III and buys
almost no accuracy at all. A reader who evaluates the system on F1 alone would
conclude the DRO layer is worthless; a reader who deploys it across a template
change would discover otherwise. So every variant reports F1 *and* the
certified radius, and the radius column is where the DRO row earns its place.
"""
from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from _common import banner, engines_from_env, save

from aegis.data import load_corpus, split_calibration_test
from aegis.pipeline import AegisConfig, AegisPipeline

VARIANTS: dict[str, dict[str, Any]] = {
    "full": {},
    "no-selective-escalation": {"escalation_budget": 0.0},
    "no-graph-propagation": {"beta1": 0.0, "beta2": 0.0},
    "no-heterogeneous-ensemble": {"engines": "rule"},
    "no-chi2-dro-selection": {"select_by": "min_overmask"},
    "no-weighted-crc": {"weighted_crc": False},
    "bonferroni-instead-of-fixed-sequence": {"ltt_method": "bonferroni"},
}


def run_variant(name: str, overrides: dict[str, Any], base: dict[str, Any],
                cal, dep) -> dict[str, Any]:
    """Each variant is measured twice, because the two columns answer different
    questions: F1 at the *uncertified* F1-maximal point (so accuracy is not
    confounded with how conservative the certificate happened to be), and
    leakage / radius at the declared alpha (so guarantee quality is measured
    where the guarantee lives)."""
    acc_cfg = AegisConfig(**{**base, **overrides, "select_by": "max_f1"})
    acc_pipe = AegisPipeline(acc_cfg)
    _, acc_report = acc_pipe.run(list(cal), list(dep))
    micro = acc_report.span_metrics.get("micro", {}) if acc_report.span_metrics else {}

    cert_cfg = AegisConfig(**{**base, **overrides})
    cert_pipe = AegisPipeline(cert_cfg)
    cert, cert_report = cert_pipe.run(list(cal), list(dep))

    row = {
        "variant": name,
        "overrides": overrides,
        "F1": round(micro.get("F1", 0.0), 4),
        "P": round(micro.get("P", 0.0), 4),
        "R": round(micro.get("R", 0.0), 4),
        "miss_at_alpha": round(cert_report.realised_leakage or 0.0, 4),
        "overmask_at_alpha": round(cert_report.realised_overmask or 0.0, 4),
        "rho_star": round(cert.rho_star, 4),
        "no_certificate": cert.empty,
        "escalated_fraction": round(cert_report.escalated_fraction, 4),
        "candidate_miss_micro": round(acc_report.candidate_miss.get("micro", 0.0), 4),
        "effective_miss_micro": round(acc_report.effective_miss.get("micro", 0.0), 4),
        "candidate_miss_person": round(acc_report.candidate_miss.get("PERSON", 0.0), 4),
        "effective_miss_person": round(acc_report.effective_miss.get("PERSON", 0.0), 4),
        "n_candidates": acc_pipe.diagnostics.get("n_candidates"),
        "graph": acc_pipe.diagnostics.get("graph"),
    }
    print(f"[aegis] {name:38s} F1={row['F1']:.4f} miss={row['miss_at_alpha']:.4f} "
          f"rho*={row['rho_star']:.4f} candmiss={row['candidate_miss_micro']:.4f} "
          f"effmiss={row['effective_miss_micro']:.4f} "
          f"effmissPERSON={row['effective_miss_person']:.4f}"
          + ("  [no certificate]" if cert.empty else ""))
    return row


# ---------------------------------------------------------------------------
# Proposition 1(a) in practice: recall against entity multiplicity
# ---------------------------------------------------------------------------


def recall_vs_multiplicity(
    docs, engines: str, theta: float = 0.60, beta1: float = 0.95,
) -> dict[str, Any]:
    """Per-occurrence recall at threshold ``theta`` as multiplicity grows.

    This is the quantitative form of the paper's central mechanism. Without
    propagation each occurrence of a recurring entity is an independent coin
    flip, so recall is flat in multiplicity. With identity-channel propagation
    one accepted occurrence lifts its whole component, so recall climbs toward
    the ``1 - (1 - q)^m`` envelope -- falling short of it because real identity
    components are not always fully connected (a truncation or an initialism
    splits one).
    """
    from collections import defaultdict

    from aegis.layer1_candidates.propagation import normalise_value, recall_amplification

    out: dict[str, Any] = {"theta": theta, "curves": {}}
    for label, on in (("with-propagation", True), ("without-propagation", False)):
        cfg = AegisConfig(
            engines=engines,
            beta1=beta1 if on else 0.0,
            beta2=0.60 if on else 0.0,
            effective_floor=theta,
        )
        pipeline = AegisPipeline(cfg)
        pipeline.form_candidates(list(docs))

        # multiplicity of a gold entity = how many times its normalised value
        # occurs across the batch
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for doc in docs:
            for gold in doc.gold:
                counts[(gold.type, normalise_value(gold.text, gold.type))] += 1

        hit: dict[int, int] = defaultdict(int)
        tot: dict[int, int] = defaultdict(int)
        for doc in docs:
            for gold in doc.gold:
                m = min(counts[(gold.type, normalise_value(gold.text, gold.type))], 6)
                tot[m] += 1
                if any(
                    c.type == gold.type and gold.iou(c) >= 0.5 and c.score >= theta
                    for c in doc.candidates
                ):
                    hit[m] += 1
        out["curves"][label] = {
            str(m): {
                "recall": round(hit[m] / tot[m], 4),
                "n_occurrences": tot[m],
            }
            for m in sorted(tot)
        }
    q = out["curves"]["without-propagation"].get("1", {}).get("recall", 0.85)
    out["theory_envelope"] = {
        str(m): round(recall_amplification(q, m), 4) for m in range(1, 7)
    }
    out["q_estimate"] = q
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="synpii")
    ap.add_argument("--engines", default=engines_from_env())
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--grid-size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resplits", type=int, default=1)
    args = ap.parse_args()

    banner(f"ablation :: {args.corpus}")
    base = {
        "engines": args.engines,
        "alpha": args.alpha,
        "gamma": 0.10,
        "delta": 0.05,
        "grid_size": args.grid_size,
        "escalation_budget": 0.30,
    }
    docs, _ = load_corpus(args.corpus, seed=7)
    if args.limit:
        docs = docs[: args.limit]
    elif args.corpus != "synpii":
        docs = docs[:1200]

    resplits: list[dict[str, Any]] = []
    for split_seed in range(args.resplits):
        cal, dep = split_calibration_test(docs, cal_frac=0.4, seed=split_seed)
        rows = [run_variant(n, o, base, cal, dep) for n, o in VARIANTS.items()]
        full = next(r for r in rows if r["variant"] == "full")
        for row in rows:
            if row["F1"] is not None and full["F1"] is not None:
                row["delta_F1"] = round(row["F1"] - full["F1"], 4)
        resplits.append({"split_seed": split_seed, "rows": rows})

    banner("Prop. 1(a) :: recall vs entity multiplicity")
    curve = recall_vs_multiplicity(docs, args.engines)
    for label, series in curve["curves"].items():
        pretty = " ".join(f"m={m}:{d['recall']:.3f}" for m, d in series.items())
        print(f"[aegis] {label:22s} {pretty}")
    print("[aegis] theory envelope     " + " ".join(
        f"m={m}:{v:.3f}" for m, v in curve["theory_envelope"].items()))

    save(
        "exp03_ablation",
        {
            "experiment": "exp03_ablation",
            "corpus": args.corpus,
            "engines": args.engines,
            "base_config": base,
            "resplits": resplits,
            "recall_vs_multiplicity": curve,
        },
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
