"""Table I: main results across the three corpora.

For each corpus we report the AEGIS operating points declared in the paper
(AEGIS-S at alpha=0.01, AEGIS-B at alpha=0.10, and the uncertified F1-maximal
AEGIS-U) together with the always-available baselines. Optional baselines
(Presidio, GLiNER, the LLM tier) are included automatically when their extras
are installed, and skipped -- loudly -- when they are not.

Every row carries realised leakage against the declared level, so the table
answers the question the paper insists on: not "how accurate is it" but "did
it respect the bound it claimed".

Usage
-----
    python experiments/exp01_main_table.py                 # all corpora
    python experiments/exp01_main_table.py --quick         # small subsample
    python experiments/exp01_main_table.py --corpora synpii
"""
from __future__ import annotations

import argparse
from typing import Any, Sequence

from _common import banner, engines_from_env, save

from aegis.data import load_corpus, split_calibration_test
from aegis.layer1_candidates import build_ensemble
from aegis.metrics import evaluate_spans
from aegis.pipeline import AegisConfig, AegisPipeline
from aegis.types import Document, Span

SEEDS = (7, 13, 21, 42, 77)


def _baseline_rows(docs: Sequence[Document], engines: str) -> list[dict[str, Any]]:
    """Score each engine on its own, as an uncontrolled baseline."""
    rows: list[dict[str, Any]] = []
    for name in [e.strip() for e in engines.split(",") if e.strip()]:
        try:
            ens = build_ensemble(name)
        except (ImportError, RuntimeError, ValueError) as exc:
            print(f"[aegis]   skipping baseline {name}: {exc}")
            continue
        pairs: list[tuple[Sequence[Span], Sequence[Span]]] = []
        leaked = total = 0
        for doc in docs:
            pred = ens.detect(doc.text)
            pairs.append((doc.gold, pred))
            for gold in doc.gold:
                total += 1
                if not any(p.type == gold.type and gold.iou(p) >= 0.5 for p in pred):
                    leaked += 1
        metrics = evaluate_spans(pairs)
        rows.append(
            {
                "system": name,
                "alpha": None,
                "P": round(metrics.micro["P"], 4),
                "R": round(metrics.micro["R"], 4),
                "F1": round(metrics.micro["F1"], 4),
                "miss": round(leaked / max(total, 1), 4),
                "certified": None,
            }
        )
    return rows


def _aegis_row(
    label: str,
    cal: Sequence[Document],
    dep: Sequence[Document],
    engines: str,
    alpha: float | None,
    grid: int,
    budget: float,
) -> dict[str, Any]:
    """One AEGIS operating point. ``alpha=None`` means the uncertified point."""
    declared = alpha if alpha is not None else 0.50
    cfg = AegisConfig(
        engines=engines, alpha=declared, gamma=0.10, delta=0.05,
        grid_size=grid, escalation_budget=budget,
        select_by="rho_star" if alpha is not None else "min_overmask",
    )
    pipeline = AegisPipeline(cfg)
    cert, report = pipeline.run(list(cal), list(dep))
    micro = report.span_metrics.get("micro", {}) if report.span_metrics else {}
    # With no certificate the stream is routed to review, so there is no
    # prediction to score -- reporting 0.0 would misread a refusal as a failure.
    none_if_empty = (lambda v: None) if cert.empty else (lambda v: round(v, 4))
    return {
        "system": label,
        "alpha": alpha,
        "P": none_if_empty(micro.get("P", 0.0)),
        "R": none_if_empty(micro.get("R", 0.0)),
        "F1": none_if_empty(micro.get("F1", 0.0)),
        "miss": round(report.realised_leakage or 0.0, 4),
        "overmask": round(report.realised_overmask or 0.0, 4),
        "utility": round(1.0 - (report.realised_overmask or 0.0), 4),
        "escalated_fraction": round(report.escalated_fraction, 4),
        "certificate": cert.to_dict(),
        "certified": None if alpha is None else (
            "held" if (report.realised_leakage or 0.0) <= alpha and not cert.empty
            else ("no-certificate" if cert.empty else "void")
        ),
        "candidate_miss_micro": round(report.candidate_miss.get("micro", 0.0), 4),
    }


def run_corpus(corpus: str, engines: str, quick: bool, grid: int) -> dict[str, Any]:
    banner(f"Table I :: {corpus}")
    limit = 200 if quick else 0
    seeds = (7,) if (quick or corpus != "synpii") else SEEDS
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        docs, stats = load_corpus(corpus, seed=seed)
        if limit:
            docs = docs[:limit]
        elif corpus != "synpii":
            docs = docs[:1500]  # keeps the pooled graph tractable; see README
        cal, dep = split_calibration_test(docs, cal_frac=0.4, seed=0)
        print(f"[aegis] seed={seed} n_cal={len(cal)} n_dep={len(dep)}")
        rows = _baseline_rows(dep, engines)
        rows.append(_aegis_row("AEGIS-S", cal, dep, engines, 0.01, grid, 0.30))
        rows.append(_aegis_row("AEGIS-B", cal, dep, engines, 0.10, grid, 0.30))
        rows.append(_aegis_row("AEGIS-U", cal, dep, engines, None, grid, 0.30))
        per_seed.append({"seed": seed, "n_cal": len(cal), "n_dep": len(dep),
                         "label_mapping": stats.as_dict(), "rows": rows})
    return {"corpus": corpus, "engines": engines, "per_seed": per_seed}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="*", default=["synpii", "gretel", "nemotron"])
    ap.add_argument("--engines", default=engines_from_env())
    ap.add_argument("--grid-size", type=int, default=1024)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    payload = {
        "experiment": "exp01_main_table",
        "quick": args.quick,
        "corpora": {
            c: run_corpus(c, args.engines, args.quick, args.grid_size)
            for c in args.corpora
        },
    }
    save("exp01_main_table", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
