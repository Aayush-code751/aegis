"""Figure 2 and the shift audit: rho_hat against rho_star.

This is the component the paper considers most consequential, so the driver is
deliberately blunt about it. For each deployment corpus we

1. calibrate on SynPII-F with **no target labels**,
2. compute the certified radius ``rho_star`` from the calibration losses,
3. measure the shift ``rho_hat`` the deployment corpus actually presents,
4. issue a verdict *before reading a single target label*, and only then
5. reveal the realised transfer leakage and check whether the verdict was right.

It also traces ``Rhat_rho`` over a range of ``rho`` for the figure, and sweeps
``alpha`` to locate each corpus's feasibility limit -- the smallest level at
which Learn-then-Test returns a non-empty set.
"""
from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np

from _common import banner, engines_from_env, save

from aegis.data import load_corpus, split_calibration_test
from aegis.layer2_risk import MaskingFamily, learn_then_test, risk_matrix
from aegis.layer3_shift import (
    bootstrap_rho_ci,
    certified_radius,
    chi2_dro_worst_case,
    finite_sample_certified_radius,
    fit_density_ratio,
)
from aegis.pipeline import AegisConfig, AegisPipeline

RHO_TRACE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.83, 0.9, 1.0, 1.1, 1.14, 1.2, 1.3, 1.4]
ALPHA_SWEEP = [0.005, 0.01, 0.02, 0.03, 0.05, 0.074, 0.10, 0.20, 0.50]


def _jsonable(x: float) -> Any:
    return "inf" if math.isinf(x) else round(x, 6)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engines", default=engines_from_env())
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--grid-size", type=int, default=1024)
    ap.add_argument("--source-seed", type=int, default=7)
    ap.add_argument("--target-limit", type=int, default=1200)
    args = ap.parse_args()

    banner("Layer III :: certified shift radius")
    source, _ = load_corpus("synpii", seed=args.source_seed)
    cal, held = split_calibration_test(source, cal_frac=0.4, seed=0)

    cfg = AegisConfig(engines=args.engines, alpha=args.alpha, grid_size=args.grid_size)
    pipeline = AegisPipeline(cfg)

    # ---- calibration-side losses at the selected operating point ---------
    pipeline.form_candidates(list(cal) + list(held))
    family = MaskingFamily(types=cfg.types).fit(list(cal) + list(held))
    grid = family.path(cfg.grid_size)
    leak, over, esc = risk_matrix(list(cal), family, grid, band_floor=cfg.band_floor)
    ltt = learn_then_test(grid, leak, over, alpha=cfg.alpha, gamma=cfg.gamma, delta=cfg.delta)
    if ltt.empty:
        print("[aegis] no certified operating point at this alpha; using the "
              "least-leakage point for the radius trace")
        j = len(grid) - 2
    else:
        j = ltt.certified[0]
    losses = np.asarray([row[j] for row in leak], dtype=np.float64)

    rho_star = certified_radius(losses, cfg.alpha)
    rho_star_hat = finite_sample_certified_radius(losses, cfg.alpha, cfg.delta, n_eta=96)
    trace = [
        {"rho": r, "R_rho": round(chi2_dro_worst_case(losses, r), 6)} for r in RHO_TRACE
    ]
    print(f"[aegis] rho_star={rho_star:.4f}  rho_star_hat={rho_star_hat:.4f} "
          f"(alpha={cfg.alpha}, n_cal={len(cal)})")

    # ---- per-corpus audit: verdict before labels, then the reveal --------
    audits: list[dict[str, Any]] = []
    for name, target in [
        ("synpii-heldout", held),
        ("gretel", load_corpus("gretel")[0][: args.target_limit]),
        ("nemotron", load_corpus("nemotron")[0][: args.target_limit]),
    ]:
        banner(f"audit :: {name}")
        ratio = fit_density_ratio(cal, target, seed=0)
        lo, hi = bootstrap_rho_ci(ratio, reps=800, seed=0)
        verdict = "valid" if ratio.rho_hat <= rho_star_hat else "void"
        print(f"[aegis]   rho_hat={_jsonable(ratio.rho_hat)} vs rho_star_hat="
              f"{rho_star_hat:.4f} -> {verdict.upper()} (no target labels used)")

        # only now do we look at the labels
        transfer = AegisPipeline(cfg)
        cert, report = transfer.run(list(cal), list(target))
        realised = report.realised_leakage or 0.0
        correct = (verdict == "valid") == (realised <= cfg.alpha)
        print(f"[aegis]   realised transfer leakage={realised:.4f} -> verdict "
              f"{'CORRECT' if correct else 'WRONG'}")
        audits.append(
            {
                "corpus": name,
                "n_deployment": len(target),
                "rho_hat": _jsonable(ratio.rho_hat),
                "rho_hat_ci95": [_jsonable(lo), _jsonable(hi)],
                "rho_star": round(rho_star, 6),
                "rho_star_hat": round(rho_star_hat, 6),
                "verdict_before_labels": verdict,
                "realised_leakage": round(realised, 6),
                "verdict_correct": bool(correct),
                "discriminator": ratio.as_dict(),
                "certificate": cert.to_dict(),
            }
        )

    # ---- alpha feasibility frontier --------------------------------------
    banner("alpha feasibility frontier")
    frontier: list[dict[str, Any]] = []
    for alpha in ALPHA_SWEEP:
        res = learn_then_test(grid, leak, over, alpha=alpha, gamma=cfg.gamma, delta=cfg.delta)
        entry = {"alpha": alpha, "feasible": not res.empty}
        if not res.empty:
            k = res.certified[0]
            entry["realised_calibration_leakage"] = round(res.mean_leak[k], 6)
            entry["t"] = round(grid[k].t, 6)
        frontier.append(entry)
        print(f"[aegis]   alpha={alpha:<6} feasible={not res.empty}")
    feasible = [e["alpha"] for e in frontier if e["feasible"]]

    save(
        "exp02_shift_radius",
        {
            "experiment": "exp02_shift_radius",
            "engines": args.engines,
            "alpha": cfg.alpha,
            "n_calibration": len(cal),
            "rho_star": round(rho_star, 6),
            "rho_star_hat": round(rho_star_hat, 6),
            "worst_case_trace": trace,
            "audits": audits,
            "alpha_frontier": frontier,
            "alpha_min_feasible": min(feasible) if feasible else None,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
