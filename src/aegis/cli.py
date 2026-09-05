"""Command-line interface: ``aegis <command> [options]``.

Commands
--------
``calibrate``  fit a certificate on one corpus and print/save it
``deploy``     apply a certificate to a corpus and report realised risk
``run``        calibrate then deploy in one shot (the common case)
``audit``      shift audit only: rho_hat vs rho_star, no labels needed
``redact``     redact a text file or stdin with a fitted certificate
``info``       corpus statistics and label-mapping counts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .data import load_corpus, split_calibration_test
from .layer3_shift import bootstrap_rho_ci, fit_density_ratio
from .pipeline import AegisConfig, AegisPipeline
from .taxonomy import FINANCE_TYPES


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", default="synpii",
                        choices=["synpii", "gretel", "nemotron"],
                        help="calibration corpus")
    parser.add_argument("--target-corpus", default="",
                        choices=["", "synpii", "gretel", "nemotron"],
                        help="deployment corpus for the zero-label transfer setting; "
                             "when set, calibration comes from --corpus and deployment "
                             "from this corpus, which is the regime rho_hat vs rho_star "
                             "is designed to gate")
    parser.add_argument("--target-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7, help="SynPII generation seed")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0,
                        help="cap the corpus size (0 = no cap); useful for smoke runs")
    parser.add_argument("--n-cal", type=int, default=0,
                        help="explicit calibration size (0 = use --cal-frac)")
    parser.add_argument("--cal-frac", type=float, default=0.4)
    parser.add_argument("--engines", default="rule,heuristic-ner,shape")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--grid-size", type=int, default=2048)
    parser.add_argument("--band-floor", type=float, default=0.20)
    parser.add_argument("--escalation-budget", type=float, default=0.30)
    parser.add_argument("--beta1", type=float, default=0.95)
    parser.add_argument("--beta2", type=float, default=0.60)
    parser.add_argument("--select-by", default="rho_star",
                        choices=["rho_star", "min_escalation", "min_overmask"])
    parser.add_argument("--no-weighted-crc", action="store_true")
    parser.add_argument("--out", default="", help="write the JSON report here")


def _config(args: argparse.Namespace) -> AegisConfig:
    return AegisConfig(
        engines=args.engines,
        types=FINANCE_TYPES,
        alpha=args.alpha,
        gamma=args.gamma,
        delta=args.delta,
        band_floor=args.band_floor,
        grid_size=args.grid_size,
        escalation_budget=args.escalation_budget,
        beta1=args.beta1,
        beta2=args.beta2,
        select_by=args.select_by,
        weighted_crc=not args.no_weighted_crc,
        seed=args.split_seed,
    )


def _load(args: argparse.Namespace):
    """Return ``(all_docs, calibration, deployment, label_stats)``.

    Without ``--target-corpus`` this is the in-domain split of one corpus. With
    it, calibration is the ``--corpus`` calibration slice and deployment is the
    *other* corpus in full -- the zero-label transfer setting.
    """
    docs, stats = load_corpus(args.corpus, seed=args.seed)
    if args.limit:
        docs = docs[: args.limit]
    cal, test = split_calibration_test(
        docs,
        n_cal=args.n_cal or None,
        cal_frac=args.cal_frac,
        seed=args.split_seed,
    )
    target = getattr(args, "target_corpus", "")
    if target and target != args.corpus:
        tgt_docs, stats = load_corpus(target, seed=args.seed)
        limit = getattr(args, "target_limit", 0) or args.limit
        if limit:
            tgt_docs = tgt_docs[:limit]
        return docs, cal, tgt_docs, stats
    return docs, cal, test, stats


def _emit(payload: dict[str, Any], out: str) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, default=float)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)


def cmd_info(args: argparse.Namespace) -> int:
    docs, cal, test, stats = _load(args)
    n_gold = sum(len(d.gold) for d in docs)
    per_type: dict[str, int] = {}
    for doc in docs:
        for span in doc.gold:
            per_type[span.type] = per_type.get(span.type, 0) + 1
    _emit(
        {
            "corpus": args.corpus,
            "n_documents": len(docs),
            "n_calibration": len(cal),
            "n_test": len(test),
            "n_gold_spans": n_gold,
            "gold_per_type": dict(sorted(per_type.items())),
            "label_mapping": stats.as_dict(),
            "mean_chars": round(sum(d.n_chars() for d in docs) / max(len(docs), 1), 1),
        },
        args.out,
    )
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Shift audit with no labels: measure rho_hat and report the AUC."""
    _, cal, test, _ = _load(args)
    ratio = fit_density_ratio(cal, test, seed=args.split_seed)
    lo, hi = bootstrap_rho_ci(ratio, reps=1000, seed=args.split_seed)
    _emit(
        {
            "corpus": args.corpus,
            "target_corpus": getattr(args, "target_corpus", "") or args.corpus,
            "n_calibration": len(cal),
            "n_deployment": len(test),
            "rho_hat": round(ratio.rho_hat, 6),
            "rho_hat_ci95": [round(lo, 6), round(hi, 6)],
            "discriminator": ratio.as_dict(),
        },
        args.out,
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    _, cal, test, _ = _load(args)
    pipeline = AegisPipeline(_config(args))
    cert, report = pipeline.run(cal, test)
    _emit(
        {
            "corpus": args.corpus,
            "config": pipeline.config.as_dict(),
            "certificate": cert.to_dict(),
            "report": {k: v for k, v in report.as_dict().items() if k != "anonymised"},
        },
        args.out,
    )
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    _, cal, test, _ = _load(args)
    pipeline = AegisPipeline(_config(args))
    cert = pipeline.calibrate(cal, test)
    _emit({"corpus": args.corpus, "certificate": cert.to_dict()}, args.out)
    return 0


def cmd_deploy(args: argparse.Namespace) -> int:
    _, cal, test, _ = _load(args)
    pipeline = AegisPipeline(_config(args))
    cert = pipeline.calibrate(cal, test)
    report = pipeline.deploy(test, cert)
    _emit(
        {
            "corpus": args.corpus,
            "certificate": cert.to_dict(),
            "report": {k: v for k, v in report.as_dict().items() if k != "anonymised"},
        },
        args.out,
    )
    return 0


def cmd_redact(args: argparse.Namespace) -> int:
    """Redact a file (or stdin) using a certificate fitted on ``--corpus``."""
    from .types import Document

    text = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
    _, cal, test, _ = _load(args)
    pipeline = AegisPipeline(_config(args))
    cert = pipeline.calibrate(cal, test)
    target = Document(doc_id="stdin", text=text)
    report = pipeline.deploy([target], cert, emit_text=True)
    if report.anonymised:
        sys.stdout.write(report.anonymised[0])
        if not report.anonymised[0].endswith("\n"):
            sys.stdout.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aegis",
        description="AEGIS: distribution-shift-certified risk control for PII redaction",
    )
    parser.add_argument("--version", action="version", version=f"aegis {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn, doc in [
        ("info", cmd_info, "corpus statistics"),
        ("audit", cmd_audit, "shift audit only (no labels required)"),
        ("calibrate", cmd_calibrate, "fit a certificate"),
        ("deploy", cmd_deploy, "apply a certificate and report realised risk"),
        ("run", cmd_run, "calibrate then deploy"),
        ("redact", cmd_redact, "redact a file or stdin"),
    ]:
        p = sub.add_parser(name, help=doc)
        _add_common(p)
        if name == "redact":
            p.add_argument("--file", default="", help="input file (default: stdin)")
        p.set_defaults(func=fn)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
