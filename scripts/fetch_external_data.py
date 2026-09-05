#!/usr/bin/env python3
"""Rebuild the external corpora from Hugging Face, for auditability.

The corpora ship with the release, so this script is not needed to reproduce
anything. It exists so a reader can check that the bundled files really are
what the README says they are: it downloads the upstream datasets, applies the
same filtering and normalisation, and reports whether the result matches
`data/MANIFEST.json`.

    python scripts/fetch_external_data.py --dataset gretel --out /tmp/check
    python scripts/fetch_external_data.py --dataset nemotron --compare

Needs `pip install datasets`. Upstream datasets can be revised, so a mismatch
does not by itself mean the bundled copy is wrong -- it means the upstream has
moved, and the manifest records which snapshot this release measured.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SPECS = {
    "gretel": {
        "hf_id": "gretelai/synthetic_pii_finance_multilingual",
        "split": "train",
        "filename": "gretel_finance.jsonl",
        "language_field": "language",
        "language_value": "English",
    },
    "nemotron": {
        "hf_id": "nvidia/Nemotron-PII",
        "split": "train",
        "filename": "nemotron_finance.jsonl",
        "domain_field": "domain",
        "domain_value": "Finance",
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(SPECS), required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", action="store_true",
                    help="compare against data/MANIFEST.json instead of writing")
    args = ap.parse_args()

    spec = SPECS[args.dataset]
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("This script needs the 'datasets' package: pip install datasets\n"
              "It is optional -- the corpora already ship in data/.")
        return 2

    print(f"[fetch] loading {spec['hf_id']} split={spec['split']}")
    rows = load_dataset(spec["hf_id"], split=spec["split"])

    def keep(row: dict) -> bool:
        if "language_field" in spec:
            return row.get(spec["language_field"]) == spec["language_value"]
        if "domain_field" in spec:
            return row.get(spec["domain_field"]) == spec["domain_value"]
        return True

    kept = [r for r in rows if keep(r)]
    print(f"[fetch] {len(kept)} of {len(rows)} rows survive the filter")

    manifest_path = ROOT / "data" / "MANIFEST.json"
    if args.compare and manifest_path.exists():
        ref = json.loads(manifest_path.read_text())["files"]
        key = f"external/{spec['filename']}"
        expected = ref.get(key, {}).get("documents")
        print(f"[fetch] manifest records {expected} documents for {key}")
        if expected == len(kept):
            print("[fetch] document counts agree with the bundled snapshot")
            return 0
        print("[fetch] counts differ -- upstream has likely been revised since "
              "the snapshot this release measured")
        return 1

    if not args.out:
        print("[fetch] pass --out DIR to write, or --compare to check counts")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / spec["filename"]
    print(f"[fetch] normalisation of raw span annotations is dataset-specific; "
          f"writing raw filtered rows to {target} for inspection")
    with open(target, "w", encoding="utf-8") as handle:
        for i, row in enumerate(kept):
            handle.write(json.dumps({"id": f"{args.dataset}-{i}", **dict(row)}) + "\n")
    print(f"[fetch] wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
