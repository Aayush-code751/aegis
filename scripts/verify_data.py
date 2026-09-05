#!/usr/bin/env python3
"""Verify the bundled corpora before an experiment trusts them.

Checks, per file: that it exists, that every line parses, that gold spans fall
inside their document and match the substring they claim, and that the
document count and SHA-256 agree with the manifest recorded at release time.
A silent corpus change is the one bug that would invalidate every number in
the release without any test failing, so this runs first in ``make smoke``
and again at Docker build time.

    python scripts/verify_data.py            # verify
    python scripts/verify_data.py --write    # (re)write the manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__import__("os").environ.get("AEGIS_DATA_ROOT", ROOT / "data"))
MANIFEST = DATA / "MANIFEST.json"

EXPECTED = [
    "synpii/synpii_finance_seed7.jsonl",
    "synpii/synpii_finance_seed13.jsonl",
    "synpii/synpii_finance_seed21.jsonl",
    "synpii/synpii_finance_seed42.jsonl",
    "synpii/synpii_finance_seed77.jsonl",
    "external/gretel_finance.jsonl",
    "external/nemotron_finance.jsonl",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(path: Path) -> dict[str, object]:
    """Count documents, spans, and the two label defects present upstream.

    ``out_of_range`` spans point past the end of their document -- a genuine
    defect in the third-party corpora, and the reason the loader drops rather
    than clamps them. ``case_only`` spans have correct offsets but a recorded
    surface that differs from the document only in case, which is cosmetic and
    changes nothing. Both counts are pinned in the manifest, so an *unexpected*
    change fails the check while the known defects do not.
    """
    docs = 0
    spans = 0
    out_of_range = 0
    case_only = 0
    text_mismatch = 0
    types: dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path.name}:{line_no}: invalid JSON ({exc})") from exc
            if "text" not in row or "id" not in row:
                raise SystemExit(f"{path.name}:{line_no}: missing 'id' or 'text'")
            docs += 1
            text = row["text"]
            for span in row.get("gold") or []:
                spans += 1
                start, end = int(span["start"]), int(span["end"])
                if not (0 <= start < end <= len(text)):
                    out_of_range += 1
                    continue
                claimed = span.get("text")
                actual = text[start:end]
                if claimed is not None and actual != claimed:
                    if actual.casefold() == str(claimed).casefold():
                        case_only += 1
                    else:
                        text_mismatch += 1
                types[span.get("type", "?")] = types.get(span.get("type", "?"), 0) + 1
    return {
        "documents": docs,
        "gold_spans": spans,
        "out_of_range_spans": out_of_range,
        "case_only_mismatches": case_only,
        "text_mismatches": text_mismatch,
        "types": dict(sorted(types.items())),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="rewrite the manifest")
    args = ap.parse_args()

    observed: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for rel in EXPECTED:
        path = DATA / rel
        if not path.exists():
            missing.append(rel)
            continue
        info = inspect(path)
        observed[rel] = info
        notes = []
        if info["out_of_range_spans"]:
            notes.append(f"{info['out_of_range_spans']} out-of-range (dropped by the loader)")
        if info["case_only_mismatches"]:
            notes.append(f"{info['case_only_mismatches']} case-only label mismatches")
        if info["text_mismatches"]:
            notes.append(f"{info['text_mismatches']} REAL text mismatches")
        flag = ("  [" + "; ".join(notes) + "]") if notes else ""
        print(f"[data] {rel:42s} docs={info['documents']:<6} spans={info['gold_spans']:<7}"
              f" sha={str(info['sha256'])[:12]}{flag}")

    if missing:
        print("\n[data] MISSING:")
        for rel in missing:
            print(f"  - {rel}")
        print("\nThe corpora ship with the release. If you are working from a "
              "sparse checkout, fetch them with scripts/fetch_external_data.py "
              "or set AEGIS_DATA_ROOT to a directory that has them.")
        return 2

    # Only a *real* surface mismatch is a hard failure: it would mean the text
    # and the labels have drifted apart, which invalidates every span metric.
    broken = {k: v for k, v in observed.items() if v["text_mismatches"]}
    if broken:
        print("\n[data] FAILED: gold spans disagree with their document text")
        for rel, info in broken.items():
            print(f"  - {rel}: {info['text_mismatches']} spans")
        return 3

    if args.write:
        MANIFEST.write_text(
            json.dumps({"files": observed}, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"\n[data] wrote {MANIFEST.relative_to(ROOT)}")
        return 0

    if not MANIFEST.exists():
        print("\n[data] no manifest yet; run with --write to record one")
        return 0

    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    drift = []
    for rel, info in observed.items():
        ref = expected.get(rel)
        if ref is None:
            drift.append(f"{rel}: not in manifest")
        elif ref["sha256"] != info["sha256"]:
            drift.append(f"{rel}: sha256 changed")
        elif ref["documents"] != info["documents"]:
            drift.append(f"{rel}: document count {ref['documents']} -> {info['documents']}")
        else:
            for key in ("gold_spans", "out_of_range_spans", "case_only_mismatches"):
                if ref.get(key) != info.get(key):
                    drift.append(f"{rel}: {key} {ref.get(key)} -> {info.get(key)}")
    if drift:
        print("\n[data] FAILED: corpora differ from the release manifest")
        for item in drift:
            print(f"  - {item}")
        return 4

    print(f"\n[data] all {len(observed)} files match the release manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
