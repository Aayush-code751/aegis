"""Shared plumbing for the experiment drivers."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
RESULTS = Path(os.environ.get("AEGIS_RESULTS", REPO_ROOT / "results"))

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def provenance() -> dict[str, Any]:
    """Everything needed to tell two runs apart."""
    import aegis

    return {
        "aegis_version": aegis.__version__,
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": {
            k: os.environ[k]
            for k in ("AEGIS_DATA_ROOT", "AEGIS_RESULTS", "AEGIS_ENGINES")
            if k in os.environ
        },
    }


def save(name: str, payload: dict[str, Any], subdir: str = "") -> Path:
    """Write a result artefact and return its path."""
    target = RESULTS / subdir if subdir else RESULTS
    target.mkdir(parents=True, exist_ok=True)
    path = target / (name if name.endswith(".json") else f"{name}.json")
    body = {"provenance": provenance(), **payload}
    path.write_text(json.dumps(body, indent=2, sort_keys=True, default=_default), encoding="utf-8")
    print(f"[aegis] wrote {path.relative_to(REPO_ROOT)}")
    return path


def _default(obj: Any) -> Any:
    import math

    if isinstance(obj, float) and math.isinf(obj):
        return "inf"
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def engines_from_env(default: str = "rule,heuristic-ner,shape") -> str:
    """Engine spec, overridable with ``AEGIS_ENGINES`` (e.g. add gliner/llm)."""
    return os.environ.get("AEGIS_ENGINES", default)


def banner(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 66 - len(title)))
