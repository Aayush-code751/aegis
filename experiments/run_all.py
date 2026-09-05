"""Run every experiment in order and write a manifest.

    python experiments/run_all.py            # full reproduction
    python experiments/run_all.py --quick    # ~2 minutes, small subsamples
    python experiments/run_all.py --only exp02 exp03

``--quick`` shrinks corpora, grids and stream counts so the whole pipeline can
be exercised in a couple of minutes; it is what CI and the Docker smoke target
run. Numbers from a quick run are *not* the numbers to cite -- the manifest
records which mode produced them.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from _common import banner, save

HERE = Path(__file__).resolve().parent

FULL: dict[str, list[str]] = {
    "exp01": ["exp01_main_table.py"],
    "exp02": ["exp02_shift_radius.py"],
    "exp03": ["exp03_ablation.py"],
    "exp04": ["exp04_monitor.py"],
    "exp05": ["exp05_chao.py"],
    "exp06": ["exp06_escalation.py"],
}

QUICK: dict[str, list[str]] = {
    "exp01": ["exp01_main_table.py", "--corpora", "synpii", "--grid-size", "256"],
    "exp02": ["exp02_shift_radius.py", "--grid-size", "256", "--target-limit", "400"],
    "exp03": ["exp03_ablation.py", "--grid-size", "256"],
    "exp04": ["exp04_monitor.py", "--streams", "40", "--length", "2000"],
    "exp05": ["exp05_chao.py", "--corpora", "synpii", "--limit", "341"],
    "exp06": ["exp06_escalation.py", "--grid-size", "256"],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--keep-going", action="store_true",
                    help="do not stop at the first failing experiment")
    args = ap.parse_args()

    plan = QUICK if args.quick else FULL
    keys = args.only or list(plan)
    results: list[dict[str, object]] = []
    failures = 0

    for key in keys:
        if key not in plan:
            print(f"[aegis] unknown experiment {key!r}; known: {', '.join(plan)}")
            return 2
        cmd = [sys.executable, *plan[key]]
        banner(f"run_all :: {key}  ({' '.join(cmd[1:])})")
        started = time.time()
        proc = subprocess.run(cmd, cwd=HERE, check=False)
        elapsed = round(time.time() - started, 1)
        ok = proc.returncode == 0
        failures += 0 if ok else 1
        results.append({"experiment": key, "argv": plan[key],
                        "returncode": proc.returncode, "seconds": elapsed, "ok": ok})
        print(f"[aegis] {key} finished in {elapsed}s (ok={ok})")
        if not ok and not args.keep_going:
            break

    save("run_all_manifest", {
        "experiment": "run_all",
        "mode": "quick" if args.quick else "full",
        "results": results,
        "n_failures": failures,
    })
    print(f"\n[aegis] {len(results) - failures}/{len(results)} experiments succeeded")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
