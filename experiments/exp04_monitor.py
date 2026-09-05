"""Monitor operating characteristics: validity first, then a fair delay race.

The experiment has two halves, because two different claims are at stake.

**Part A -- validity.** Each monitor runs at its own *nominal* threshold on
null streams. The e-process and e-detector must respect ``delta`` (Ville's
inequality guarantees it); the repeated fixed-window test must not, and the
size of the violation is the point: watched continuously it raises spurious
alarms at several times its nominal level, so an operator who recalibrates on
alarms spends a large share of their calibration budget chasing noise.

**Part B -- a fair race.** Comparing delays at different false-alarm rates
would be meaningless, so every baseline's threshold is first *calibrated on
null streams* to hit the same empirical false-alarm rate, and only then are
detection delays measured. The e-detector's advantage under matched
false-alarm rates is the honest version of the claim.

Streams are simulated because the quantity of interest is the monitor's
operating characteristic, which needs many independent null and shifted
streams; pre- and post-change leakage rates are taken from the measured
corpora.

**How to read Part B, honestly.** Once its threshold has been tuned on null
streams drawn from the true null, a CUSUM can be *competitive with or faster
than* the e-detector on delay -- and in this implementation it is. That does
not undercut the argument, it sharpens it: the tuning requires knowing the
null leakage distribution and re-tuning whenever ``alpha`` or the loss
distribution moves, which is exactly what a deployment does not have. The
e-process is valid at every stopping time with no tuning, no null-distribution
knowledge and no multiplicity penalty. Part A is the claim that matters; Part B
is reported because leaving it out would have been the dishonest choice.
"""
from __future__ import annotations

import argparse
import math
from typing import Any, Callable

import numpy as np

from _common import banner, save

from aegis.layer4_monitor import EDetector, EProcess


def cusum_alarm(stream: np.ndarray, alpha: float, threshold: float, drift: float = 0.02) -> int | None:
    """One-sided Gaussian CUSUM -- deliberately misspecified for a bounded mean."""
    s = 0.0
    for t, x in enumerate(stream, start=1):
        s = max(0.0, s + (x - alpha - drift))
        if s > threshold:
            return t
    return None


def fixed_window_alarm(stream: np.ndarray, alpha: float, window: int, z: float) -> int | None:
    """Fixed-window one-sided test applied at every step: valid once, not repeated."""
    se = math.sqrt(max(alpha * (1 - alpha), 1e-12) / window)
    for t in range(window, len(stream) + 1):
        if (stream[t - window : t].mean() - alpha) / se > z:
            return t
    return None


def simulate(rng: np.random.Generator, n: int, rate: float) -> np.ndarray:
    return (rng.random(n) < rate).astype(np.float64)


def _false_alarm_rate(alarm: Callable[[np.ndarray], int | None], streams: list[np.ndarray]) -> float:
    return sum(alarm(s) is not None for s in streams) / max(len(streams), 1)


def _calibrate_threshold(
    make_alarm: Callable[[float], Callable[[np.ndarray], int | None]],
    streams: list[np.ndarray],
    target_fa: float,
    lo: float,
    hi: float,
    iterations: int = 22,
) -> tuple[float, float]:
    """Bisect a monitor's threshold until its empirical FA rate hits the target.

    Monotone in the threshold (a higher bar can only alarm less), so bisection
    is well posed. Returns ``(threshold, achieved_fa)``.
    """
    best = (hi, _false_alarm_rate(make_alarm(hi), streams))
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        fa = _false_alarm_rate(make_alarm(mid), streams)
        if fa > target_fa:
            lo = mid
        else:
            hi = mid
            best = (mid, fa)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--streams", type=int, default=200)
    ap.add_argument("--length", type=int, default=5000)
    ap.add_argument("--changepoint", type=int, default=500)
    ap.add_argument("--shift-rates", nargs="*", type=float, default=[0.35, 0.38])
    ap.add_argument("--shift-names", nargs="*", default=["gretel-like", "nemotron-like"])
    ap.add_argument("--window", type=int, default=200)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    null_streams = [simulate(rng, args.length, args.alpha) for _ in range(args.streams)]

    # ---- Part A: validity at nominal thresholds --------------------------
    banner("Part A :: validity at nominal thresholds")
    nominal: dict[str, Callable[[np.ndarray], int | None]] = {
        "e-process": lambda s: EProcess(args.alpha, args.delta).run(s).alarm_time,
        "e-detector": lambda s: EDetector(args.alpha, args.delta).run(s).alarm_time,
        # Textbook choices a practitioner would reach for:
        "cusum (h=5)": lambda s: cusum_alarm(s, args.alpha, 5.0),
        "fixed-window (z=1.645, repeated)":
            lambda s: fixed_window_alarm(s, args.alpha, args.window, 1.645),
    }
    part_a = {}
    for name, fn in nominal.items():
        fa = _false_alarm_rate(fn, null_streams)
        part_a[name] = round(fa, 4)
        flag = "OK" if fa <= args.delta + 1e-9 else f"VIOLATES ({fa / args.delta:.1f}x)"
        print(f"[aegis] {name:34s} FA={fa:.3f}  nominal={args.delta}  {flag}")

    # ---- Part B: thresholds calibrated to a common FA rate ---------------
    banner(f"Part B :: delays at matched FA rate = {args.delta}")
    cusum_h, cusum_fa = _calibrate_threshold(
        lambda h: (lambda s: cusum_alarm(s, args.alpha, h)), null_streams, args.delta, 0.5, 400.0
    )
    fw_z, fw_fa = _calibrate_threshold(
        lambda z: (lambda s: fixed_window_alarm(s, args.alpha, args.window, z)),
        null_streams, args.delta, 1.0, 12.0
    )
    print(f"[aegis] calibrated CUSUM h={cusum_h:.2f} (FA={cusum_fa:.3f}); "
          f"fixed-window z={fw_z:.2f} (FA={fw_fa:.3f})")

    monitors: dict[str, Callable[[np.ndarray], int | None]] = {
        "e-process": lambda s: EProcess(args.alpha, args.delta).run(s).alarm_time,
        "e-detector": lambda s: EDetector(args.alpha, args.delta).run(s).alarm_time,
        "cusum (calibrated)": lambda s: cusum_alarm(s, args.alpha, cusum_h),
        "fixed-window (calibrated)":
            lambda s: fixed_window_alarm(s, args.alpha, args.window, fw_z),
    }
    calibrated_fa = {name: round(_false_alarm_rate(fn, null_streams), 4)
                     for name, fn in monitors.items()}

    delays: dict[str, dict[str, Any]] = {}
    for name, rate in zip(args.shift_names, args.shift_rates):
        shifted = [
            np.concatenate([
                simulate(rng, args.changepoint, args.alpha),
                simulate(rng, args.length - args.changepoint, rate),
            ])
            for _ in range(args.streams)
        ]
        row: dict[str, Any] = {}
        for monitor, fn in monitors.items():
            observed = [
                fn(s) - args.changepoint
                for s in shifted
                if (t := fn(s)) is not None and t > args.changepoint
            ]
            row[monitor] = {
                "median_delay": int(np.median(observed)) if observed else None,
                "mean_delay": round(float(np.mean(observed)), 1) if observed else None,
                "detected_fraction": round(len(observed) / len(shifted), 3),
            }
        delays[name] = row
        print(f"[aegis] {name}: " + ", ".join(
            f"{k}={v['median_delay']}" for k, v in row.items()))

    save(
        "exp04_monitor",
        {
            "experiment": "exp04_monitor",
            "alpha": args.alpha,
            "delta": args.delta,
            "n_streams": args.streams,
            "stream_length": args.length,
            "changepoint": args.changepoint,
            "part_a_nominal_false_alarm": part_a,
            "part_b_calibrated_thresholds": {
                "cusum_h": round(cusum_h, 4),
                "fixed_window_z": round(fw_z, 4),
                "achieved_false_alarm": calibrated_fa,
            },
            "detection_delay": delays,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
