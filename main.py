"""gta6-twitch-ccu-forecast-01 entry: kernel-weighted launch-shape forecast.

Model (data-driven from each case's own `comparables`, no leakage from the target's
launch-day observations beyond hour 1):

1. Align every comparable's hourly rows to its release day (day_start + [0,24) UTC),
   hours 1..24; skip comparables missing any of hours 1..24 or without an hour-1 print.
2. For each comparable i compute log-ratios r_i(h) = log(v_i(h) / v_i(1)), h = 1..24.
3. Median path m(h): Gaussian kernel on log v1 (bandwidth 1.0) weighted mean of the
   r_j(h) over ALL comparables, centered at the TARGET's log(p1) — conditioning the
   center on the target's own hour-1 size (nested leave-one-comparable-out log-pinball
   CV: 0.204 vs 0.402 for a global-median center).
4. Spread: per-hour leave-one-out residuals e_i(h) = r_i(h) - m_i(h), where m_i(h) is
   the kernel fit with curve i held out (centered at curve i's own log v1), are heavy
   tailed and asymmetric (right-skewed spikes), so quantile offsets o_tau(h) are
   empirical residual quantiles (linear interpolation, monotone by construction),
   lightly regularised toward a symmetric gaussian offset with s(h) = residual std:
       o_tau(h) = 0.8 * empirical_tau(h) + 0.2 * z_tau * s(h)
   This keeps honesty with 37 curves while damping small-sample noise.
5. q_tau(h) = v1 * exp(m(h) + o_tau(h)) for the five scored levels, rounded half-up to
   2 decimals. If a case carries no usable comparables, fall back to the starter's
   persistence random walk (sigma 0.15 per sqrt hour).

Deterministic: files processed in sorted name order, pure float arithmetic, sorted
residual lists; JSON separators fixed. Python 3.13 standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
SIGMA_FALLBACK = 0.15
BANDWIDTH = 1.0
EMP_WEIGHT = 0.8
HOURS = list(range(1, 25))


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up (JavaScript Math.round parity)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def launch_day_curves(comparables: list | None) -> list[tuple[str, float, list[float]]]:
    """(name, log v1, [log(v_h / v_1) for h = 1..24]) for usable comparables."""
    curves = []
    if not comparables:
        return curves
    for comp in comparables:
        if not isinstance(comp, dict):
            continue
        name = comp.get("name")
        if not isinstance(name, str):
            continue
        day_start = comp.get("day_start")
        hourly = comp.get("hourly")
        if not day_start or not isinstance(day_start, str) or not isinstance(hourly, list):
            continue
        try:
            ds = parse_utc(day_start)
        except ValueError:
            continue
        if "Grand Theft Auto VI" in name:
            continue  # the target's own pre-release category: not a launch curve
        by_hour = {}
        for row in hourly:
            if not isinstance(row, dict) or "from" not in row or "viewers" not in row:
                continue
            frm = row["from"]
            if not isinstance(frm, str):
                continue
            try:
                t = parse_utc(frm)
            except ValueError:
                continue
            try:
                v = float(row["viewers"])
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(v):
                continue
            offset = (t - ds).total_seconds() / 3600.0
            h = int(offset)
            if 0 <= offset - h < 1e-9 and 1 <= h + 1 <= 24 and h + 1 not in by_hour:
                by_hour[h + 1] = max(1.0, v)
        if len(by_hour) != 24 or 1 not in by_hour:
            continue
        v1 = by_hour[1]
        curves.append((name, math.log(v1), [math.log(by_hour[h]) - math.log(v1) for h in HOURS]))
    return curves


def quantile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation quantile of an ascending list (deterministic)."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = p * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def model_from_curves(curves: list[tuple[str, float, list[float]]], target_lv1: float):
    """Kernel regression of launch shape on log v1.

    Returns (m[h -> log-ratio], offsets[level][h]). The median path m(h) is the
    Gaussian-kernel weighted mean of all comparables' r_j(h), centered at the
    target's log(p1). Offsets come from leave-one-out residuals: each curve's fit
    excludes itself and is centered at that curve's own log v1.
    """
    n = len(curves)
    if n < 3:
        return None
    lv1 = [c[1] for c in curves]
    spread = quantile(sorted(lv1), 0.75) - quantile(sorted(lv1), 0.25)
    bw = BANDWIDTH if spread > 0 else 1.0

    def kernel_fit(center: float, skip: int) -> list[float] | None:
        num = [0.0] * 24
        den = 0.0
        for j, (_, xj, rj) in enumerate(curves):
            if j == skip:
                continue
            w = math.exp(-0.5 * ((xj - center) / bw) ** 2)
            den += w
            for k in range(24):
                num[k] += w * rj[k]
        if den <= 0:
            return None
        return [num[k] / den for k in range(24)]

    # Median path: kernel-weighted mean around the TARGET's log(p1), all curves in.
    target_fit = kernel_fit(target_lv1, -1)
    if target_fit is None:
        return None
    m_path = {h: target_fit[h - 1] for h in HOURS}
    # Residuals against each curve's own leave-one-out fit (centered at its log v1).
    resid = {h: [] for h in HOURS}
    for i, (_, x, r) in enumerate(curves):
        fit = kernel_fit(x, i)
        if fit is None:
            fit = r[:]
        for h in HOURS:
            resid[h].append(r[h - 1] - fit[h - 1])
    offsets = {}
    for lvl, z in zip(LEVELS, Z):
        row = {}
        for h in HOURS:
            e = sorted(resid[h])
            s = math.sqrt(sum(v * v for v in e) / len(e))
            row[h] = EMP_WEIGHT * quantile(e, lvl) + (1.0 - EMP_WEIGHT) * z * s
        offsets[lvl] = row
    return m_path, offsets


def fallback_offsets():
    """Starter persistence spread: 0.15 * sqrt(h-1) per normal quantile, flat median."""
    offsets = {}
    for lvl, z in zip(LEVELS, Z):
        offsets[lvl] = {h: z * SIGMA_FALLBACK * math.sqrt(h - 1) for h in HOURS}
    return {h: 0.0 for h in HOURS}, offsets


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    fitted = model_from_curves(launch_day_curves(case.get("comparables")), math.log(p1))
    m_path, offsets = fitted if fitted is not None else fallback_offsets()
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        base = math.log(p1) + m_path.get(h, 0.0)
        qs = [math.exp(base + offsets[lvl].get(h, 0.0)) for lvl in LEVELS]
        qs = [min(max(q, 0.01), 1e9) for q in qs]
        for k in range(1, 5):
            qs[k] = max(qs[k], qs[k - 1])
        hours.append({"hour": h, "q": [round2(q) for q in qs]})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
