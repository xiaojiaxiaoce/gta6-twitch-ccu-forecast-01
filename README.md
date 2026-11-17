# gta6-twitch-ccu-forecast-01

Forecast Grand Theft Auto VI's launch-day Twitch concurrent viewers, hour by hour.

## Entry point

`main.py`, Python 3.13, **standard library only** (no network, no third-party packages).

## Running

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` JSON case from `MAKEFUN_INPUT_DIR` (default `/input`) and writes
`<id>.out` to `MAKEFUN_OUTPUT_DIR` (default `/output`):

```json
{"hours": [{"hour": 2, "q": [1000, 1200, 1500, 1900, 2400]}]}
```

23 rows, hours 2–24 each once, five finite positive non-decreasing values per row,
quantile levels 0.05, 0.25, 0.5, 0.75, 0.95, values rounded half-up to 2 decimals.

## Model

Deterministic, fitted per case from that case's own data:

1. **Launch-shape curves.** Each comparable in `comparables` is aligned to its
   `day_start`; hours 1–24 of the release day become log-ratios
   `r_i(h) = log(v_i(h) / v_i(1))`. Comparables missing any launch-day hour are
   skipped, as is the target's own pre-release category entry.
2. **Median path `m(h)`.** Gaussian kernel on `log v1` (bandwidth 1.0) weighted mean
   of the `r_j(h)` over all comparables, centered at the **target's** `log(p1)` —
   conditioning the center on the target's own hour-1 size. (Nested
   leave-one-comparable-out log-pinball CV favored this over a global-median center.)
3. **Quantile spread.** Per-hour leave-one-out residuals `e_i(h) = r_i(h) − m_i(h)`
   (each curve's fit excludes itself and is centered at its own `log v1`) are
   heavy-tailed and right-skewed, so offsets combine an empirical residual quantile
   (weight 0.8) with a Gaussian `z_tau · s(h)` term (weight 0.2, `s(h)` the residual
   standard deviation) for small-sample stability.
4. **Output.** `q_tau(h) = v1 · exp(m(h) + offset_tau(h))`, clamped to (0, 1e9] and
   enforced non-decreasing across levels.

## Fallback

If a case has no usable comparables (fewer than three full launch-day curves), the
model falls back to the starter's persistence random walk in log space:
`q_tau(h) = v1 · exp(z_tau · 0.15 · sqrt(h − 1))`.

## Determinism

Files are processed in sorted name order; all computation is pure float arithmetic on
sorted structures, so repeated runs write identical bytes.
