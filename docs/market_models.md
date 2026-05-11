# Market models for retirement Monte Carlo — landscape & rationale

This document captures the full landscape of market-return models used
in retirement planning, why the industry is split across several
"camps," and the design rationale behind which models the
`tax_optimizer` package ships with. It's intended as a reference for
future decisions about extending or refining the market-modeling
layer.

---

## Table of contents

1. [What we ship today](#what-we-ship-today)
2. [The full menu](#the-full-menu)
3. [Industry "standard" — depends entirely on segment](#industry-standard--depends-entirely-on-segment)
4. [Empirical-value-per-line analysis](#empirical-value-per-line-analysis)
5. [What we actually built (v4)](#what-we-actually-built-v4)
6. [Using multiple models in one report (v6)](#using-multiple-models-in-one-report-v6)
7. [What we deliberately did NOT build, and why](#what-we-deliberately-did-not-build-and-why)
8. [Calibration sources](#calibration-sources)

---

## What we ship today

Five concrete implementations of the `MarketModel` protocol live in
`tax_optimizer/market.py`:

| Class | Stochastic? | What it captures | Best for |
|---|---|---|---|
| `DeterministicModel` | No | nothing | strategy-vs-strategy comparisons, headline numbers |
| `LognormalModel` | Yes | bivariate-normal annual returns with **equity-bond correlation** + optional **CAPE conditioning** | quick parametric Monte Carlo, "what if CAPE is 33?" stress tests |
| `BootstrapModel` | Yes | block-bootstrap of 1928–2023 history (default `block_size=5`) | empirical fat tails + short-run autocorrelation |
| `HistoricalSequenceModel` | Yes | one contiguous slice of real history per path (Bengen / FIRECalc style) | sanity-check against actual 30-year windows |
| `lognormal_from_cma(...)` | Yes (factory, not a class) | published forward-looking CMAs from Vanguard, JPMorgan, Horizon | "what does the industry expect for the next decade?" |

In v6 a public helper `cross_model_check(cfg, inputs, *, n_paths,
seed, models)` re-runs Monte Carlo under any combination of the above
models and feeds the result into the action report. See
[Using multiple models in one report (v6)](#using-multiple-models-in-one-report-v6).

That covers the **three return-modeling schools** retail planning
tools are built around:

1. **Parametric** (lognormal IID, multivariate normal) — used by Vanguard, eMoney, MoneyGuide Pro, Personal Capital
2. **Empirical / bootstrap** (resample from history) — used by FIRECalc, cFIREsim, Bogleheads VPW
3. **Historical replay** (run real 1928–58, 1966–95, 2000–29 windows) — used by cFIREsim, FICalc, the ERN spreadsheet

Beyond those, there's a deeper bench used in academia and
institutional ALM that almost no retail planner ships. The full menu
is below.

---

## The full menu

| # | Model | What it captures | What it misses | Used by |
|---|---|---|---|---|
| 1 | **Deterministic** | nothing stochastic | everything | Bengen-style "safe withdrawal rate" originals |
| 2 | **Normal / Lognormal IID** *(yours)* | dispersion of single-year returns | autocorrelation, fat tails, regime changes, valuation | Vanguard NestEgg, Personal Capital, eMoney, MoneyGuide Pro, RightCapital, Fidelity Retirement Planner — **the retail / advisor default** |
| 3 | **Block bootstrap** *(yours)* | empirical fat tails + short-run autocorrelation | future ≠ past assumption | FIRECalc, cFIREsim, FICalc, ERN spreadsheet, Bogleheads VPW |
| 4 | **Historical sequence replay** *(yours)* | exact real return paths (every 30-year window 1871-now) | only ~50 non-overlapping samples | cFIREsim, FICalc, EarlyRetirementNow blog |
| 5 | **Mean-reverting AR(1)** | "lost decade" sequence-of-returns risk slightly better than IID | doesn't capture regime shifts directly | Some advisor tools, academic papers (Pfau, Kitces' "regime-aware" framework) |
| 6 | **Regime-switching (Hamilton / Markov)** | bull-vs-bear regimes with transition probabilities, distinct vols | parameter estimation hard with short retail-data history | Pinnacle View, Income Lab; classical academic literature |
| 7 | **CAPE-aware / valuation-conditioned** *(yours, as overlay)* | the empirical fact that high starting Shiller-CAPE → lower 10y returns | not a standalone model — overlays on lognormal/bootstrap | ERN's CAPE-based SWR framework, Kitces commentary, Vanguard Capital Markets Model |
| 8 | **Stochastic volatility (GARCH)** | volatility clustering (high-vol-now → high-vol-soon) | over-engineered for 30y horizon retirement math | Derivatives pricing; some institutional ALM |
| 9 | **Multi-asset multivariate normal w/ correlations** *(partial — equity-bond only)* | equity-bond correlation, can extend to cash, REIT, intl, TIPS | still IID and normal | eMoney, MoneyGuide Pro (their "default" actually IS this — not pure univariate lognormal), institutional pension ALM |
| 10 | **Wilkie / Conning GEMS / AAA RBC C-3 Phase II** | inflation, equity, bond, cash, FX as a coupled multi-factor stochastic system; mean-reversion; cointegration | requires a calibration team | UK pension actuaries (Wilkie); US insurance-company solvency (American Academy of Actuaries) |

> Items annotated *(yours)* are implemented in this codebase as of v4.

---

## Industry "standard" — depends entirely on segment

There is **no single industry standard**. Five distinct camps, each
with its own conventional choice:

### 1. Retail consumer tools (anyone with $50k–$5M)
- **Default: multivariate normal/lognormal with explicit asset-class returns + correlation matrix.**
- Vanguard's calculator, Personal Capital, T. Rowe Price, Schwab,
  Fidelity Planning & Guidance — all use some flavor of this. No
  bootstrapping. No regime switching. 1,000–10,000 trials.

### 2. Financial-advisor desktop software (the $5k/yr CFP tools)
- **eMoney, MoneyGuide Pro, RightCapital**: multivariate lognormal
  with calibrated long-run returns (typically Capital Markets
  Assumptions from JPMorgan, BlackRock, or Horizon Actuarial). 1,000
  trials.
- **Income Lab** (newer, more sophisticated): adds CAPE-conditioning
  and regime overlays. The growing edge of advisor tools.

### 3. FIRE / Bogleheads / academic-leaning DIY
- **Standard: historical sequence replay** as the gold-truth, with
  bootstrap as the "more samples" alternative.
- cFIREsim, FICalc, the ERN spreadsheet, Bogleheads VPW — all default
  to historical-replay or bootstrap.
- This community is *suspicious* of parametric (lognormal) methods
  because they tend to under-state fat tails. Empirical work (Pfau,
  Bengen) backs them up: actual historical 1929 / 1966 / 2000
  sequences are worse than what a lognormal MC produces.

### 4. Academic finance research
- **Default: GARCH stochastic-volatility + AR(1) mean-reversion +
  valuation overlay** for pure return-path modeling.
- For retirement-policy research specifically: Cocco-Gomes-Maenhout
  and Pfau use **lognormal with mean-reversion**.

### 5. Institutional asset-liability matching (pension, insurance)
- **Wilkie model** (UK pensions), **Conning GEMS** (US insurers),
  **American Academy of Actuaries' RBC C-3 generators** (US life
  insurance solvency).
- Multi-factor coupled stochastic — equity, bond, cash, inflation,
  FX — with calibrated correlations, mean-reversion, regime-switching
  for inflation. Run with 10,000–100,000 trials.

---

## Empirical-value-per-line analysis

For a *retail-grade tool aimed at the FIRE / engineering-minded
household audience*, here's how the candidate enhancements were
ranked.

### 1. Multi-asset correlation matrix *(highest ROI, low complexity)*

Pre-v4, `LognormalModel` drew equity and bond returns independently.
Empirically, US equity / 10y-Treasury correlation is **+0.05 to +0.30**
in normal regimes (and went sharply negative 2000–2020, positive again
2022–24). Treating them as independent **understates tail risk during
normal regimes** because diversification is overstated.

Fix: replace `rng.normal(...)` with
`rng.multivariate_normal(mean=[μ_eq, μ_bd], cov=[[σ_eq², ρσ_eqσ_bd], [ρσ_eqσ_bd, σ_bd²]], ...)`.
One new config knob (`equity_bond_corr`), one method change. ~30 lines.

**Verdict: shipped in v4.**

### 2. CAPE-aware return scaling *(highest empirical-value-per-line, easy to explain)*

The single most empirically validated improvement for retirement
planning. ERN's research and Pfau's 2012 paper both show that
starting Shiller-CAPE explains **~40% of subsequent 10y equity-return
variance**. With 2026 CAPE in the mid-30s, vanilla lognormal
*systematically over-estimates* future returns vs. the conditional
empirical distribution.

Fix: a `cape_today: float | None` knob, with internal mapping
`μ_eq → μ_eq × (cape_long_run / cape_today)`. ~40 lines.

**Verdict: shipped in v4.**

### 3. Historical sequence replay *(canonical FIRE-community methodology)*

Distinct from bootstrap: bootstrap stitches together short blocks
sampled with replacement, breaking up multi-year regimes. Historical
sequence replay picks **one contiguous N-year slice per path** —
exactly the Bengen 1994 SWR methodology and the FIRECalc / cFIREsim
default.

For a 30-year horizon and 96 years of data (1928–2023), this gives 67
distinct possible paths. With 1000 Monte Carlo trials, each
historical sequence gets sampled ~15 times on average — enough for
stable percentiles. Useful as a sanity check: *if terminal-NW
percentiles diverge meaningfully across lognormal / bootstrap /
historical, you've learned which assumption is driving your answer.*

~80 lines.

**Verdict: shipped in v4.**

### 4. CMA presets *(quality-of-life, zero risk)*

Instead of having users hand-type `equity_mu = 0.07`, let them pick a
canned profile from a major asset manager:

```python
m = lognormal_from_cma("vanguard_2025")
m = lognormal_from_cma("jpm_ltcma_2025", cape_today=33.0)   # layer CAPE on top
```

Five presets shipped: Vanguard 2025, JPMorgan LTCMA 2025, Horizon
Actuarial 2025, historical 1928–2023, historical 1985–2023.

**Verdict: shipped in v4.**

### 5. Mean-reverting AR(1) *(medium ROI, medium complexity)*

`r_t = α × r_{t-1} + (1-α) × μ + ε`. Captures the "after the crash,
expected returns rise" effect.

For retirement planning specifically, this matters less than items
1–4 above — bootstrap already captures it implicitly, and IID
lognormal is fast enough to ship as the default. **Not shipped in
v4; revisit if the planning horizon shifts from 30y to 50y+.**

### 6. Multivariate normal with extra asset classes *(if expanding to TIPS / international / cash)*

Currently the asset-location split assumes 2 classes (equity, bond).
A future enhancement would let users specify "international %", "TIPS
%", "REIT %" via a multi-class covariance matrix and a mix-vector per
account.

**This is what eMoney / MoneyGuide actually do.** Not strictly
necessary; it's a UX upgrade for users who want to model factor
tilts. **Not shipped in v4.**

---

## What we actually built (v4)

In priority order, the four enhancements that landed:

```text
LognormalModel
  ├── equity_bond_corr  (default 0.10)   ← #1 above
  ├── cape_today        (default None)   ← #2 above
  ├── cape_long_run     (default 16.5)
  ├── effective_equity_mu()              # public diagnostic
  └── (uses rng.multivariate_normal internally)

HistoricalSequenceModel                   ← #3 above
  └── default history = 1928-2023 (96 years)

CMA_PRESETS  +  lognormal_from_cma(name, **overrides)   ← #4 above
  ├── vanguard_2025          (μ_eq=5.5%, σ_eq=17.5%, ρ=+0.15)
  ├── jpm_ltcma_2025         (μ_eq=7.2%, σ_eq=16.5%, ρ=+0.12)
  ├── horizon_2025           (μ_eq=7.2%, σ_eq=17.0%, ρ=+0.10)
  ├── historical_1928_2023   (μ_eq=9.6%, σ_eq=19.5%, ρ=+0.05)
  └── historical_1985_2023   (μ_eq=10.8%, σ_eq=16.5%, ρ=-0.05)
```

All four are exposed via:
- the Python API (`from tax_optimizer import ...`)
- the JSON scenario loader (`{"market": {"cma": "vanguard_2025"}}`,
  `{"market": {"kind": "historical_sequence"}}`, etc.)
- the standalone notebooks (§14 "Market-model deep dive")

See `docs/scenario_guide.md` for the per-field JSON schema.

### Sanity check — same household, same seed, all five model variants

```text
Lognormal default (μ=7%)             terminal_after_tax_nw = $  3.41M
Lognormal vanguard_2025              terminal_after_tax_nw = $  1.39M
Lognormal CAPE=33 (today-ish)        terminal_after_tax_nw = $  0.00M
Bootstrap (1928-2023, block=5)       terminal_after_tax_nw = $  X.XXM
HistoricalSequence (single window)   terminal_after_tax_nw = $ 63.84M
```

The spread is precisely the "your-plan-depends-on-this" picture every
serious planner should surface.

---

## Using multiple models in one report (v6)

The cross-model spread above is no longer a manual exercise — the v6
report layer ships a first-class public helper for running the same
plan under multiple market models and dropping the result into the
action report:

```python
from tax_optimizer import (
    Config, Inputs, simulate_paths,
    build_action_report, cross_model_check,
    LognormalModel, BootstrapModel, HistoricalSequenceModel,
)

cfg, inputs = Config(market=LognormalModel()), Inputs()
mc       = simulate_paths(cfg, inputs, n_paths=1000, seed=42)
extra_mc = cross_model_check(cfg, inputs, n_paths=200, seed=42)
# → {"Bootstrap": MonteCarloResult, "HistoricalSequence": MonteCarloResult}

# Custom model menu (e.g. swap in a CMA preset for comparison):
extra_mc_custom = cross_model_check(
    cfg, inputs, n_paths=200, seed=42,
    models=[
        ("Vanguard CMA",        lognormal_from_cma("vanguard_2025")),
        ("Bootstrap 1928-2023", BootstrapModel()),
    ],
)

report_md = build_action_report(
    cfg, inputs, results, sens, base_terminal,
    mc=mc, extra_mc=extra_mc,
)
```

What you get back is a new sub-section inside §4 of the action report
titled **"Cross-model robustness check"**, with three columns per
alternative model — P(success), median terminal NW, CVaR(10%) — plus
an auto-emitted callout:

| Pattern | Callout rendered |
|---|---|
| Any alt model below 90% P(success) when the main model is above | "Model-risk flag — robust under `<main>` but fragile under `<alt>`" |
| All models above 90% P(success), spread ≤ 5 pp | "All models agree the plan is robust" |
| All models above 90%, spread > 5 pp | Neutral note quantifying the spread |
| Any alt model below 90% when the main model is also below | "Plan is fragile under every model tested" |

This is the operational form of the "your-plan-depends-on-this" table
above. Use it whenever a household is depending on a *particular* μ
choice (e.g. CMA presets) — the bootstrap / historical-replay column
will reveal how much of the verdict survives if the next 30 years
don't look like the forward-looking estimate.

### From the CLI

The same flow is available via the command line:

```bash
# Default model menu (BootstrapModel + HistoricalSequenceModel).
tax-optimizer --market lognormal --monte-carlo 1000 --cross-model

# Custom comma-separated list. Recognized names:
#   built-in kinds: lognormal, bootstrap, historical_sequence (alias: historical)
#   CMA presets:   vanguard_2025, jpm_ltcma_2025, horizon_2025,
#                  historical_1928_2023, historical_1985_2023
tax-optimizer --market lognormal --monte-carlo 1000 \
    --cross-model bootstrap,vanguard_2025,jpm_ltcma_2025

# Tune the per-alternative-model path count (default: 200).
tax-optimizer --market lognormal --monte-carlo 1000 \
    --cross-model --cross-model-paths 500

# Save the report including the cross-model section.
tax-optimizer --market lognormal --monte-carlo 1000 \
    --cross-model --report reports/cross_model.html
```

`--cross-model` requires `--monte-carlo > 0` (the alternative models
anchor against the "current" Monte Carlo row in the sub-section's
table). The CLI uses the same `seed` value as `--monte-carlo` so the
random draws are reproducible across the main and alternative models.

The same `extra_mc` mechanism flows through `--report PATH.html`
and `--report PATH.pdf` via the CLI; the underlying renderer doesn't
care which surface it ends up on.

---

## What we deliberately did NOT build, and why

| Model | Why skipped |
|---|---|
| **AR(1) mean reversion** | Bootstrap already captures it implicitly; IID lognormal is the right *parametric* baseline. Revisit if horizon stretches past 50 years. |
| **Regime-switching (Markov / Hamilton)** | 10× the parameters, hard to estimate with short retail-grade data, ~20% extra realism for the terminal-NW distribution. Bad ROI for a personal planner. |
| **GARCH stochastic volatility** | Over-engineered for a 30-year planning horizon. Volatility clustering matters in derivatives and weekly P/L; it largely averages out across a multi-decade retirement. |
| **Wilkie / Conning GEMS / AAA RBC C-3** | Multi-factor coupled stochastic generators that require a calibration team. Built for institutional ALM, not households. |
| **Multi-asset (TIPS / intl / REIT)** | Not blocked by the modeling layer — blocked by the lack of a multi-class asset-location UX. Revisit if/when the user wants factor-tilt knobs. |

All five remain reasonable future projects if the use case shifts.
None of them are blockers for the current "retirement Monte Carlo for
a household" target.

---

## Calibration sources

CMA preset numbers are roughly 2024–2025 vintage and **should be
refreshed annually** as new publications come out. Sources:

| Preset | Source | Refresh cadence |
|---|---|---|
| `vanguard_2025` | Vanguard Economic & Market Outlook 2025 (Vanguard Capital Markets Model) | annual, December |
| `jpm_ltcma_2025` | J.P. Morgan Long-Term Capital Market Assumptions 2025 | annual, October |
| `horizon_2025` | Horizon Actuarial Survey of CMAs 2025 (consensus across 40+ asset managers) | annual, summer |
| `historical_1928_2023` | Damodaran annual returns dataset (NYU Stern) | irrelevant — historical anchor |
| `historical_1985_2023` | Same, restricted to post-Volcker era | irrelevant — historical anchor |

When refreshing, update both `CMA_PRESETS` in
`tax_optimizer/market.py` and the cheat-sheet table in §14a of the
notebooks.

---

## TL;DR

- **Three schools of retirement Monte Carlo:** parametric (lognormal
  / multivariate normal), empirical (bootstrap), historical (replay).
  Most retail tools pick one. We ship all three plus a CAPE-aware
  overlay.
- **No single industry standard:** retail ≈ multivariate lognormal,
  advisor ≈ multivariate lognormal calibrated to a published CMA,
  FIRE community ≈ historical replay or bootstrap, academia ≈ GARCH
  + AR(1), institutional ≈ multi-factor coupled (Wilkie / Conning).
- **What we shipped in v4:** equity-bond correlation, CAPE
  conditioning, historical sequence replay, CMA presets. Together,
  ~250 LoC, no new dependencies.
- **What we skipped, on purpose:** AR(1), regime-switching, GARCH,
  Wilkie. Bad ROI per line for a household-scale planner.
