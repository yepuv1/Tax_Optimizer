# Scenario File Reference Guide

A field-by-field reference for `scenarios/*.json` files (the input format consumed by `tax-optimizer --scenario PATH` and `apply_scenario(...)`), assembled from the running explanations in this project's chat history.

> Use this alongside `tax_optimizer/scenario.py` (the loader / validator) and `scenarios/example01.json` / `scenarios/example02.json` (worked examples).

---

## Table of contents

- [Top-level structure](#top-level-structure)
- [`config.market` — return-model parameters](#configmarket--return-model-parameters)
  - [What `kind: "lognormal"` does](#what-kind-lognormal-does)
  - [The four numbers, decoded](#the-four-numbers-decoded)
  - [Calibration vs. real data](#calibration-vs-real-data)
  - [Three pre-baked profiles you can drop in](#three-pre-baked-profiles-you-can-drop-in)
  - [Where each parameter "should" sit, with sources](#where-each-parameter-should-sit-with-sources)
  - [The other three market kinds](#the-other-three-market-kinds)
- [`config.asset_location` — per-account equity / bond split](#configasset_location--per-account-equity--bond-split)
  - [The `uniform_equity_pct` shortcut](#the-uniform_equity_pct-shortcut)
  - [The per-bucket form](#the-per-bucket-form)
  - [Effect at simulation time](#effect-at-simulation-time)
  - [When to use which](#when-to-use-which)
- [`config.ss_cola_rate` — Social Security annual COLA](#configss_cola_rate--social-security-annual-cola)
- [`config.bracket_indexing_rate` — annual indexing of brackets, std deduction, and IRMAA](#configbracket_indexing_rate--annual-indexing-of-brackets-std-deduction-and-irmaa)
- [`config.taxable_equity_div_yield` / `taxable_bond_interest_yield` — explicit yield on taxable-account holdings](#configtaxable_equity_div_yield--taxable_bond_interest_yield--explicit-yield-on-taxable-account-holdings)
- [`config.state_regime` — state income tax regime (CA / NY / IL / MA / stateless)](#configstate_regime--state-income-tax-regime-ca--ny--il--ma--stateless)
- [`config.heir_marginal_rate` — bequest tax in terminal-NW objective](#configheir_marginal_rate--bequest-tax-in-terminal-nw-objective)
- [`config.medicare_base_b_d_premium` — base Medicare Part B + Part D premium](#configmedicare_base_b_d_premium--base-medicare-part-b--part-d-premium-tier-c-b) *(Tier C)*
- [`config.health_pre65_today` — pre-Medicare healthcare cost knob](#confighealth_pre65_today--pre-medicare-healthcare-cost-knob-tier-c-b) *(Tier C)*
- [`config.irmaa_lookback_years` — IRMAA 2-year MAGI lookback](#configirmaa_lookback_years--irmaa-2-year-magi-lookback-tier-c-b) *(Tier C)*
- [`config.aca_enabled` / `aca_benchmark_premium_per_adult` / `aca_max_contrib_pct` — ACA premium tax credit](#configaca_enabled--aca_benchmark_premium_per_adult--aca_max_contrib_pct--aca-premium-tax-credit-tier-c-b) *(Tier C)*
- [`config.stepup_at_first_death` — community-property full step-up](#configstepup_at_first_death--community-property-full-step-up-tier-c-b) *(Tier C)*
- [`config.optimize_ss_claim_age` — adds SS claim-age axes to the optimizer](#configoptimize_ss_claim_age--adds-ss-claim-age-axes-to-the-optimizer-tier-c-c) *(Tier C)*
- [`inputs.ss` — Social Security claim age, FRA, and benefits](#inputsss--social-security-claim-age-fra-and-benefits)
- [`inputs.pension` — pension cash balance and start age](#inputspension--pension-cash-balance-and-start-age)
- [`inputs.spouse_*_employer_match_*` — employer 401(k) match](#inputsspouse__employer_match__--employer-401k-match)
- [`inputs.spouse_*_traditional_ira_contrib` / `_roth_ira_contrib` / `_backdoor_roth` — IRA contribution paths](#inputsspouse__traditional_ira_contrib--_roth_ira_contrib--_backdoor_roth--ira-contribution-paths)
- [`inputs.spouse_*_mega_backdoor_enabled` / `_after_tax_401k_pct` — mega-backdoor Roth](#inputsspouse__mega_backdoor_enabled--_after_tax_401k_pct--mega-backdoor-roth)
- [`inputs.contrib.hsa_family` — household HSA contribution target](#inputscontribhsa_family--household-hsa-contribution-target)
- [Migration: `config.ss_start_age` / `config.pension_start_age`](#migration-configss_start_age--configpension_start_age)
- [Output metric: `median_ruin_year_offset`](#output-metric-median_ruin_year_offset)
- [CLI stress-test recipes](#cli-stress-test-recipes)

---

## Top-level structure

```json
{
  "config":  { ...simulation / strategy knobs... },
  "inputs":  { ...household "about me" data... }
}
```

Both sections are optional and fields are optional within each. Unknown keys raise a `ScenarioError` so typos don't silently no-op. The two top-level keys reflect the runtime `Config` and `Inputs` dataclasses one-to-one:

- **`Config`** — the *simulation / strategy*. Macro assumptions (`nominal_growth_rate`, `inflation`, `wage_growth`, `ss_cola_rate`, `bracket_indexing_rate`, `taxable_equity_div_yield`, `taxable_bond_interest_yield`), `rmd_start_age`, withdrawal / conversion strategy, and the modular blocks `market` / `asset_location` / `spending` / `mortality`.
- **`Inputs`** — the *household*. Spouse ages, retire ages, salaries, contribution rates, Roth-401(k) splits, **employer 401(k) match** (`spouse_*_employer_match_rate` / `_max_pct`), starting balances, Social Security amounts and per-spouse claim ages (`inputs.ss.start_age_a` / `start_age_b`, `fra_a` / `fra_b`), HSA contribution target (`inputs.contrib.hsa_family`), pension data (incl. start age `inputs.pension.start_age`), expected expenses, etc.

---

## `config.market` — return-model parameters

```json
"market": {
  "kind": "lognormal",
  "equity_mu":    0.07,
  "equity_sigma": 0.18,
  "bond_mu":      0.04,
  "bond_sigma":   0.06
}
```

This block configures the **stochastic market-return model** the Monte Carlo runs against. It only kicks in when you run with `--monte-carlo N` or `simulate_paths(...)`; deterministic single-path runs use whatever path the seed happens to draw.

### What `kind: "lognormal"` does

Picks the `LognormalModel` engine. There are three options:

- `"deterministic"` — constant equity / bond returns. Single fixed path. What the deterministic 4-strategy comparison uses.
- `"lognormal"` — independent yearly draws from a normal distribution. Cheap, fast, statistically clean, easy to explain.
- `"bootstrap"` — block-bootstrap resampling from real 1928–2023 history. Preserves fat tails and short-run autocorrelation.

Implementation note (despite the conventional `lognormal` naming): the model samples *arithmetic* annual returns from a normal distribution, not log returns. That's the standard simplification financial planners use for one-year horizons.

```python
# tax_optimizer/market.py
def begin_path(self, n_years: int, rng: np.random.Generator) -> None:
    self._equity_path = rng.normal(self.equity_mu, self.equity_sigma, size=n_years)
    self._bond_path   = rng.normal(self.bond_mu,   self.bond_sigma,   size=n_years)
```

### The four numbers, decoded

| Param | Example value | Means |
|---|---|---|
| `equity_mu` | `0.07` | Expected (mean) annual stock return — 7%/yr nominal |
| `equity_sigma` | `0.18` | Standard deviation of annual stock return — 18% (±1σ ≈ −11% to +25%) |
| `bond_mu` | `0.04` | Expected annual bond return — 4%/yr nominal |
| `bond_sigma` | `0.06` | Standard deviation of annual bond return — 6% |

Each year of every Monte Carlo path:
- Equity return is drawn `~ Normal(equity_mu, equity_sigma)`.
- Bond return is drawn `~ Normal(bond_mu, bond_sigma)`.
- Equity and bond draws are **independent** (zero correlation — historically slightly negative).
- Each year's draw is **independent of the previous year's** (no autocorrelation — historically there are runs of bull/bear years).

Each account bucket then grows by `equity_pct * equity_return + bond_pct * bond_return`, where `equity_pct` comes from `asset_location` (see below).

### Calibration vs. real data

The `LognormalModel` docstring documents what real US data looks like over 1928–2023 (Damodaran NYU Stern data, embedded in the bootstrap arrays):

| | Equity μ | Equity σ | Bond μ | Bond σ |
|---|---:|---:|---:|---:|
| **Long-run realized 1928–2023** | 9.6% | 19.5% | 4.6% | 7.7% |
| **Post-1990 realized** | ~10% | 18% | ~5% | 5.6% |
| **Example values above** (`0.07 / 0.18 / 0.04 / 0.06`) | 7% | 18% | 4% | 6% |

So the example values are **somewhat conservative forward-looking**: equity ~250 bps below the 96-year realized (high CAPE / low yield discount), bond μ near current 10y Treasury YTM, σ near post-1990 (which understates 1980s rate-shock volatility).

### Three pre-baked profiles you can drop in

| Profile | `equity_mu` | `equity_sigma` | `bond_mu` | `bond_sigma` | When to use |
|---|---:|---:|---:|---:|---|
| **Conservative** (stress test) | **0.05** | **0.19** | **0.035** | **0.07** | "What if the next 30 years look more like 2000–2009?" |
| **Base** (mainstream CMA) | **0.07** | **0.18** | **0.04** | **0.06** | Mid-of-the-road forward CMA consensus |
| **Optimistic** | **0.085** | **0.18** | **0.045** | **0.07** | Closer to long-run historical realized |
| **Long-run historical** *(reference only)* | **0.096** | **0.195** | **0.046** | **0.077** | Pure backward-looking; rarely used as base case |

A defensible "**median-of-major-CMAs**" calibration as of late 2025–2026:

```json
"market": {
  "kind": "lognormal",
  "equity_mu":    0.065,   /* -50 bps from 7%; matches Vanguard/BlackRock CMA mid-point */
  "equity_sigma": 0.18,    /* mainstream */
  "bond_mu":      0.045,   /* +50 bps; closer to current YTM */
  "bond_sigma":   0.07     /* +100 bps; honors 2022's rate shock */
}
```

### Where each parameter "should" sit, with sources

#### `equity_mu` — Expected annual stock return

| Source / model | Forward value | Notes |
|---|---:|---|
| Vanguard CMA (10-yr, 2024) | 4.5–6.5% (US) / 7.0–9.0% (intl) | Reflects high US valuations |
| BlackRock CMA (10-yr) | ~6.5% (US large) | |
| JPM Long-Term CMA 2024 | 7.0% (US large), 8.0% (intl) | |
| Research Affiliates | 6.0–7.0% | Valuation-aware |
| Morningstar (10-yr fwd) | ~9.3% | More optimistic |
| **Long-run realized 1928–2023** | **9.6%** | S&P 500 total return, Damodaran NYU |

**Defensible range: 5–8%.** Below 5% is a true bear-case stress test; above 8% is "history will repeat" — defensible but contrarian today.

#### `equity_sigma` — Annual stock-return stdev

| Reference | Value |
|---|---:|
| 1928–2023 realized | **19.5%** |
| Post-1990 realized | **18.0%** |
| Most professional Monte Carlos | **17–19%** |

**Defensible range: 16–20%.** Going below 15% understates real-world tail risk; above 22% is overly punitive given how diversified-portfolio realized vol has actually behaved.

#### `bond_mu` — Expected annual bond return

| Source | Forward value | Notes |
|---|---:|---|
| Vanguard CMA (US Agg) | 4.7–5.7% | Mostly current YTM |
| JPM Long-Term CMA | ~5.1% | |
| BlackRock | 4.0–5.0% | |
| **Long-run realized** | **4.6%** | 10y Treasury, 1928–2023 |

**Defensible range: 3.5–5.5%.** Today's 10-year yield (~4.3%) is roughly the bond μ a forward-looking planner would use.

#### `bond_sigma` — Annual bond-return stdev

| Reference | Value |
|---|---:|
| 1928–2023 realized | **7.7%** |
| Post-1990 realized | **5.6%** |
| Recent (2010s–early 2020s) | **6.5%** (with 2022 = a 14% loss outlier) |
| Most planning Monte Carlos | **5–8%** |

**Defensible range: 5–8%.** A `0.06` matches the post-1990 regime; `0.07` honors the 2022 rate-shock and 1980s-style high-vol regime.

### The other three market kinds

```json
/* Deterministic — single-path, no randomness. Used by deterministic runs. */
"market": { "kind": "deterministic", "equity": 0.06, "bond": 0.04 }

/* Bootstrap — block-resample real 5-year history blocks. Preserves fat tails. */
"market": { "kind": "bootstrap", "block_size": 5 }

/* Historical sequence replay — one contiguous N-year slice per path
   (Bengen / FIRECalc methodology). Distinct from bootstrap: preserves
   real multi-year regimes instead of stitching short blocks together. */
"market": { "kind": "historical_sequence" }

/* CMA preset shortcut — implies kind="lognormal", pulls a published
   forward-looking assumption set. See market_models.md for the menu. */
"market": { "cma": "vanguard_2025" }
"market": { "cma": "vanguard_2025", "cape_today": 33.0 }   /* layer CAPE on top */
```

Use `bootstrap` as a tail-risk reality check: it preserves real history's runs of bull/bear years and the 1929 / 1973 / 2008 fat tails that lognormal under-states.

Use `historical_sequence` when you want the canonical FIRE-community methodology — exact 30-year windows from 1928–2023 — as a sanity check against the parametric and bootstrap models.

> **For the full landscape of available models, the industry segments
> that use each one, and the design rationale behind which models
> we ship, see [`market_models.md`](market_models.md).**

---

## `config.asset_location` — per-account equity / bond split

```json
"asset_location": { "uniform_equity_pct": 0.7 }
```

`asset_location` describes the **equity / bond split** *per account type* (pretax 401(k)/IRA, Roth, taxable brokerage, HSA). It's the lever behind the "asset-located portfolio" recommendation in the action report — bonds in pretax (sheltered from ordinary income), equities in Roth (max tax-free compounding), etc.

### The `uniform_equity_pct` shortcut

Setting `uniform_equity_pct: 0.7` says "give every account type the same **70% stocks / 30% bonds** mix." It's the simplest form and intentionally skips the textbook asset-location optimization in favor of a single, easy-to-reason-about mix.

The scenario loader translates this shortcut into an `AssetLocation.uniform(equity_pct=0.7)` factory call, which is identical to writing the per-bucket form:

```json
"asset_location": {
  "pretax_equity_pct":  0.7,
  "roth_equity_pct":    0.7,
  "taxable_equity_pct": 0.7,
  "hsa_equity_pct":     0.7
}
```

You can use **one form or the other, never both** — combining `uniform_equity_pct` with any per-bucket key raises a `ScenarioError`.

### The per-bucket form

The "textbook asset-located" defaults (what `AssetLocation()` uses with no override):

```python
# tax_optimizer/market.py
@dataclass
class AssetLocation:
    pretax:  AssetMix = AssetMix(0.40, "pretax_balanced")     # 40% eq / 60% bond — bonds sheltered
    roth:    AssetMix = AssetMix(1.00, "roth_aggressive")     # 100% eq — max tax-free compounding
    taxable: AssetMix = AssetMix(0.80, "taxable_growth")      # 80% eq / 20% bond
    hsa:     AssetMix = AssetMix(0.80, "hsa_growth")          # 80% eq — long-horizon shelter
```

### Effect at simulation time

Each year, every account bucket grows by `equity_pct * equity_return + bond_pct * bond_return`:

```python
def annual_return(self, equity_r: float, bond_r: float) -> float:
    return self.equity_pct * equity_r + self.bond_pct * bond_r
```

With `uniform_equity_pct: 0.7` and the example lognormal market, the **expected** annual return for every bucket is:

\[
0.7 \times 7\% + 0.3 \times 4\% = 6.1\% \text{ nominal}
\]

vs. the textbook split where pretax compounds at ~5.2% (40/60), Roth at ~7% (100% equity), and taxable/HSA at ~6.4% (80/20). That's a deliberate trade-off: easier to reason about (one growth assumption everywhere) at the cost of missing **asset-location alpha** (~10–30 bps/yr lifetime by most studies).

### When to use which

- **`uniform_equity_pct: 0.7`** — what example02 uses. Reproduces the v1 single-scalar growth feel, just risk-adjusted to 70/30. Great default for first-pass planning when you don't yet hold a real bond sleeve.
- **Textbook asset-located** (`AssetLocation()` with no override, or per-bucket form) — meaningful for households that *actually* run different mixes per account. The asset-location lever is what produces the "bonds in pretax, equities in Roth" recommendation.
- **`uniform_equity_pct: 1.0`** (the `Config()` default) — pure-equity-everywhere; matches the v1 deterministic numbers byte-for-byte when also using a `DeterministicModel` and a flat spending profile.

---

## `config.ss_cola_rate` — Social Security annual COLA

```json
"config": {
  "ss_cola_rate": null
}
```

| Value | Means |
|---|---|
| `null` (default) | Follow `cfg.inflation` (so SS keeps real-dollar value across the horizon) |
| `0.0` | Disable COLA — SS stays flat-nominal forever (the v1 behavior; only useful as a regression hedge) |
| `0.025` | Pin SS COLA to a constant 2.5%/yr regardless of `cfg.inflation` |

SSA grants an annual cost-of-living adjustment based on Q3 CPI-W each year. We approximate that with a constant rate, applied as `(1 + ss_cola_rate) ** year_offset` to the FRA-PIA before it lands on the income line. **Without this on, a $32k MFJ benefit at 2.5% inflation under-counts SS PV by ~40-50% over a 30-year retirement** — that was the v1 bug.

---

## `config.bracket_indexing_rate` — annual indexing of brackets, std deduction, and IRMAA

```json
"config": {
  "bracket_indexing_rate": null
}
```

| Value | Means |
|---|---|
| `null` (default) | Follow `cfg.inflation` |
| `0.0` | Freeze brackets at the regime's quoted-year nominals across the entire horizon (the v1 stealth-bracket-creep behavior; useful as a "what if Congress freezes brackets" stress test) |
| `0.02` | Pin indexing to a constant rate (e.g. simulate slow-CPI environments) |

In real life the IRS indexes ordinary brackets, LTCG brackets, the standard deduction, and IRMAA tier MAGI thresholds + surcharges annually to chained CPI. The simulator applies `factor = (1 + bracket_indexing_rate) ** year_offset` to those four. **NIIT** ($250k MFJ / $200k single) and **SS provisional-income thresholds** ($32k/$44k MFJ) are intentionally NOT scaled — they're statutory and have not been indexed since 2013 and 1993 respectively.

Setting `bracket_indexing_rate=0.0` is also useful for stress-testing the impact of NIIT non-indexing: more and more retirees will cross the $250k MFJ MAGI threshold over time.

---

## `config.taxable_equity_div_yield` / `taxable_bond_interest_yield` — explicit yield on taxable-account holdings

```json
"config": {
  "taxable_equity_div_yield":   0.016,
  "taxable_bond_interest_yield": 0.035
}
```

| Field | Default | Means |
|---|---:|---|
| `taxable_equity_div_yield` | `0.016` | Qualified-dividend yield on the equity portion of `state.taxable` |
| `taxable_bond_interest_yield` | `0.035` | Ordinary-interest yield on the bond portion of `state.taxable` |

Each year the simulator hits AGI / NIIT / IRMAA / SS-provisional-income with this yield against the taxable account's start-of-year balance, allocated to equity vs bond using `asset_location.taxable`. Without this, post-retirement AGI has nothing in it during the gap-year between retirement and SS — implausible, and the bug it patched made tax-line projections look ~3% too low across the back half of retirement.

Set both to `0.0` to reproduce the v1 "taxable account is invisible to AGI between withdrawals" behavior.

Defaults reflect a roughly-2024 broad-market portfolio: ~1.6% qualified-div yield on US equities, ~3.5% interest on the bond sleeve.

---

## `config.state_regime` — state income tax regime (CA / NY / IL / MA / stateless)

```json
"config": {
  "state_regime": "CA",
  "state_regime_change_year_offset": null,
  "state_regime_change_target": null
}
```

State income tax can swing optimal Roth-conversion timing, optimal claim-age, and "stay vs. move" decisions in the high-tax states by hundreds of thousands of dollars over a 30-year horizon. The simulator ships five regime presets (string lookup, case-insensitive):

| Key | Description |
| --- | --- |
| `"stateless"` (or `"none"`) | No state income tax. Correct for FL / TX / WA / NV / TN / NH / AK / SD / WY. Default. |
| `"CA"` | California progressive brackets (1% – 12.3%), 9.3% top bracket on most working-age income, **HSA contributions NOT deductible at the state level**, SS exempt. |
| `"NY"` | New York progressive brackets (4% – 10.9%), SS exempt, **$20k/filer pension+IRA+Roth-conversion exclusion at 59½+** (so two-spouse couples shelter $40k/yr of retirement income from NY tax). |
| `"IL"` | Illinois flat 4.95% on wages but **all retirement income exempt** (SS, pension, IRA, 401(k), Roth conversion). The cleanest "great state to retire in if you converted aggressively beforehand" regime. |
| `"MA"` | Massachusetts flat 5%, SS exempt, IRA/pension taxed normally. The 9% MA millionaire surtax is intentionally not modeled. |

Change of regime mid-horizon (e.g. retire to FL after working in CA): set `state_regime_change_year_offset` to the move year and `state_regime_change_target` to the destination preset. The simulator switches over from that year onward.

Bracket thresholds + standard deduction + retirement exclusion all index annually using the same `bracket_indexing_rate` as federal, so a CA scenario run for 30 years doesn't suffer phantom bracket creep at the state level either.

For a state not bundled (OR / NJ / HI / MD / etc.), construct a custom `StateTaxRegime` instance in Python and pass it to `cfg.state_regime` directly — see `tax_optimizer/tax/state.py` for the dataclass shape and the bundled regimes as worked examples.

---

## `config.heir_marginal_rate` — bequest tax in terminal-NW objective

```json
"config": { "heir_marginal_rate": 0.22 }
```

Default `0.22`. The marginal rate the optimizer assumes heirs will pay on inherited *pretax* (and HSA) balances. Roth and taxable balances (stepped-up cost basis at death) flow through tax-free; pretax + HSA must be drained over 10 years post-inheritance under the SECURE Act and incur the heir's ordinary-income rate.

Set higher (`0.28` – `0.35`) if your heirs are themselves high earners; set to `0.0` to recover the v1 behavior of treating $1 pretax = $1 Roth at horizon (which silently over-rewards "leave it pretax" plans). The exact intra-decade timing of the SECURE 10-year drawdown isn't modeled — the simulator applies a flat haircut on the terminal pretax + HSA balance.

This knob only affects the **terminal-NW objective** (used by the optimizer, sensitivity, and Monte Carlo terminal stats). It does not change the simulated cash-flow path itself.

---

## `config.medicare_base_b_d_premium` — base Medicare Part B + Part D premium *(Tier C-B)*

```json
"config": { "medicare_base_b_d_premium": 2500 }
```

Combined Medicare Part B + Part D base premium per Medicare-enrolled spouse, in **today's dollars**. The simulator inflates it forward by `cfg.inflation` and bills it on every year either spouse is age ≥ 65 (separate from IRMAA). 2026 published Part B base ≈ $2,084/yr; a typical Part D plan adds ~$650/yr, landing in the $2.5–2.8k range. Default `2500` is a mid-of-range number; set higher if your Part D plan is rich, or `0` if you've already baked Medicare into `spending.base_spending`.

The simulator now exposes a `medicare_base_premium` column on the per-year DataFrame so you can audit the line directly.

---

## `config.health_pre65_today` — pre-Medicare healthcare cost knob *(Tier C-B)*

```json
"config": { "health_pre65_today": 12000 }
```

Household healthcare cost (in today's dollars) charged each year **at least one spouse is alive AND below 65**. Per-spouse share is the household total × (`n_pre_medicare` / 2). Inflates by `cfg.inflation`. Default `0` (back-compat — assumes you've baked it into base spending). For a pre-65 retiree on the open marketplace, $10–18k/spouse/yr is a realistic 2026 range.

If you set `aca_enabled=True`, this knob is ignored (the ACA benchmark premium replaces it for pre-65 spouses); see below.

---

## `config.irmaa_lookback_years` — IRMAA 2-year MAGI lookback *(Tier C-B)*

```json
"config": { "irmaa_lookback_years": 2 }
```

Default `2`. SSA bills IRMAA in year T using AGI from year T-2 (with a one-year fallback in filing-edge cases). Pre-Tier-C, the simulator used current-year AGI, which **overstated** the IRMAA cliff in any year with a transient income spike (Roth conversion, retirement-year severance). Under the lookback, a year-0 conversion shows up in IRMAA at year 2; for early simulation years where lag-2 AGI hasn't accumulated yet, IRMAA defaults to the lowest tier — matching SSA's first-year-of-Medicare rule.

Set to `0` to revert to the pre-Tier-C (current-year) behavior. Set to `1` to use prior-year AGI (a sometimes-useful approximation when your filing pattern lags).

---

## `config.aca_enabled` / `aca_benchmark_premium_per_adult` / `aca_max_contrib_pct` — ACA premium tax credit *(Tier C-B)*

```json
"config": {
  "aca_enabled": true,
  "aca_benchmark_premium_per_adult": 14000,
  "aca_max_contrib_pct": 0.085
}
```

Models the **post-IRA-2022 enhanced subsidies**: each enrolled adult's premium contribution is capped at `aca_max_contrib_pct` (default 8.5%) of MAGI; the credit is `max(0, benchmark - cap)`. No income cliff (the pre-2021 400% FPL cliff is gone). For a 60-year-old in 2026 the second-lowest-cost-silver-plan benchmark averages ~$14k/yr (varies wildly by state).

Modeling notes / scope:
- MAGI ≈ AGI for almost every household (untaxed SS adds < 1% in scope of typical ACA filers); we use AGI as a proxy.
- No FPL × household-size lookup (Tier D). The 8.5% cap applies cliff-free at all incomes ≥ 150% FPL post-2022.
- The credit offsets cash outflow (not a federal-tax line item) since most households take it as advance APTC paid directly to the insurer.
- If `aca_enabled=True` AND any spouse is below 65, the `health_pre65_today` knob is **bypassed** — the benchmark premium replaces it and is offset by the credit.

`aca_benchmark_premium` and `aca_apt_credit` are exposed as DataFrame columns for auditing.

---

## `config.stepup_at_first_death` — community-property full step-up *(Tier C-B)*

```json
"config": { "stepup_at_first_death": true }
```

Default `false` (conservative). When `true`, the surviving spouse's `cumulative_basis` on the taxable account is reset to fair-market-value the year the first spouse dies. Models the **community-property full step-up** (CA, WA, ID, NM, AZ, LA, NV, TX, WI). Common-law half-step-up (the rest of the country) is in the Tier D backlog.

Mechanically: at the start of the year-of-first-death, `cumulative_basis = state.taxable`. Future taxable-bucket withdrawals during that survivor's lifetime realize gains only on appreciation **after** the step-up date.

---

## `config.optimize_ss_claim_age` — adds SS claim-age axes to the optimizer *(Tier C-C)*

```json
"config": { "optimize_ss_claim_age": true }
```

Default `false`. When `true`, the strategy optimizer adds `ss_claim_age_a` and `ss_claim_age_b` to its decision vector — discrete grid `{62, 65, 67, 70}`. Major lever for asymmetric couples (one high earner, one low). Without this, the optimizer treats `inputs.ss.start_age_a` / `start_age_b` as fixed.

When `inputs.spouse_*_mega_backdoor_enabled=True`, the optimizer **also** adds `mega_backdoor_pct_*` axes automatically — no separate flag needed. Both Tier-C axis additions are dynamically discovered by `_build_decision_vector_meta(cfg, inputs)`; see [`tax_optimizer/optimizer.py`](../tax_optimizer/optimizer.py) for the contract.

Pair this with `--mc-seed N` (or `optimize_household(..., mc_seed=N)`) when running a stochastic-market objective; the seed pins the MC draw so repeated optimizer runs are apples-to-apples.

---

## `inputs.ss` — Social Security claim age, FRA, and benefits

```json
"inputs": {
  "ss": {
    "monthly_spouse_a": 4030,
    "monthly_spouse_b": 2208,
    "start_age_a":     70,
    "start_age_b":     65,
    "fra_a":           67,
    "fra_b":           67
  }
}
```

| Field | Type | Default | Means |
|---|---|---:|---|
| `monthly_spouse_a` | float | `2700.0` | Spouse A's **PIA at FRA**, today's dollars |
| `monthly_spouse_b` | float | `2200.0` | Spouse B's **PIA at FRA**, today's dollars |
| `start_age` | int | `70` | Legacy single-knob fallback; applies to both spouses iff `start_age_a` / `start_age_b` are unset |
| `start_age_a` | int? | `null` | Per-spouse claim age for Spouse A; overrides `start_age` |
| `start_age_b` | int? | `null` | Per-spouse claim age for Spouse B; overrides `start_age` |
| `fra_a` | int | `67` | Full Retirement Age, Spouse A. `67` is correct for anyone born 1960 or later (covers anyone hitting retirement age in this model's typical horizon) |
| `fra_b` | int | `67` | Full Retirement Age, Spouse B |

**Important** — interpret `monthly_spouse_*` as the SSA-quoted **PIA at FRA**, NOT the at-claim amount. The simulator scales it actuarially based on `start_age_*` vs `fra_*`:

- **Below FRA (early claim):** −5/9% per month for the first 36 months early, then −5/12% per month thereafter. Age 62 vs FRA 67 ⇒ −30%.
- **At FRA:** ×1.00 (the quoted PIA).
- **Above FRA (delayed retirement credits):** +8%/yr (= +2/3% per month) up to age 70. Age 70 vs FRA 67 ⇒ +24%. No further credits past 70.

After the actuarial scaling, the per-year benefit is multiplied by `(1 + ss_cola_rate) ** year_offset` (see `config.ss_cola_rate`).

The legacy single `start_age` still works for back-compat. Setting `start_age_a` and `start_age_b` separately unlocks the most common real-world strategy: high-PIA spouse delays to 70 (max delayed credits + bigger survivor benefit), low-PIA spouse claims at FRA or earlier.

When one spouse dies, the survivor switches to the **survivor benefit**: keeps the larger of the two scaled benefits, drops the smaller. Controlled by `Mortality.ss_survivor_keeps_higher` (default `True`, per SSA rules).

---

## `inputs.pension` — pension cash balance and start age

```json
"inputs": {
  "pension": {
    "balance_today":    50000,
    "monthly_at_nrd":   1500,
    "start_age":        65
  }
}
```

| Field | Type | Default | Means |
|---|---|---:|---|
| `balance_today` | float | `0.0` | Current cash-balance pension account ($, today) |
| `monthly_at_nrd` | float | `0.0` | Expected monthly annuity at the start age (today's $) |
| `start_age` | int | `65` | Normal Retirement Date — age Spouse A's pension annuity begins |

The simulator runs a **cash-balance projector**: grows `balance_today` forward to `start_age` (4.8% interest credit + an annual credit on Spouse A's salary), compares it to a reference projection, and scales the annuity proportionally. So:

- If `balance_today` grows *exactly* as expected → you receive `monthly_at_nrd × 12` each year for life.
- If it grows *faster* (better salary trajectory or you happen to lag retirement past `start_age` while still earning) → you receive a higher annuity scaled by `actual_balance / expected_balance`.
- If `balance_today` is `0` (the common case for households without a private pension), the annuity is also `0`.

Survivor handling: when Spouse A dies while Spouse B is alive, the surviving spouse continues to receive `mortality.pension_survivor_pct × monthly_at_nrd`. Common values:

- `0.50` — 50% joint-and-survivor (typical employer default)
- `0.75` — 75% J&S
- `1.00` — 100% J&S (survivor receives the full annuity)

---

## `inputs.spouse_*_employer_match_*` — employer 401(k) match

```json
"inputs": {
  "spouse_a_employer_match_rate":    0.50,
  "spouse_a_employer_match_max_pct": 0.06,
  "spouse_b_employer_match_rate":    1.00,
  "spouse_b_employer_match_max_pct": 0.06
}
```

| Field | Type | Default | Means |
|---|---|---:|---|
| `spouse_a_employer_match_rate` | float | `0.0` | Fraction of Spouse A's elective deferral the employer matches |
| `spouse_a_employer_match_max_pct` | float | `0.0` | Cap on the match as a fraction of Spouse A's salary |
| `spouse_b_employer_match_rate` | float | `0.0` | Same shape for Spouse B |
| `spouse_b_employer_match_max_pct` | float | `0.0` | Same shape for Spouse B |

Match formula:

```
match = salary * min(employee_deferral_pct, max_pct) * rate
```

**Common plan shapes mapped to these knobs:**

| Real-world plan | `rate` | `max_pct` | Resulting match (if employee defers ≥ max_pct) |
|---|---:|---:|---:|
| 50% on first 6% (very common) | `0.50` | `0.06` | 3% of salary |
| 100% on first 6% (common at large employers) | `1.00` | `0.06` | 6% of salary |
| 100% on first 3% + 50% on next 2% (Safe Harbor) | *combine* | — | 4% of salary — closest single-formula approximation: `1.00 / 0.04` |
| Non-elective 3% (Safe Harbor non-elective) | `1.00` | `0.03` | 3% — assumes employee defers ≥3% |
| No match | `0.0` | `0.0` | 0 |

**Mechanics worth knowing:**

- **Always pre-tax.** Even if the employee elects 100% Roth-401(k), the match always lands in the pre-tax bucket per IRS rules.
- **Doesn't count against the elective-deferral cap.** The match is governed by the §415(c) overall annual additions limit (~$70k 2026), which is rarely binding and the simulator does NOT model.
- **Stops at retirement.** No salary → no match.
- **Sized on the post-cap deferral.** If the employee blows through the $23,500 elective limit at, say, 30% of $300k salary, the simulator's effective deferral pct is `23,500 / 300,000 ≈ 7.8%`, and a "100% on first 6%" plan matches on 6% (not 30%).

Visible in the per-year DataFrame as `employer_match_a` and `employer_match_b`.

---

## `inputs.spouse_*_traditional_ira_contrib` / `_roth_ira_contrib` / `_backdoor_roth` — IRA contribution paths

```json
"inputs": {
  "spouse_a_traditional_ira_contrib": 0,
  "spouse_b_traditional_ira_contrib": 0,
  "spouse_a_roth_ira_contrib": 7000,
  "spouse_b_roth_ira_contrib": 7000,
  "spouse_a_backdoor_roth": false,
  "spouse_b_backdoor_roth": false
}
```

Three independent IRA contribution paths per spouse, all sharing one annual cap:

| Field | Vehicle | Tax effect |
|---|---|---|
| `spouse_*_traditional_ira_contrib` | Traditional IRA (deductible) | Reduces Box-1 wages by the contribution; balance lands in pretax. Modeled as universally deductible — IRS deductibility phase-out for 401(k)-covered filers is **not** modeled. If your AGI is past that phase-out, prefer the backdoor below. |
| `spouse_*_roth_ira_contrib` | Direct Roth IRA | Subject to MAGI phase-out (MFJ $236k–$246k 2026). Above the upper bound the contribution is silently zeroed. Below, full amount flows to Roth. From after-tax cash. |
| `spouse_*_backdoor_roth` (`bool`) | Backdoor Roth IRA | Non-deductible Traditional IRA contribution + same-year Roth conversion. Income-uncapped. **Pro-rata-aware**: if the spouse's pretax 401(k)/IRA balance is non-zero, a fraction of the conversion is taxable as ordinary income. With zero pretax, the backdoor is fully tax-free. |

**Cap stack:** the simulator enforces one annual IRA cap per spouse ($7k under 50, $8k 50+) across all three lines, allocated in priority order Traditional → direct Roth (after phase-out) → backdoor on the leftover room. So setting `traditional_ira_contrib=7000` AND `backdoor_roth=true` consumes the cap with the Traditional contribution; the backdoor toggle is silently a no-op.

**Eligibility:** requires earned income (own or spousal), so the contribution lines are gated on (alive AND either spouse working). Stops cleanly at retirement.

Visible in the per-year DataFrame as `ira_traditional_*`, `ira_roth_direct_*`, `ira_backdoor_*`, plus `ira_backdoor_taxable_conv` for the pro-rata-taxable portion.

---

## `inputs.spouse_*_mega_backdoor_enabled` / `_after_tax_401k_pct` — mega-backdoor Roth

```json
"inputs": {
  "spouse_a_mega_backdoor_enabled": true,
  "spouse_a_after_tax_401k_pct": 0.10,
  "spouse_b_mega_backdoor_enabled": false,
  "spouse_b_after_tax_401k_pct": 0.0
}
```

The **single most powerful Roth-savings lever** when available. Routes additional dollars into Roth via after-tax 401(k) contributions + same-day in-plan Roth conversion. Plan-dependent: most workplace 401(k)s do NOT allow this (you need both *after-tax* contributions AND *in-plan Roth conversion* or *in-service rollover*). Check plan documents before enabling.

| Field | Type | Default | Means |
|---|---|---:|---|
| `spouse_*_mega_backdoor_enabled` | bool | `false` | Master toggle. When `false`, `_after_tax_401k_pct` is ignored. |
| `spouse_*_after_tax_401k_pct` | float | `0.0` | Fraction of salary routed to after-tax 401(k). |

The simulator caps the after-tax dollars annually at the §415(c) overall plan-additions limit ($70k 2026) minus the employee's elective deferrals minus employer match — typically $25k–$45k of additional Roth headroom for a high earner who's also maxing the regular elective limit. After-tax contributions are NOT pre-tax (no Box-1 reduction) and NOT FICA-exempt; they come from take-home pay, so they reduce the household's after-tax cash surplus and add to the Roth balance with zero income-tax impact at conversion.

Visible in the per-year DataFrame as `mega_backdoor_a` / `mega_backdoor_b`.

---

## `inputs.contrib.hsa_family` — household HSA contribution target

```json
"inputs": {
  "contrib": {
    "hsa_family": 8550
  }
}
```

| Field | Type | Default | Means |
|---|---|---:|---|
| `hsa_family` | float | `8550.0` | Annual HSA family-coverage contribution target, today's dollars |

The simulator caps this at the IRS family-coverage limit + the 55+ catch-up if either spouse qualifies, and zeroes it once both spouses hit Medicare eligibility (65). HSA contributions are pre-tax (reduce Box 1 wages alongside traditional 401(k)) and **also FICA-exempt in real life via cafeteria plans**, though the simulator's FICA pass currently uses gross W-2 wages — a known small approximation.

**Spending the HSA, age-aware:** before age 65, the HSA balance is only spent tax-free against the long-term-care shock (`spending.ltc_shock`). After either spouse hits 65, the HSA *unlocks* as a fourth retirement bucket: the deficit cascade now draws from HSA before pretax (taxable → Roth → HSA → pretax), with HSA non-medical withdrawals taxed at ordinary income rate (no 20% penalty per IRS post-65). Because the HSA has no RMD obligation and dodges several state-tax frictions, it's strictly preferred over pretax for any deficit the cascade has to cover.

---

## Migration: `config.ss_start_age` / `config.pension_start_age`

These two fields **moved out of `config`** and now live on the nested Inputs blocks:

| Old location | New location |
|---|---|
| `config.ss_start_age` | `inputs.ss.start_age` |
| `config.pension_start_age` | `inputs.pension.start_age` |

Old scenario files fail fast with a precise migration message:

> ScenarioError: These age-gate fields moved out of `config`: `['ss_start_age']`. Update your scenario JSON: `ss_start_age -> inputs.ss.start_age`.

The same migration hint is surfaced for `--set` overrides (`--set config.ss_start_age=70` → "use `inputs.ss.start_age` instead"). This change matched the prior `spouse_*` migration: all "about me" timing knobs now cluster with the dollar amounts they gate.

---

## Output metric: `median_ruin_year_offset`

```python
# Returned by `MonteCarloResult.summary()`
"median_ruin_year_offset": <int or -1>
```

A Monte Carlo statistic that answers: *"In the failure paths, how soon does the household typically run out of money?"*

### Definitions

- **Ruin** = first simulation-year offset where total liquid assets (`pretax_balance + roth_balance + taxable_balance`) fall below that year's `spending_need`. If a path never hits this condition, it's tagged `-1` ("never ruined").
- **`median_ruin_year_offset`** = the median (50th percentile) ruin offset across **only the paths that did ruin**. If every path succeeded, the value is `-1` and the report renders it as a dash.

The number is a **simulation-year offset** (0-indexed), not a calendar year and not an age. To translate, add it to:

- Calendar year: `start_year + offset`
- Spouse A age: `spouse_a_age_start + offset`
- Spouse B age: `spouse_b_age_start + offset`

### Worked example with `example02.json`

With `spouse_a_age_start=54`, `spouse_b_age_start=53`, `horizon_age=85`, `start_year=2026`, a value of **`31`** means:

| Field | Value |
|---|---:|
| Year offset | **31** |
| Calendar year | `2026 + 31` = **2057** |
| Spouse A age | `54 + 31` = **85** |
| Spouse B age | `53 + 31` = **84** |

That coincides with `horizon_age=85` (the very last simulated year) — meaning *failures squeak through almost to horizon, then nick the floor on a bad final-year sequence.* That's a much milder failure mode than ruining at, say, age 75.

### Reading it together with `prob_success`

The two metrics are complementary — neither tells the full story alone:

| Metric | What it answers |
|---|---|
| `prob_success` | **How often** do you fail? (frequency) |
| `median_ruin_year_offset` | **When** do you fail, if you do? (timing) |
| `cvar_terminal_p10` | **How bad** is it conditional on a tail outcome? (severity) |

A 90% prob_success with `median_ruin_year_offset=31` is a **very different** plan than 90% prob_success with `median_ruin_year_offset=15`. Both have a 10% failure rate but the first means tiny terminal-year shortfalls; the second means real change-your-life ruins in your 60s/70s.

### Levers that move it

To **earlier** ruin (worse — useful for stress tests): lower `equity_mu`, higher `equity_sigma`, earlier `mortality.year_of_death_a`, bigger `inputs.annual_expenses`, smile + LTC shock, earlier `inputs.spouse_*_retire_age`.

If small bumps to those levers move `median_ruin_year_offset` from year 30+ down to year 15–20, you've found a real fragility worth hedging (longer cash buffer, deferred retirement, lower spending floor).

---

## CLI stress-test recipes

Three quick `--set` overrides you can run without editing the JSON file. Compare `prob_success`, `terminal_p5`, `cvar_terminal_p10`, and `median_ruin_year_offset` across the three:

```bash
# Pessimistic: cut equity premium 200 bps, raise vol
tax-optimizer --scenario scenarios/example02.json --monte-carlo 2000 \
  --set 'config.market={"kind":"lognormal","equity_mu":0.05,"equity_sigma":0.21,"bond_mu":0.04,"bond_sigma":0.06}'

# Optimistic: equity at historical
tax-optimizer --scenario scenarios/example02.json --monte-carlo 2000 \
  --set 'config.market={"kind":"lognormal","equity_mu":0.085,"equity_sigma":0.18,"bond_mu":0.045,"bond_sigma":0.07}'

# Bootstrap reality check (preserves fat tails)
tax-optimizer --scenario scenarios/example02.json --monte-carlo 2000 \
  --set 'config.market={"kind":"bootstrap","block_size":5}'
```

If `prob_success` stays ≥ 90% across **Conservative + bootstrap**, the plan is genuinely robust to assumption choice. If only **Base + Optimistic** pass, you're depending on a non-conservative market view to make the plan work — which is your **single biggest fragility lever**, almost always larger than any tax-optimization decision the optimizer is making.

The honest framing: **`equity_mu` is the parameter most likely to be wrong**, and a 200 bps swing in it dwarfs almost everything the optimizer can do for you. Stress-test it accordingly.

---

## Per-spouse / per-bucket override patterns

A handful of common `--set` patterns to bookmark:

```bash
# Per-spouse SS claim ages (high-PIA spouse delays to 70 for max
# delayed credits + bigger survivor benefit; low-PIA claims at FRA)
--set inputs.ss.start_age_a=70 \
--set inputs.ss.start_age_b=67 \
--set inputs.ss.fra_a=67 \
--set inputs.ss.fra_b=67

# Legacy single-knob form still works (both spouses claim at 68)
--set inputs.ss.start_age=68

# Add a real pension to a household that didn't have one
--set inputs.pension.balance_today=50000 \
--set inputs.pension.monthly_at_nrd=1500 \
--set inputs.pension.start_age=65

# Add an employer 401(k) match (50% on first 6% — most common plan)
--set inputs.spouse_a_employer_match_rate=0.50 \
--set inputs.spouse_a_employer_match_max_pct=0.06 \
--set inputs.spouse_b_employer_match_rate=0.50 \
--set inputs.spouse_b_employer_match_max_pct=0.06

# Stress test: freeze tax brackets for the entire horizon (no indexing)
--set config.bracket_indexing_rate=0.0

# Stress test: disable SS COLA (flat-nominal SS)
--set config.ss_cola_rate=0.0

# Shift to textbook asset-located portfolio
--set 'config.asset_location={"pretax_equity_pct":0.40,"roth_equity_pct":1.00,"taxable_equity_pct":0.80,"hsa_equity_pct":0.80}'

# Stress widow's penalty: Spouse A dies year 25
--set 'config.mortality={"year_of_death_a":25,"pension_survivor_pct":0.5}'

# Switch to TCJA sunset starting year 5
--set config.tax_regime=tcja \
--set config.regime_change_year_offset=5 \
--set config.regime_change_target=sunset
```

`--print-defaults` after any combination of `--scenario` / `--set` / high-level flags dumps the **fully-resolved** scenario JSON, which you can pipe into a file as a reproducible "frozen" snapshot of exactly what was simulated.
