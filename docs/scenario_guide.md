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
  - [The other two market kinds](#the-other-two-market-kinds)
- [`config.asset_location` — per-account equity / bond split](#configasset_location--per-account-equity--bond-split)
  - [The `uniform_equity_pct` shortcut](#the-uniform_equity_pct-shortcut)
  - [The per-bucket form](#the-per-bucket-form)
  - [Effect at simulation time](#effect-at-simulation-time)
  - [When to use which](#when-to-use-which)
- [`inputs.ss` — Social Security claim age and benefits](#inputsss--social-security-claim-age-and-benefits)
- [`inputs.pension` — pension cash balance and start age](#inputspension--pension-cash-balance-and-start-age)
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

- **`Config`** — the *simulation / strategy*. Macro assumptions (`nominal_growth_rate`, `inflation`, `wage_growth`), `rmd_start_age`, withdrawal / conversion strategy, and the modular blocks `market` / `asset_location` / `spending` / `mortality`.
- **`Inputs`** — the *household*. Spouse ages, retire ages, salaries, contribution rates, Roth-401(k) splits, starting balances, Social Security amounts (incl. claim age `inputs.ss.start_age`), pension data (incl. start age `inputs.pension.start_age`), expected expenses, etc.

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

### The other two market kinds

```json
/* Deterministic — single-path, no randomness. Used by deterministic runs. */
"market": { "kind": "deterministic", "equity": 0.06, "bond": 0.04 }

/* Bootstrap — block-resample real 5-year history blocks. Preserves fat tails. */
"market": { "kind": "bootstrap", "block_size": 5 }
```

Use `bootstrap` as a tail-risk reality check: it preserves real history's runs of bull/bear years and the 1929 / 1973 / 2008 fat tails that lognormal under-states.

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

## `inputs.ss` — Social Security claim age and benefits

```json
"inputs": {
  "ss": {
    "monthly_spouse_a": 4030,
    "monthly_spouse_b": 2208,
    "start_age": 70
  }
}
```

| Field | Type | Default | Means |
|---|---|---:|---|
| `monthly_spouse_a` | float | `2700.0` | Spouse A's monthly SS benefit (today's $) at the claim age |
| `monthly_spouse_b` | float | `2200.0` | Spouse B's monthly SS benefit (today's $) at the claim age |
| `start_age` | int | `70` | Claim age applied to **both** spouses (single knob) |

The simulator pays each spouse's monthly amount × 12 once that spouse reaches `start_age`. There is **no FRA actuarial adjustment** in the model, so enter the amount you expect to receive **AT** the claim age (e.g., the SSA Quick-Calculator PIA for that specific claim age).

When one spouse dies, the survivor switches to the **survivor benefit**: keeps the larger of the two monthly amounts, drops the smaller. Controlled by `Mortality.ss_survivor_keeps_higher` (default `True`, per SSA rules).

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
# Bump SS claim age (was a config field; now inputs.ss.start_age)
--set inputs.ss.start_age=68

# Add a real pension to a household that didn't have one
--set inputs.pension.balance_today=50000 \
--set inputs.pension.monthly_at_nrd=1500 \
--set inputs.pension.start_age=65

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
