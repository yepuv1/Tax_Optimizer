# Modeling Guide

How to use Tax Optimizer end-to-end: starting a scenario, the
configuration paths (Dash, JSON, CLI), every modeling area the
package covers, and which deeper doc to read for the math behind
each. This is the **task-oriented** doc — "I want to model X,
where do I start?" — and it points at the field-level
[`scenario_guide.md`](scenario_guide.md), the per-module
[`architecture.md`](architecture.md), and the topic-specific deep
dives ([`roth_conversion.md`](roth_conversion.md),
[`market_models.md`](market_models.md),
[`annuity_guide.md`](annuity_guide.md),
[`dashboard.md`](dashboard.md)) as you need them.

If you only want the field reference, jump straight to
[`scenario_guide.md`](scenario_guide.md). If you only want to run
the Dash app, [`dashboard.md`](dashboard.md) is the focused guide.

---

## Contents

- [Mental model: Config vs Inputs, plus six pluggable blocks](#mental-model-config-vs-inputs-plus-six-pluggable-blocks)
- [Quick start](#quick-start)
- [Configuration paths](#configuration-paths)
  - [Dash app: form-driven](#dash-app-form-driven)
  - [Scenario JSON: durable, sharable](#scenario-json-durable-sharable)
  - [CLI / Python API: scriptable](#cli--python-api-scriptable)
  - [Picking the right path](#picking-the-right-path)
- [Modeling area by area](#modeling-area-by-area)
  - [1. The household: ages, retirement, filing status](#1-the-household-ages-retirement-filing-status)
  - [2. Working-years income and contributions](#2-working-years-income-and-contributions)
  - [3. Starting balances and cost basis](#3-starting-balances-and-cost-basis)
  - [4. Federal tax regime](#4-federal-tax-regime)
  - [5. State tax](#5-state-tax)
  - [6. Social Security](#6-social-security)
  - [7. Pension and annuity contracts](#7-pension-and-annuity-contracts)
  - [8. HSA](#8-hsa)
  - [9. Healthcare costs (pre-65, Medicare, IRMAA, ACA)](#9-healthcare-costs-pre-65-medicare-irmaa-aca)
  - [10. Spending profile](#10-spending-profile)
  - [11. Mortality and the widow's penalty](#11-mortality-and-the-widows-penalty)
  - [12. Market model and asset location](#12-market-model-and-asset-location)
  - [13. Withdrawal strategy and the deficit cascade](#13-withdrawal-strategy-and-the-deficit-cascade)
  - [14. Roth conversions](#14-roth-conversions)
  - [15. RMDs](#15-rmds)
  - [16. Mega-backdoor Roth and backdoor Roth](#16-mega-backdoor-roth-and-backdoor-roth)
  - [17. The optimizer (when to use it)](#17-the-optimizer-when-to-use-it)
- [Run modes](#run-modes)
- [Reading the output](#reading-the-output)
- [Common workflows](#common-workflows)
- [Validation and the most useful error messages](#validation-and-the-most-useful-error-messages)
- [Where to go next](#where-to-go-next)

---

## Mental model: Config vs Inputs, plus six pluggable blocks

Two top-level dataclasses divide every knob this package exposes:

- **`Inputs`** — *household facts*. Spouse ages, salaries,
  contribution percentages, starting balances, Social Security
  PIAs, pension data, annuity contracts, health premiums,
  filing-status discriminator. "About me" data — mostly stable
  year-to-year.
- **`Config`** — *simulation strategy and assumptions*. Macro
  assumptions (inflation, growth, wage growth, COLA), withdrawal
  strategy, tax regime, state regime, market model, asset
  location, spending profile, mortality, healthcare cost knobs,
  optimizer scope. Things you'd reasonably change between
  scenarios on the same household.

Within `Config`, six pieces are **pluggable** — you can swap them
independently by name without touching simulator code:

| Block | What you choose | Built-ins |
|---|---|---|
| `tax_regime` | Federal bracket schedule | `tcja` / `pre_tcja_2017` / `sunset_2026` |
| `state_regime` | State income tax | `stateless` / `ca` / `ny` / `il` / `ma` |
| `market` | Return generator | `deterministic` / `lognormal` / `bootstrap` / `historical_sequence` (plus CMA presets) |
| `asset_location` | Equity vs bond per account | `uniform_equity_pct` shortcut, or per-bucket `pretax`/`roth`/`taxable`/`hsa` mix |
| `spending` | How spending evolves | `flat` / `smile` / manual phases + lumps + LTC |
| `mortality` | When each spouse dies | Per-spouse `year_of_death`, MFJ→single transition, pension survivor %, SS survivor logic |

This taxonomy matters because it tells you *where* to look when
something seems off. "Why is my marginal rate weird?" → look at
`tax_regime` first. "Why is my taxable balance growing so
slowly?" → look at `market` and `asset_location`. "Why does
spending blow up at 92?" → look at `spending` (LTC shock).

The full per-module breakdown lives in
[`docs/architecture.md`](architecture.md). The field-by-field
JSON reference lives in
[`docs/scenario_guide.md`](scenario_guide.md).

---

## Quick start

Three steps to a first scenario you can iterate on:

```bash
# 1. Install
pip install -e ".[notebook]"     # or just -e "." for the dashboard alone

# 2. Run with package defaults (a 54/53-year-old MFJ couple)
tax-optimizer                    # CLI: prints a styled report to the terminal
tax-optimizer-app                # Dash: opens http://127.0.0.1:8050

# 3. Copy the template as a starting point for a custom scenario
cp scenarios/template.json scenarios/my_plan.json
# Edit my_plan.json, then:
tax-optimizer --scenario scenarios/my_plan.json
```

`scenarios/template.json` is the canonical "every knob, with its
default value" reference — copy it, delete the fields you don't
care about, override the ones you do, and you're done.

Two ready-to-run example scenarios ship with the package:

- **`scenarios/example01.json`** — the canonical narrative
  scenario: 52/50 MFJ couple, $235k combined gross, retirement at
  65/67, TCJA→sunset regime change in year 5, lognormal market,
  smile spending with LTC shock, deterministic widow's-penalty
  test (Spouse A dies year 30).
- **`scenarios/example02.json`** — a deeper "stress test"
  scenario the [`retirement_planning_demo.ipynb`](../retirement_planning_demo.ipynb)
  notebook walks through.
- **`scenarios/example_single.json`** — single-filer household.
- **`scenarios/example_annuity_nonqualified.json`** —
  non-qualified annuity with §72(b) exclusion ratio.
- **`scenarios/example_pension_lump_sum.json`** — pension elected
  as a tax-free pretax-IRA rollover at NRD.

Use any of them as a starting point with `cp scenarios/X.json
scenarios/my_plan.json`.

---

## Configuration paths

Three ways to set knobs. They share the same underlying data
model — every JSON field, every Dash form input, and every
`--set` CLI override resolves to the same `Config` / `Inputs`
fields.

### Dash app: form-driven

Best for: exploring, tweaking interactively, demoing,
showing the action plan to a financial advisor or spouse.

```bash
tax-optimizer-app                # http://127.0.0.1:8050
tax-optimizer-app --port 9000    # different port
tax-optimizer-app --debug        # hot-reload, dev console
tax-optimizer-app-prod           # production WSGI (waitress); needs `[prod]` extra
```

Form structure:

- **Simple tier** (default view) — the ~30 most-used knobs
  grouped into "Ages & retirement", "Income", "Contributions",
  "Starting balances", "Pension", "Annuity", "Macro & horizon".
  Every label has a hover tooltip explaining what it does.
- **Advanced tier** — toggle "Show advanced" to expose another
  ~100 knobs (state tax regime, market parameters, asset-location
  per-bucket overrides, spending phases, IRMAA / ACA detail,
  mega-backdoor Roth, optimizer scope, etc.).
- **Run mode** — pick "single sim", "four strategies", or
  "four strategies + Monte Carlo" via the radio at the top.
- **Load / Save scenario** — drag-and-drop a JSON file in (or
  click "Save as JSON" to capture the current form state).

The right-hand pane shows six results tabs (Overview / Taxes /
Strategies / Monte Carlo / Year-by-year / Report). Detail in
[`docs/dashboard.md`](dashboard.md).

### Scenario JSON: durable, sharable

Best for: version control, sharing scenarios across a team,
reproducible experiments, automation.

```json
{
  "config": {
    "start_year": 2026,
    "horizon_age": 95,
    "tax_regime": "tcja",
    "withdrawal_strategy": "conventional",
    ...
  },
  "inputs": {
    "household_kind": "mfj",
    "spouse_a_age_start": 52,
    "spouse_b_age_start": 50,
    ...
  }
}
```

Any field is optional. The loader (`tax_optimizer.scenario.
load_scenario_file`) reads the dataclass surfaces and uses the
default for any missing key. Unknown keys raise `ScenarioError`
so typos don't silently no-op.

Field reference: [`docs/scenario_guide.md`](scenario_guide.md).
Annuity-specific: [`docs/annuity_guide.md`](annuity_guide.md).

### CLI / Python API: scriptable

Best for: automation, batch experiments, notebooks, one-off
overrides without editing a JSON file.

```bash
# Defaults, terminal report
tax-optimizer

# Custom scenario
tax-optimizer --scenario scenarios/my_plan.json

# Override individual fields without editing the JSON
tax-optimizer --scenario scenarios/my_plan.json \
    --set inputs.spouse_a_age_start=55 \
    --set config.tax_regime=sunset

# HTML report
tax-optimizer --scenario scenarios/my_plan.json --report report.html

# Monte Carlo with 1,000 paths
tax-optimizer --market lognormal --monte-carlo 1000

# Optimizer (differential evolution over Roth-401(k) % + conversion bracket)
tax-optimizer --optimize

# All flags
tax-optimizer --help
```

Or drive directly from Python:

```python
from tax_optimizer import Config, Inputs
from tax_optimizer.simulator import simulate
from tax_optimizer.report import build_action_report

cfg = Config(tax_regime="tcja", horizon_age=95)
inp = Inputs(spouse_a_age_start=52, spouse_b_age_start=50)

df = simulate(cfg, inp)              # ~80-column per-year DataFrame
report = build_action_report(cfg, inp, df)
print(report.markdown)               # or .html, or .summary
```

### Picking the right path

| You want to... | Use |
|---|---|
| Explore a what-if interactively | Dash |
| Show the action plan to a non-technical reader | Dash → Report tab → "Download HTML" |
| Pin a scenario for next quarter's review | JSON in version control |
| Compare 50 variants of one parameter | Python script + `--set` or programmatic `Config` / `Inputs` |
| Embed in a notebook with custom plots | Python API + `simulate()` |
| Find the best Roth conversion strategy | `tax-optimizer --optimize` |
| Stress-test against sequence-of-returns risk | `--market lognormal --monte-carlo 1000` |

---

## Modeling area by area

Each subsection below covers one "thing you'd want to model",
the relevant knobs, recommended defaults, and where the math
lives if you want to go deeper.

### 1. The household: ages, retirement, filing status

| Knob | What it sets |
|---|---|
| `inputs.household_kind` | `"mfj"` (default, two spouses) or `"single"` (one filer). Drives federal brackets, FICA Additional-Medicare threshold, IRMAA tier table, SS calculation, HSA family-vs-self cap. |
| `inputs.spouse_a_age_start` / `spouse_b_age_start` | Ages at year 0. For single households, `spouse_b_*` fields are silently ignored (and disabled in the Dash form). |
| `inputs.spouse_a_retire_age` / `spouse_b_retire_age` | When each spouse stops earning W-2 wages. Salaries → 0 from this age on. |
| `config.start_year` | Calendar year of year 0. Anchors tax brackets and IRS limits. |
| `config.horizon_age` | Age of the **older** spouse at the end of the simulation. |

Recommended starting points:

- Both spouses still working, retiring early-to-mid 60s →
  `retire_age = 65–67`. SS claim age ≠ retirement age (see §6).
- Already retired → set `retire_age` to your current age (or 0 to
  guarantee no W-2 wages from year 0). Set `spouse_*_gross` to 0
  in `inputs.income`.

> **Single filer?** Set `household_kind: "single"`. Set
> `spouse_b_gross`, `spouse_b_*_pretax_*`, `monthly_spouse_b`
> etc. to 0 in their respective sub-blocks. The Dash form
> grays out spouse-B fields automatically when "Single" is
> selected.

### 2. Working-years income and contributions

```json
"inputs": {
  "income": {
    "spouse_a_gross": 140000,
    "spouse_b_gross": 95000,
    "spouse_a_bonus": 15000,
    "interest": 0,
    "capital_gains": 0,
    "dividends": 0
  },
  "spouse_a_total_contrib_pct": 0.10,    // 10% of pay → 401(k)
  "spouse_b_total_contrib_pct": 0.08,
  "spouse_a_roth_401k_pct": 0.0,         // 0% Roth, 100% pretax (default)
  "spouse_b_roth_401k_pct": 0.0,
  "spouse_a_employer_match_rate": 0.50,  // 50% on first 6% of pay
  "spouse_a_employer_match_max_pct": 0.06,
  "spouse_a_roth_ira_contrib": 7000,     // direct Roth IRA contribution
  "spouse_a_traditional_ira_contrib": 0,
  "spouse_a_backdoor_roth": false,
  "spouse_a_mega_backdoor_enabled": false,
  "spouse_a_after_tax_401k_pct": 0.0
}
```

Mechanics:

- `total_contrib_pct` is the employee's combined Traditional +
  Roth 401(k) deferral as a fraction of `gross + bonus`.
- `roth_401k_pct` is what fraction of THAT goes to the Roth
  bucket. `0.0` = 100% pretax, `1.0` = 100% Roth. The simulator
  enforces the IRS elective-deferral cap (`$23,500` in 2025, with
  age-50+ catch-ups baked into `tax_optimizer.limits`).
- `employer_match_*` always lands in the **pretax** bucket (IRS
  rule), regardless of the employee's Roth-vs-Traditional
  election. It does NOT count against the elective-deferral cap.
- Direct Roth IRA contributions go through `_roth_ira_contrib`;
  set `_backdoor_roth=true` to model the non-deductible
  Traditional → Roth maneuver (with proper pro-rata rule). See
  §16.
- Mega-backdoor Roth (after-tax 401(k) → Roth) goes through
  `_mega_backdoor_enabled` + `_after_tax_401k_pct`. See §16.
- `inputs.income.interest` / `capital_gains` / `dividends` are
  ADDITIONAL income lines (separate from portfolio yield, which
  comes from the market model). Use these for non-portfolio
  cash income (rental income, side gigs, taxable bond interest
  outside the modeled accounts).

### 3. Starting balances and cost basis

```json
"inputs": {
  "starting": {
    "spouse_a_pretax_401k": 400000,
    "spouse_b_pretax_401k": 250000,
    "spouse_a_pretax_ira": 0,
    "spouse_b_pretax_ira": 0,
    "spouse_a_roth_ira": 60000,
    "spouse_b_roth_ira": 0,
    "hsa": 25000,
    "taxable_brokerage": 120000,
    "pension_balance": 0
  }
}
```

Two important nuances:

1. **Pretax 401(k) vs pretax IRA matter separately.** The
   backdoor Roth pro-rata rule (IRC §408(d)(2)) aggregates only
   IRA balances, not 401(k). If you have $400k in a former-employer
   401(k), put it in `pretax_401k`, not `pretax_ira`, or the
   simulator will overstate your backdoor's taxable conversion.
2. **Cost basis on the taxable brokerage.** Set via
   `config.cap_gains_basis_fraction` (default 0.5). It's the
   fraction of `starting.taxable_brokerage` that's already
   basis. The remaining portion is unrealized gain. Withdrawals
   from taxable apply this fraction at year 0; later years use a
   running `state.cumulative_basis` that updates as gains realize
   and new contributions add basis.

### 4. Federal tax regime

```json
"config": {
  "tax_regime": "tcja",
  "regime_change_year_offset": 5,
  "regime_change_target": "sunset"
}
```

Three built-in regimes:

| Regime | When | What's different |
|---|---|---|
| `"tcja"` *(default)* | Current law (post-2017 TCJA) | $30k MFJ / $15k single std deduction (2025), 10/12/22/24/32/35/37 brackets. |
| `"pre_tcja_2017"` | Pre-TCJA (2017 brackets) | $12.7k MFJ / $6.35k single std deduction, 10/15/25/28/33/35/39.6 brackets, personal exemption add-back, AMT broader. |
| `"sunset_2026"` | TCJA sunsets after 2025 | Reverts to pre-TCJA but with bracket inflation indexing. |

The `regime_change_*` pair lets you stress-test the sunset:
schedule a swap from one regime to another mid-simulation. The
example01 scenario does exactly this — `tcja` for years 0-4,
`sunset` from year 5 onward.

The simulator also handles:

- **Age-65+ standard deduction add-on** (IRC §63(f)) per
  eligible spouse, plus the 2025 OBBBA senior bonus during its
  2025–2028 window.
- **AMT** (IRC §55) for regimes that have it — modeled as a
  parallel computation and `tax = max(regular, tentative_minimum)`.
- **NIIT** (3.8% on investment income above MAGI thresholds).
- **Additional Medicare Tax** (0.9% on wages > $200k single /
  $250k MFJ).

### 5. State tax

```json
"config": {
  "state_regime": "ca",
  "state_regime_change_year_offset": null,
  "state_regime_change_target": null
}
```

Built-in regimes:

| Regime | Highlights |
|---|---|
| `"stateless"` *(default)* | Zero state tax (FL/TX/WA/NV residents). |
| `"ca"` | Top marginal 13.3%, no SS tax, HSA add-back, mental-health surtax. |
| `"ny"` | Up to 10.9%, $20k/filer retirement-income exclusion at 59½+. |
| `"il"` | Flat 4.95%, full retirement-income exclusion (pension, IRA, Roth conv, SS). |
| `"ma"` | Flat 5%, SS exempt, IRA/pension taxed normally. |

Use `state_regime_change_*` for residency-relocation scenarios
(e.g. "I'm in CA today, moving to FL in year 8 at retirement").

Annuity income is treated identically to pension income for
state tax — the simulator rolls `annuity_taxable` into the
`pension` kwarg passed to `state_tax`, so retirement-income
exclusions apply.

### 6. Social Security

```json
"inputs": {
  "ss": {
    "monthly_spouse_a": 3100,            // PIA at FRA in today's $
    "monthly_spouse_b": 2500,
    "start_age": 67,                      // legacy fallback
    "start_age_a": 70,                    // Spouse A claims at 70 (max DRC)
    "start_age_b": 67,                    // Spouse B claims at FRA
    "fra_a": 67,
    "fra_b": 67
  }
}
```

Mechanics:

- `monthly_*` is the **PIA at FRA in today's dollars** — the
  standard SSA estimate. The simulator applies (a) an actuarial
  scaling factor for early/delayed claim (62 = -25–30%, 70 =
  +24–32%), and (b) a COLA inflator (`config.ss_cola_rate`,
  defaults to `cfg.inflation`).
- Per-spouse claim ages override the legacy single `start_age`.
- Survivor benefits (SSA): a widow(er) can claim on the
  deceased's record at age 60. Modeled in `simulator.py` using
  `ss_survivor_keeps_higher` from the mortality block.

For a 25/30-year horizon, the **biggest planning lever in the
whole package** is often Spouse A's SS claim age. Use
`config.optimize_ss_claim_age = true` to add it to the optimizer
search space — see §17.

### 7. Pension and annuity contracts

Two separate buckets, each with its own `lump_sum_mode` knob:

```json
"inputs": {
  "pension": {
    "balance_today": 350000,
    "monthly_at_nrd": 1700,
    "start_age": 65,
    "lump_sum_mode": "rollover_pretax"    // or "cash" or "none"
  },
  "annuity": {
    "balance_today": 200000,
    "monthly_at_start": 1000,
    "start_age": 65,
    "tax_kind": "non_qualified",          // or "qualified"
    "cost_basis": 50000,
    "expected_payout_years": 20,
    "lump_sum_mode": "none"
  }
}
```

Pension is a cash-balance accrual model (BP-RAP-calibrated by
default). Annuity is a passive contract you already own.

Full guide with all four modes, the §72(b) exclusion ratio, and
the IRC §72(t)/§72(q) 10% surtax rules:
[`docs/annuity_guide.md`](annuity_guide.md).

### 8. HSA

```json
"inputs": {
  "starting": { "hsa": 25000 },
  "contrib": { "hsa_family": 8550 }
}
```

The HSA is the simulator's most tax-efficient bucket. While both
spouses are eligible (under 65 and on an HDHP), the simulator
contributes up to `contrib.hsa_family` per year (capped at the
IRS family limit, or self-only for single households). Growth is
tax-free. Distributions:

- Pre-65 → must be qualified medical (no spending model for this;
  the HSA is treated as a pure savings vehicle until 65).
- Post-65 → fully accessible for general spending; ordinary tax
  on non-medical use.

In the deficit cascade (§13), HSA acts as a tax-deferred bucket
only after either spouse hits 65.

### 9. Healthcare costs (pre-65, Medicare, IRMAA, ACA)

Three relevant blocks:

```json
"config": {
  "medicare_base_b_d_premium": 2500,           // Part B + Part D base, today's $
  "health_pre65_today": 0,                      // pre-Medicare healthcare cost
  "irmaa_lookback_years": 2,                    // standard IRS 2-year lookback

  "aca_enabled": false,
  "aca_benchmark_premium_per_adult": 14000,
  "aca_max_contrib_pct": 0.085
},
"inputs": {
  "health_premiums": {
    "spouse_a_medical": 6500,
    "spouse_a_dental": 450,
    "spouse_a_vision": 180,
    ...
  }
}
```

Mechanics:

- Pre-65 health costs are paid out of `cash_inflow` directly.
- At 65, Medicare kicks in: base premium per spouse +
  IRMAA surcharge keyed off MAGI from `irmaa_lookback_years` ago.
- IRMAA tiers are filing-status-aware (single thresholds are
  half of MFJ).
- ACA: when `aca_enabled=true` AND a pre-Medicare adult exists,
  the household's premium contribution caps at
  `aca_max_contrib_pct` of MAGI; the credit equals
  `max(0, benchmark - cap)`. Treated as cash (offsets cash
  outflow), not a federal tax line item — most filers take it as
  advance APTC paid to the insurer.

> **Roth conversion + IRMAA interaction.** A Roth conversion
> spikes AGI, which spikes IRMAA two years later. The simulator
> sees this automatically via the lookback. If you're converting
> aggressively in your first Medicare year, expect the IRMAA
> column to jump in year +2.

### 10. Spending profile

```json
"config": {
  "spending": {
    "kind": "smile",
    "base_spending": 95000,
    "inflation": 0.025,
    "ltc_years": 4,
    "ltc_annual_today": 100000
  }
}
```

Three built-in profiles via the `kind` discriminator:

- **`flat`** — constant real spending across all years. Use as
  the conservative baseline.
- **`smile`** — Blanchett-style retirement smile: ~95% of base
  in early-retirement go-go years, drops ~5% real/yr through
  slow-go (mid-70s), then rises ~3% real/yr in no-go (LTC) years.
  Empirically grounded in retiree-spending studies.
- **Manual** — supply explicit `phases` (list of dicts with
  `start_age`, `end_age`, `multiplier`), `lumps` (list of
  `{year_offset, amount}`), and an `ltc` block.

The smile + LTC shock is the most realistic default for a
deterministic plan; flat is the most conservative for stress-
testing.

There's also a **legacy** scalar fallback,
`config.annual_expenses_today`, used when `spending` is `null`.
Don't mix the two.

### 11. Mortality and the widow's penalty

```json
"config": {
  "mortality": {
    "year_of_death_a": 30,                  // Spouse A dies year 30
    "year_of_death_b": null,                // Spouse B lives full horizon
    "pension_survivor_pct": 0.5,
    "ss_survivor_keeps_higher": true
  }
}
```

What the simulator does on a death event:

- Filing status flips MFJ → single from the year of death.
- Spousal IRA rollover: deceased's pretax balance moves to the
  survivor's pretax bucket (preserves tax deferral).
- Step-up in basis on `state.taxable` if
  `config.stepup_at_first_death=true` (community-property
  default; non-community-property states get a 50% step-up
  modeled separately).
- Pension annuity scales by `pension_survivor_pct`.
- SS survivor benefit applies if `ss_survivor_keeps_higher=true`.
- The "widow's penalty" — single brackets compress, IRMAA
  thresholds halve — kicks in automatically.

The CLI shorthand `--widow N` is equivalent to setting
`year_of_death_a = N`.

### 12. Market model and asset location

Two independent choices: (a) what return distribution to draw from,
and (b) how stocks/bonds are split across each account.

#### Market model

```json
"config": {
  "market": {
    "kind": "lognormal",
    "equity_mu": 0.07,
    "equity_sigma": 0.18,
    "bond_mu": 0.04,
    "bond_sigma": 0.06,
    "equity_bond_corr": 0.10
  }
}
```

Built-ins:

| `kind` | When to use |
|---|---|
| `"deterministic"` | Single point estimate. Good for explainable plans, "what does my advisor's 6% assumption produce?" Fast. |
| `"lognormal"` | Stochastic Monte Carlo. The default for `--monte-carlo`. Calibratable to historical or forward-looking CMA expectations. |
| `"bootstrap"` | Resamples actual historical years (1928–present). Captures fat tails and skew that lognormal misses. |
| `"historical_sequence"` | Consecutive 30-year windows from history. Strongest sequence-of-returns risk modeling. |
| CMA presets (e.g. `vanguard_2025`, `jpm_ltcma_2025`) | Forward-looking institutional capital-market assumptions. Use for "expected returns are lower in the next decade" scenarios. |

Pick lognormal for default Monte Carlo, bootstrap or
historical_sequence for the **cross-model robustness check**
(`--cross-model`). Detail: [`docs/market_models.md`](market_models.md).

#### Asset location

```json
"config": {
  "asset_location": {
    "uniform_equity_pct": 0.7
  }
}
```

The shortcut sets every account to 70/30 equity/bond. For a more
realistic asset-location strategy (bonds in pretax, stocks in
Roth and taxable for step-up benefits):

```json
"asset_location": {
  "pretax":  { "equity_pct": 0.30 },
  "roth":    { "equity_pct": 0.95 },
  "taxable": { "equity_pct": 0.85 },
  "hsa":     { "equity_pct": 0.95 }
}
```

Per-account growth applies the appropriate equity/bond mix to
that bucket's balance.

### 13. Withdrawal strategy and the deficit cascade

```json
"config": {
  "withdrawal_strategy": "conventional"
}
```

Three strategies:

| Strategy | What it does |
|---|---|
| `"conventional"` *(default)* | Drain taxable → Roth → HSA-after-65 → pretax (in that order). Simple and tax-efficient for moderate-spend retirees. |
| `"proportional"` | Withdraw from each bucket in proportion to its balance. Smooths long-run tax drag. |
| `"bracket_fill"` | Blends withdrawals + Roth conversions to fill brackets up to `bracket_fill_target` each year. The most aggressive Roth conversion strategy. |

Regardless of strategy, when there's a same-year deficit (lump
event, conversion-tax shortfall, gap-year overrun), the simulator
runs the **deficit cascade**:

```
taxable → Roth → HSA-after-65 → pretax
```

Pre-tax draws are grossed up via 50-iteration bisection
(`_solve_pretax_for_net`) to find the gross amount that produces
the target net-of-tax. Same for taxable (which has cap-gains tax
on the gain portion).

### 14. Roth conversions

Two ways to size a conversion:

```json
"config": {
  // Method A: fixed dollar amount per year
  "roth_conversion_amount": 50000,

  // Method B: fill to a target bracket
  "roth_conversion_target_bracket": 0.22,
  "withdrawal_strategy": "bracket_fill",     // implies bracket-fill
  "bracket_fill_target": 0.22,

  // Liquidity guard (TC-tier-C)
  "cap_conversion_by_liquidity": true,
  "conversion_taxable_use_ratio": 0.5,
  "protect_roth_in_conversion_years": true
}
```

The liquidity guard prevents the simulator from running a
conversion it can't pay tax on without raiding Roth (which would
defeat the purpose). The `conversion_taxable_use_ratio` knob
controls how much of taxable can be used for tax-paying capacity
(default 50% — more aggressive = larger conversions, less
emergency reserve).

Detail with formulas, liquidity-guard math, audit recipes:
[`docs/roth_conversion.md`](roth_conversion.md).

### 15. RMDs

```json
"config": { "rmd_start_age": 75 }
```

Per IRS SECURE 2.0:
- Born 1951–1959 → RMD at 73.
- Born 1960+ → RMD at 75.

The simulator computes per-spouse RMDs at the start of each year
(IRS Uniform Lifetime Table) and applies them as ordinary income
**before** any optional Roth conversion. This ordering matters —
the conversion sizer sees post-RMD taxable income line as its
floor.

### 16. Mega-backdoor Roth and backdoor Roth

```json
"inputs": {
  "spouse_a_backdoor_roth": false,           // simple backdoor
  "spouse_a_mega_backdoor_enabled": false,   // mega backdoor
  "spouse_a_after_tax_401k_pct": 0.0
}
```

- **Backdoor Roth** (`_backdoor_roth=true`): contributes the full
  IRA cap as a non-deductible Traditional IRA, then immediately
  converts to Roth. The simulator models the **pro-rata rule**
  (IRC §408(d)(2)) — if pretax-IRA balance is non-zero, the
  conversion is taxable proportionally. Income-uncapped (the
  whole point of the backdoor).
- **Mega-backdoor Roth** (`_mega_backdoor_enabled=true` +
  `_after_tax_401k_pct`): contributes after-tax 401(k) up to the
  §415(c) overall cap (`$70k` in 2025), then in-plan converts to
  Roth. Requires a plan that allows after-tax + in-plan
  conversion. Subject to the same §415(c) limit as employer
  match.

If your plan permits it, mega-backdoor is one of the highest-
leverage knobs in the package. Optimizer can search over it
(`config.optimize_mega_backdoor=true`).

### 17. The optimizer (when to use it)

```bash
tax-optimizer --optimize
tax-optimizer --optimize --mc-objective cvar      # robust under MC
tax-optimizer --optimize --mc-objective prob      # max P(success)
```

`tax_optimizer.optimizer.optimize_household` runs SciPy's
`differential_evolution` over a decision vector that always
includes:

- Spouse A Roth-401(k) %
- Spouse B Roth-401(k) %
- Conversion bracket index (which bracket cap to fill to)

Optionally, with the right `Config` flags:

- Mega-backdoor % per spouse (`optimize_mega_backdoor=true`)
- SS claim ages per spouse (`optimize_ss_claim_age=true`)

Three objectives:

| `--mc-objective` | What it maximizes |
|---|---|
| `terminal` *(default)* | Median terminal liquid net worth (deterministic if no MC, median if MC). |
| `cvar` | CVaR(10%) terminal NW — the average of the worst 10% of paths. Robust against bad-luck sequences. |
| `prob` | Probability of horizon success (no ruin). |

Use the optimizer when (a) your scenario has 3+ correlated knobs
you can't intuit, or (b) you want a defensible "best Roth
conversion plan" recommendation. For single-knob sensitivity
(e.g. "what's the right Roth-401(k) %?") a simpler sweep is
faster and more interpretable.

---

## Run modes

The CLI and Dash both expose four orthogonal "what kind of run":

| Mode | CLI flag | Dash radio | What you get |
|---|---|---|---|
| **Single sim** | (default) | "Single sim" | One deterministic year-by-year DataFrame from your scenario. |
| **Four strategies** | `--strategies` | "Four strategies" | Same scenario evaluated under four canonical strategies side-by-side: `S0_baseline` (current), `S1_all_roth_401k` (100% Roth deferral), `S2_bracket_fill_22` (fill to 22% with Roth conversions), `S3_optimized` (differential evolution). |
| **Monte Carlo** | `--monte-carlo N` | "Four + Monte Carlo" | N independent paths through `cfg.market`. Returns a `MonteCarloResult` with `prob_success()`, `cvar_terminal()`, percentiles. Adds a "Risk picture" section to the action report. |
| **Optimizer** | `--optimize` | (in "Four strategies" — `S3_optimized`) | Differential evolution to tune Roth-401(k) % + conversion bracket (and optionally mega-backdoor + SS claim ages). |

You can stack them: `--strategies --monte-carlo 1000 --optimize`
runs the four strategies AND a Monte Carlo on each AND optimizes
S3 with MC objective.

The **cross-model robustness check** (`--cross-model`) re-runs
the same plan under alternate market models and adds a §4
sub-section to the report comparing terminal NW, P(success), and
CVaR across them. Default companions are `bootstrap` +
`historical_sequence`; override with
`--cross-model bootstrap,vanguard_2025,jpm_ltcma_2025`.

---

## Reading the output

Two output surfaces:

### The year-by-year DataFrame (`simulate()` return)

`simulate(cfg, inp)` returns a pandas DataFrame with ~80 columns
per year. Key column families:

| Family | Examples | What |
|---|---|---|
| **Identity** | `year`, `spouse_a_age`, `spouse_b_age`, `alive_a`, `alive_b`, `filing_status`, `regime` | Per-year context |
| **Income** | `wages`, `pension`, `pension_lump_sum`, `annuity_taxable`, `annuity_tax_free`, `ssn`, `interest_income`, `qualified_dividends`, `ordinary_dividends` | All income lines |
| **Tax** | `federal_tax`, `state_tax`, `marginal`, `agi`, `taxable_income`, `irmaa`, `irmaa_tier`, `early_distribution_penalty`, `niit`, `amt` | Tax lines and AGI/MAGI |
| **Withdrawals** | `pretax_withdrawal`, `pretax_withdrawal_a`/`_b`, `roth_withdrawal`, `taxable_withdrawal` | What came out of where |
| **Conversions** | `roth_conversion`, `roth_conversion_a`/`_b`, `roth_conv_bracket_target`, `roth_conv_capped_by_liquidity`, `roth_conv_tax_capacity` | Roth conversion mechanics |
| **RMDs** | `rmd`, `rmd_a`, `rmd_b` | Per-spouse RMDs |
| **Contributions** | `elective_deferral_a`/`_b`, `ira_traditional_a`/`_b`, `ira_roth_direct_a`/`_b`, `ira_backdoor_a`/`_b`, `mega_backdoor_a`/`_b`, `employer_match_a`/`_b` | Contribution detail |
| **FICA** | `fica`, `fica_oasdi`, `fica_medicare`, `fica_additional_medicare`, `state_sdi` | Payroll tax |
| **Healthcare** | `medicare_base_premium`, `health_pre65`, `aca_benchmark_premium`, `aca_apt_credit` | Health costs |
| **Spending** | `spending_need`, `unfunded` | Spending and shortfalls |
| **Balances** | `pretax_balance`, `pretax_a_balance`, `pretax_b_balance`, `roth_balance`, `taxable_balance`, `hsa_balance`, `pension_balance`, `annuity_balance`, `cumulative_basis` | End-of-year balances |
| **Mortality** | `spousal_rollover` | First-death event rows |

Use this DataFrame for any custom plot, sanity-check, or audit.
The **Year-by-year tab in the Dash app** is a sortable / filterable
view of exactly this DataFrame.

### The action plan report

`build_action_report(cfg, inp, df)` produces a 9-section
narrative that the CLI / Dash render in three formats (terminal
via Rich, HTML, or PDF):

1. **TL;DR** — one-paragraph headline.
2. **Snapshot** — household, balances, contributions today.
3. **Levers** — which knobs to pull, ranked.
4. **Outcomes** — terminal NW, real CAGR, success probability.
5. **Risk picture** — Monte Carlo distribution and CVaR (only with `--monte-carlo`).
6. **Tornado** — sensitivity to each knob, ranked.
7. **Year-1 actions** — concrete this-year to-dos.
8. **Year-by-year timeline** — narrative across the horizon.
9. **Caveats** — what's NOT modeled.

Get the report from:

```bash
tax-optimizer --report report.html              # styled HTML
tax-optimizer --report report.pdf               # PDF (needs [pdf] extra)
tax-optimizer --report-archive                  # timestamped HTML under reports/
tax-optimizer | glow -                          # raw markdown to a renderer
```

Or directly in Python:

```python
from tax_optimizer.report import build_action_report

report = build_action_report(cfg, inp, df)
print(report.markdown)         # str
print(report.html)             # str
print(report.summary)          # short text summary (the old --no-report output)
```

---

## Common workflows

### "I want to see my plan today, no fuss"

```bash
tax-optimizer-app
# → http://127.0.0.1:8050, click Run, read the Report tab
```

### "I want to compare 'all pretax 401(k)' vs 'all Roth 401(k)'"

In Dash, pick run mode "Four strategies". The S0_baseline
column shows your current allocation; S1_all_roth_401k shows
100% Roth deferral with the same scenario. The Strategies tab
puts them side-by-side.

Or:

```bash
tax-optimizer --strategies --scenario scenarios/my_plan.json
```

### "I want to know how risky my plan is to bad markets"

```bash
tax-optimizer --market lognormal --monte-carlo 1000 --scenario scenarios/my_plan.json
```

The "Risk picture" section in the report shows P(success), the
worst 10% (CVaR) terminal NW, and the median ruin year.

### "I want the most robust plan I can find under sequence-of-returns risk"

```bash
tax-optimizer --optimize \
    --market lognormal --monte-carlo 500 \
    --mc-objective cvar \
    --scenario scenarios/my_plan.json
```

The optimizer then maximizes the worst 10% of outcomes — much
more conservative than the default "median terminal NW" objective.

### "I want a printed action plan for my financial advisor"

```bash
tax-optimizer --scenario scenarios/my_plan.json --report plan.pdf
```

Or in Dash: Report tab → "Download HTML" / "Download PDF".

### "I want to model the TCJA sunset"

```bash
tax-optimizer --regime sunset --regime-change-year 5
```

Or in JSON:

```json
{
  "config": {
    "tax_regime": "tcja",
    "regime_change_year_offset": 5,
    "regime_change_target": "sunset"
  }
}
```

### "I want to test the widow's penalty on my plan"

```bash
tax-optimizer --widow 25 --scenario scenarios/my_plan.json
```

(Spouse A dies year 25. Equivalent to `mortality.year_of_death_a = 25`.)

### "I want to model relocating from CA to FL at retirement"

```json
{
  "config": {
    "state_regime": "ca",
    "state_regime_change_year_offset": 13,
    "state_regime_change_target": "stateless"
  }
}
```

(Year 13 = first retirement year for example01's 52-year-old.)

### "I want to see the impact of taking my pension as a lump sum"

Configure two scenarios, one with `pension.lump_sum_mode="none"`
(monthly), one with `pension.lump_sum_mode="rollover_pretax"`
(rollover) or `"cash"` (taxable distribution). The Strategies tab
in Dash, or the four-strategy CLI mode, makes the comparison
trivial. Detail: [`docs/annuity_guide.md`](annuity_guide.md).

### "I want to find my optimal SS claim age"

```bash
tax-optimizer --optimize --scenario scenarios/my_plan.json
```

with `config.optimize_ss_claim_age = true`. The optimizer adds
both spouses' claim ages to its decision vector and reports the
best.

---

## Validation and the most useful error messages

Both `Inputs(...)` and `Config(...)` validate aggressively at
construction time, plus the scenario loader rejects unknown JSON
keys. Common errors:

| Error | Cause | Fix |
|---|---|---|
| `ScenarioError: Unknown top-level scenario keys: ['_comment']` | Unrecognized top-level field. | Remove the field, or nest it under `config` / `inputs`. |
| `ScenarioError: Unknown config field: 'foo'` | Typo in a `config` key, or a deprecated field. | Check `scenarios/template.json` for the right name. |
| `ValueError: Inputs.household_kind must be 'mfj' or 'single'` | Typo. | One of those two literal strings. |
| `ValueError: Non-qualified annuity cannot use lump_sum_mode='rollover_pretax'` | IRC prohibition. | Use `"cash"` or set `tax_kind="qualified"`. |
| `ValueError: inputs.annuity.expected_payout_years must be > 0` | Set to 0 or negative. | Positive integer (default 20). |
| Migration error: `config.ss_start_age moved to inputs.ss.start_age` | Old field name. | Rename per the message. |
| `dash_app: Address already in use` | Port 8050 occupied. | `tax-optimizer-app --port 9000`. |
| `weasyprint not found` (with `--report PATH.pdf`) | PDF extra not installed. | `pip install -e ".[pdf]"` + system libs (pango/cairo). |

The key point: **typos don't silently no-op**. If the loader
accepts your scenario, the field names are right.

---

## Where to go next

By goal:

- **"I want the field reference for every JSON knob"** →
  [`docs/scenario_guide.md`](scenario_guide.md).
- **"I want to use the Dash app fully"** →
  [`docs/dashboard.md`](dashboard.md).
- **"I want to understand Roth conversion sizing and the
  liquidity guard math"** →
  [`docs/roth_conversion.md`](roth_conversion.md).
- **"I want to choose the right market model for my Monte
  Carlo"** → [`docs/market_models.md`](market_models.md).
- **"I want to model an annuity or take a lump-sum on my
  pension"** → [`docs/annuity_guide.md`](annuity_guide.md).
- **"I want to extend the package — add a new tax regime, market
  model, or spending profile"** →
  [`docs/architecture.md`](architecture.md).
- **"I want to see a worked narrative example"** → the
  [`retirement_planning_demo.ipynb`](../retirement_planning_demo.ipynb)
  notebook walks through `scenarios/example02.json` end-to-end.
- **"I want to see what changed in this release"** →
  [`CHANGELOG.md`](../CHANGELOG.md).

Feedback / bugs / new feature requests:
[github.com/yepuv1/Tax_Optimizer](https://github.com/yepuv1/Tax_Optimizer)/issues.
