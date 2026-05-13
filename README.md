# Retirement Tax Optimizer

A Python package + notebook + CLI that jointly optimizes retirement-tax decisions for a married-filing-jointly couple, with realistic stochastic and demographic risk modeling:

1. **Pre-retirement** allocation between Traditional 401(k) and Roth 401(k) for each spouse.
2. **In-retirement** withdrawal sequencing across taxable / pretax / Roth buckets.
3. **Roth-conversion** sizing during the retirement-to-RMD gap years.

The deterministic engine models federal brackets, LTCG, NIIT, IRMAA, Social-Security provisional-income taxation, and per-spouse RMDs. The stochastic engine layers on top of it Monte Carlo sequence-of-returns risk, mortality / "widow's-penalty" single-filer transitions, asset location, smile-shaped retirement spending with lump events, and switchable tax regimes.

## What's in the box

```
.
├── tax_optimizer/                    # the package (single source of truth)
│   ├── __init__.py
│   ├── __main__.py                   # CLI: python -m tax_optimizer
│   ├── inputs.py                     # StartingBalances / CurrentIncome / Inputs
│   ├── config.py                     # Config aggregating every knob
│   ├── state.py                      # mutable per-year State
│   ├── rmd.py                        # IRS Uniform Lifetime divisors
│   ├── pension.py                    # cash-balance projector
│   ├── mortality.py                  # widow's-penalty / single-filer transition
│   ├── spending.py                   # SpendingProfile + smile + lump events + LTC
│   ├── market.py                     # Deterministic / Lognormal (w/ correlation + CAPE) / Bootstrap / HistoricalSequence + CMA presets + AssetLocation
│   ├── ira.py                        # Traditional / direct Roth / backdoor Roth allocation (pro-rata aware)
│   ├── limits.py                     # IRS contribution limits (401k, HSA, IRA) + helpers
│   ├── tax/
│   │   ├── regimes.py                # TaxRegime + TCJA_EXTENDED / PRE_TCJA_2017 / SUNSET_2026
│   │   ├── federal.py                # regime + filing-status aware federal_tax
│   │   ├── state.py                  # StateTaxRegime + STATELESS / CA / NY / IL / MA presets
│   │   └── irmaa.py                  # MFJ + Single IRMAA tiers
│   ├── withdrawals.py                # withdraw_for_need + per-strategy solvers + tax-efficient deficit cascade (taxable→Roth→HSA-after-65→pretax)
│   ├── conversion.py                 # planned_roth_conversion (RMD-aware, per-spouse capped)
│   ├── payroll.py                    # FICA: per-W-2 + Form-8959 household reconciliation
│   ├── simulator.py                  # single-path year loop (Medicare base + IRMAA lookback + ACA + step-up)
│   ├── monte_carlo.py                # simulate_paths + MonteCarloResult
│   ├── metrics.py                    # terminal-NW (heir_marginal_rate aware), lifetime-tax-NPV, summarize
│   ├── optimizer.py                  # optimize_household (mega-backdoor + SS-claim-age axes, mc_seed thread)
│   ├── sensitivity.py                # tornado + plain-English actions / takeaways
│   ├── report.py                     # build_action_report + compare_scenarios + cross_model_check
│   └── plots.py                      # matplotlib helpers
├── docs/
│   ├── architecture.md               # per-module reference + cross-cutting diagrams
│   ├── roth_conversion.md            # how Roth conversion sizing + liquidity guards work
│   ├── scenario_guide.md             # reference for every scenario JSON field
│   └── market_models.md              # market-model landscape & design rationale
├── CHANGELOG.md                      # feature / fix log (update on every change)
├── tax_optimizer_standalone.ipynb    # canonical demo notebook (imports from the package)
├── retirement_planning_demo.ipynb    # decision-focused walkthrough driven by example02.json
├── pyproject.toml
├── LICENSE
└── README.md
```

## Requirements

Python **3.10+** on macOS, Linux, or Windows.

## Setup

```bash
git clone https://github.com/vijayyepuri/Tax_Optimizer.git
cd Tax_Optimizer
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[notebook]"
```

(`pip install -e .` is sufficient if you don't want Jupyter.)

[`uv`](https://docs.astral.sh/uv/) works equivalently and is much faster:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[notebook]"
```

## Usage

### CLI

By default the CLI emits the same action plan the standalone notebook
renders (household snapshot, recommended levers, expected outcomes,
top sensitivities, year-by-year action timeline, year-by-year
withdrawal & conversion table, hygiene, caveats), pretty-printed to
the terminal via [Rich](https://rich.readthedocs.io/). When stdout is
piped or redirected, raw markdown is emitted instead so the report
composes cleanly with other tools (`glow`, `pandoc`, `less`, etc.).

For a saved deliverable use `--report PATH.html` for a styled HTML
document or `--report PATH.pdf` for a PDF (PDF requires the optional
`[pdf]` extra plus WeasyPrint's system libraries — see below).
Markdown file output is intentionally not supported by the CLI; if
you need raw markdown, use the `build_action_report()` Python API.

```bash
# Pretty terminal report (default).
python -m tax_optimizer

# Styled HTML.
python -m tax_optimizer --report action_report.html

# PDF (after installing the optional [pdf] extra + system libs).
python -m tax_optimizer --report action_report.pdf

# Drop a timestamped HTML archive under reports/ in addition to stdout.
python -m tax_optimizer --report-archive

# Monte Carlo (sequence-of-returns risk) — adds a "Risk picture" section.
python -m tax_optimizer --market lognormal --monte-carlo 1000

# Cross-model robustness check — adds §4 sub-section comparing the same
# plan under alternative market models (defaults: bootstrap + historical
# sequence). Requires --monte-carlo > 0.
python -m tax_optimizer --market lognormal --monte-carlo 1000 --cross-model

# Custom model list (any mix of built-in kinds + CMA presets).
python -m tax_optimizer --market lognormal --monte-carlo 1000 \
    --cross-model bootstrap,vanguard_2025,jpm_ltcma_2025

# Collapse §7 to retirement years only (legacy v1-v6 compact view).
# Default behaviour is to show the full horizon including working years.
python -m tax_optimizer --year-table-scope retirement

# Widow's-penalty stress test (Spouse A dies year 25).
python -m tax_optimizer --widow 25

# TCJA sunsets in year 5.
python -m tax_optimizer --regime sunset --regime-change-year 5

# Smile-shaped retirement spending + LTC shock.
python -m tax_optimizer --spending smile

# Optimize for CVaR(10%) instead of point-estimate terminal NW.
python -m tax_optimizer --market lognormal --monte-carlo 500 --mc-objective cvar

# Older terse text output (strategy table + plain-English actions only).
python -m tax_optimizer --no-report

# Pipe raw markdown into another renderer.
python -m tax_optimizer | glow -
```

`python -m tax_optimizer --help` lists all flags.

#### Report output flags

| Flag | Effect |
|---|---|
| (none) | Pretty-print to stdout when stdout is a TTY; raw markdown when piped. |
| `--report PATH` | Write the report to `PATH`. Extension must be `.html` or `.pdf`. If `PATH` is an existing directory, the file lands as `PATH/action_report.html`. Stdout is suppressed when this flag is used (pair with `--also-stdout` to keep both). |
| `--report-archive` | Also drop a timestamped HTML copy under `./reports/action_report_YYYY-MM-DD_HHMMSS.html` so re-runs don't clobber prior plans. |
| `--also-stdout` | When `--report PATH` writes a file, *also* print the report to stdout. |
| `--quiet` | Suppress stdout output of the report (useful with `--report PATH`). |
| `--no-report` | Skip the action plan entirely; emit the older short-form text summary instead. |

##### PDF output

PDF rendering is provided by [WeasyPrint](https://weasyprint.org/),
which is an optional dependency because it relies on system libraries
(pango, cairo, gdk-pixbuf):

```bash
# Python side
pip install 'tax-optimizer[pdf]'    # installs weasyprint

# System libraries
brew install pango                   # macOS
sudo apt install libpango-1.0-0 libpangoft2-1.0-0   # Debian/Ubuntu
```

If WeasyPrint or its system libs are missing, `--report foo.pdf` exits
with a clear install hint. HTML output requires no extras and works
out of the box.

#### What the action report contains

`build_action_report` emits a markdown document organized into nine
fixed sections. Anything labelled *(v6)* below was added in the Tier R
"action-report polish" batch — see `CHANGELOG.md` for entry-level
detail.

| # | Section | What it answers |
|---|---|---|
| TL;DR *(v6)* | Verdict + lever changes + key risk readings | "Should I act on this plan, and what's the headline change?" |
| 1 | Household snapshot + assumptions block *(v6)* | "What did the model assume about me, my market view, and my heirs?" |
| 1a | Widow's-penalty paragraph *(v6, conditional)* | "How big is the bracket jump the year my spouse dies?" |
| 1b | Regime-change paragraph *(v6, conditional)* | "How big is the bracket jump at TCJA sunset (or other regime swap)?" |
| 2 | Recommended levers | The 3 optimizer decision variables and the deltas vs. baseline. |
| 2a | Optimizer rationale *(v6)* | "*Why* did the optimizer choose this — what pattern is it responding to?" |
| 3 | Expected outcomes (4-strategy side-by-side *(v6)*) | "How do S0 / S1 / S2 / S3 compare on terminal NW, lifetime tax, IRMAA, peak marginal?" |
| 3a | Heir-rate sensitivity sweep *(v6)* | "Does this verdict survive if my heir is in the 22% bracket instead of 32%?" |
| 4 | Risk picture (Monte Carlo) — optional | P(success), CVaR(10%), median ruin year. Only rendered when `mc=` is passed. |
| 4a | Cross-model robustness check *(v6, conditional)* | Same plan re-scored under Bootstrap + HistoricalSequence. Only rendered when `extra_mc=` is passed. |
| 5 | Tornado sensitivity | The knobs that move terminal NW the most, ranked. |
| 6 | This year's concrete actions *(v6)* | Year-1 dollar amounts: 401(k) deferral, mega-backdoor, backdoor IRA, HSA, conversion target, expected tax bill. |
| 7 | Year-by-year withdrawal & conversion timeline | Per-year action plan. Defaults to the **full horizon** (every simulated year, with a `RETIRE @ N` marker row dividing accumulation from drawdown); pass `year_table_scope="retirement"` / `--year-table-scope retirement` for the legacy retirement-only compact view. State-tax / healthcare columns auto-toggle on. |
| 8 | Hygiene checklist | RMDs taken, bracket-management calls, IRMAA cliff awareness, etc. |
| 9 | Caveats | What the model does and doesn't cover. |

Two helpers ship alongside the report renderer for multi-scenario and
multi-model comparisons:

- **`compare_scenarios(scenarios, *, mc=None)`** *(v6)* — renders an
  N-column markdown diff of independent household configurations.
  Pass `[(name, cfg, inputs), ...]`; best value per row is bolded.
  Use for "stay vs. move state", "retire at 62 vs. 67", "different
  market regimes", etc.
- **`cross_model_check(cfg, inputs, *, n_paths=200, seed=42, models=None)`**
  *(v6)* — re-runs Monte Carlo under alternative market models
  (defaults: `BootstrapModel` + `HistoricalSequenceModel`) and returns
  `{name: MonteCarloResult}`. Passing the result back as
  `build_action_report(..., extra_mc=...)` adds the cross-model
  robustness sub-section inside §4. Also exposed via the CLI as
  `--cross-model [MODELS]` / `--cross-model-paths N` (see the Monte
  Carlo example above).

The Python API exposes the same renderer plus the new HTML/PDF helpers:

```python
from tax_optimizer import (
    Config, Inputs, simulate, simulate_paths, summarize,
    build_action_report, compare_scenarios, cross_model_check,
    optimize_household, tornado_sensitivity, LognormalModel,
)
from tax_optimizer.render import write_html, write_pdf, render_terminal
from pathlib import Path

cfg, inputs = Config(market=LognormalModel()), Inputs()
results = {
    "S0_baseline": (cfg, simulate(cfg, inputs), summarize(simulate(cfg, inputs))),
}
opt_cfg, opt_inputs, _ = optimize_household(cfg, inputs, objective="terminal")
df_opt = simulate(opt_cfg, opt_inputs)
results["S3_optimized"] = (opt_cfg, df_opt, summarize(df_opt))

sens, base_terminal = tornado_sensitivity(cfg, inputs)
mc = simulate_paths(opt_cfg, opt_inputs, n_paths=500, seed=42)

# (v6) Robustness across alternative market models — adds a
# "Cross-model robustness check" section inside §4 of the report.
extra_mc = cross_model_check(opt_cfg, opt_inputs, n_paths=200, seed=42)

report_md = build_action_report(
    cfg, inputs, results, sens, base_terminal,
    mc=mc, extra_mc=extra_mc,
)

# (v6) Side-by-side comparison of N household variants.
comparison_md = compare_scenarios([
    ("Stay in CA",       cfg, inputs),
    ("Move to TX",       cfg_tx, inputs),
    ("Retire 2y later",  cfg, inputs_late),
])

render_terminal(report_md)                       # pretty stdout
write_html(report_md, Path("action_report.html"))
# write_pdf(report_md, Path("action_report.pdf"))  # needs [pdf] extra
```

`optimize_s3` remains as a backward-compatible alias for
`optimize_household`. New code should use `optimize_household`.

#### Custom scenarios

The CLI also accepts a JSON scenario file for everything that isn't covered
by the high-level flags (starting balances, salaries, contribution rates, SS
amounts, custom market / spending objects, etc.). All sections are optional —
omitted fields keep their built-in defaults. Unknown keys raise an error so
typos don't silently no-op.

```bash
# Run with the bundled example scenario.
python -m tax_optimizer --scenario scenarios/example01.json

# Generate a starting template you can edit.
python -m tax_optimizer --print-defaults > my_plan.json
```

Quick one-off tweaks without writing a file (repeatable, JSON literals,
dotted paths into either `config` or `inputs`):

```bash
python -m tax_optimizer \
    --set inputs.spouse_a_age_start=55 \
    --set inputs.spouse_a_retire_age=67 \
    --set config.horizon_age=95 \
    --set inputs.starting.hsa=25000 \
    --set 'config.market={"kind":"lognormal","equity_mu":0.07,"equity_sigma":0.18}'
```

Precedence (low → high): built-in defaults → `--scenario` file → high-level
flags (`--regime`, `--market`, `--spending`, …) → `--set` overrides.

Scenario schema (every key is optional). All "about me" data — including
spouse ages, retire ages, contribution rates, and Roth-401(k) splits —
lives under `inputs`. The scenario layer routes those fields onto the
right internal object for you, so you only have one place to edit
household data:

```json
{
  "config": {
    "horizon_age": 95,
    "tax_regime": "sunset",
    "regime_change_year_offset": 5,
    "regime_change_target": "tcja",
    "market":   { "kind": "lognormal", "equity_mu": 0.07, "equity_sigma": 0.18 },
    "spending": { "kind": "smile", "base_spending": 95000,
                  "ltc_years": 4, "ltc_annual_today": 100000 },
    "mortality": { "year_of_death_a": 25, "pension_survivor_pct": 0.5 },
    "asset_location": { "uniform_equity_pct": 0.7 }
  },
  "inputs": {
    "spouse_a_age_start": 52,
    "spouse_b_age_start": 50,
    "spouse_a_retire_age": 65,
    "spouse_b_retire_age": 67,
    "spouse_a_total_contrib_pct": 0.10,
    "spouse_b_total_contrib_pct": 0.08,
    "spouse_a_roth_401k_pct": 0.0,
    "spouse_b_roth_401k_pct": 0.0,
    "starting": { "spouse_a_pretax_401k": 400000, "hsa": 25000 },
    "income":   { "spouse_a_gross": 140000, "spouse_a_bonus": 15000 },
    "ss":       { "monthly_spouse_a": 3100, "monthly_spouse_b": 2500 }
  }
}
```

> Expected spending is set on the **`config`** side, either via the
> `spending` block (`base_spending` + optional smile / LTC / lump
> events) or — when `spending` is `null` — the `annual_expenses_today`
> scalar. The legacy `inputs.annual_expenses` field is retained for
> backward compatibility but is ignored by the simulator and emits a
> `DeprecationWarning` when set. See
> [`docs/scenario_guide.md`](docs/scenario_guide.md#spending-knobs--three-names-one-effective-value)
> for migration details.

The household-specific fields under `inputs` are:

| `inputs.<field>` | Range | What it does |
|---|---|---|
| `spouse_a_age_start`, `spouse_b_age_start` | int (years) | Each spouse's age at simulation start. |
| `spouse_a_retire_age`, `spouse_b_retire_age` | int (years) | Age each spouse stops earning W-2 income. |
| `spouse_a_total_contrib_pct`, `spouse_b_total_contrib_pct` | 0.0–1.0 | Fraction of salary deferred into 401(k) (Traditional + Roth combined). |
| `spouse_a_roth_401k_pct`, `spouse_b_roth_401k_pct` | 0.0–1.0 | Of that deferral, the fraction routed to Roth (the rest is Traditional). |

These fields used to live on `Config`; they're now part of `Inputs`
together with the rest of the "about me" data so the same `Config`
strategy can be reused across households. Old scenario files that put
them under `config.<field>` (or `--set config.spouse_*=…`) get a clear
migration error pointing at the new location.

See `scenarios/example01.json` (and `example02.json`) for full-featured
examples, and run `--print-defaults` against any scenario to see the
resolved values.

**Starting from scratch?** Copy `scenarios/template.json` — it lists
every available knob with its default value, and the
`scenarios/README.md` file in the same directory documents the
polymorphic options for `market` / `spending` / `state_regime` /
`tax_regime` / `asset_location`. The template is enforced in sync
with the dataclass surface by `tests/test_scenario_template.py`, so
new knobs always make it into the reference file.

##### Allowed values for enum-style fields

Anywhere a string name appears below, it is case-insensitive.

**`config.tax_regime`** and **`config.regime_change_target`** — pick the
active bracket / IRMAA / NIIT / standard-deduction tables. Aliases in
parentheses are accepted equivalents.

| Value | Bundled regime | Notes |
|---|---|---|
| `"tcja"` (`"tcja_extended"`) | `TCJA_EXTENDED` | Default. TCJA brackets extended past 2025. |
| `"pre_tcja"` (`"pre_tcja_2017"`) | `PRE_TCJA_2017` | 2017-style brackets, higher top rate, smaller std deduction. |
| `"sunset"` (`"sunset_2026"`) | `SUNSET_2026` | TCJA sunset variant: rates revert, std deduction drops. |

`regime_change_target` accepts the same set of values, plus `null` to
disable a previously-set mid-simulation regime change. Combine with
`regime_change_year_offset` (an integer year-offset from start) to flip
regimes mid-plan.

**`config.market.kind`** — yearly return source. Each kind has its own
parameter set; only the parameters listed for that kind are accepted.

| `kind` | Parameters (all optional; defaults shown) | Meaning |
|---|---|---|
| `"deterministic"` | `equity` (0.07), `bond` (0.04) | Constant returns every year. |
| `"lognormal"` | `equity_mu` (0.07), `equity_sigma` (0.18), `bond_mu` (0.04), `bond_sigma` (0.06) | Independent yearly draws from a normal distribution on returns. |
| `"bootstrap"` | `block_size` (5), `equity_history` (1928–2023 S&P 500), `bond_history` (1928–2023 10y Treasury) | Block-bootstrap from a historical series. Pass your own histories as JSON arrays of annual returns to override the defaults. |

`config.market` can also be `null` to fall back to a `DeterministicModel`
seeded from `config.nominal_growth_rate` (the v1-compatible behavior).

**`config.spending.kind`** — retirement-spending shape. `base_spending`
is required for every kind.

| `kind` | Parameters | Meaning |
|---|---|---|
| `"flat"` | `base_spending`, `inflation` (0.025) | Single phase, inflation-adjusted forward. |
| `"smile"` | `base_spending`, `inflation` (0.025), `ltc_years` (3), `ltc_annual_today` (80000) | Blanchett retirement smile (working/go-go/slow-go/no-go) plus an LTC shock in the final `ltc_years` years. |
| `"custom"` | `base_spending`, `inflation` (0.025), `phases` (list of `{age_lo, age_hi, multiplier, label}`), `lump_events` (list of `{year_offset, amount_today, label, preferred_source}`), `ltc_shock` (`{years, annual_cost_today}` or `null`) | Fully user-defined phases, one-off cash outflows, and optional LTC shock. |

`config.spending` can also be `null` to use a flat profile derived from
`config.annual_expenses_today` and `config.inflation`.

**`config.asset_location`** — per-account equity/bond mix. Two forms,
mutually exclusive:

```json
"asset_location": { "uniform_equity_pct": 0.7 }
```
…sets every account to the same equity %, or:

```json
"asset_location": {
  "pretax_equity_pct":  0.4,
  "roth_equity_pct":    1.0,
  "taxable_equity_pct": 0.8,
  "hsa_equity_pct":     0.8
}
```

**`config.mortality`** — accepts the full `Mortality` field set:
`year_of_death_a` (int year-offset or `null`), `year_of_death_b` (same),
`pension_survivor_pct` (0.0–1.0, default 0.5), `ss_survivor_keeps_higher`
(bool, default `true`).

For `lump_events[].preferred_source` (inside a custom spending profile),
the allowed values are `"taxable"`, `"pretax"`, `"roth"`, `"any"`
(default).

### Python API

```python
from tax_optimizer import (
    Config, Inputs, simulate, simulate_paths, optimize_household,
    LognormalModel, Mortality, SpendingProfile, SUNSET_2026,
)
from dataclasses import replace

inputs = Inputs()           # defaults match the README scenario
cfg = Config(market=LognormalModel())

# Single deterministic path:
df = simulate(cfg, inputs)

# 1000-path Monte Carlo (mc_seed makes the path set reproducible):
mc = simulate_paths(cfg, inputs, n_paths=1000, seed=42)
print(mc.summary())          # prob_success, p5/p50/p95 terminal, CVaR(10%), ...

# Stress-test the widow's penalty:
cfg_widow = replace(cfg, mortality=Mortality(year_of_death_a=25))

# Optimize for probability of success instead of point-estimate NW
# (returns the full triple — Config + Inputs + raw decision vector):
cfg_opt, inputs_opt, x_opt = optimize_household(
    cfg, inputs, objective="p_success", n_paths=500, mc_seed=42,
)
```

### Notebooks

Two demo notebooks live at the repo root:

```bash
# Canonical walkthrough — every feature, in order, with explanatory prose.
jupyter lab tax_optimizer_standalone.ipynb

# Decision-focused workflow driven by scenarios/example02.json (added v6).
jupyter lab retirement_planning_demo.ipynb
```

**`tax_optimizer_standalone.ipynb`** is the canonical reference. It
imports from the package and walks through:

1. Scenario inputs.
2. First-year tax sanity check.
3. S0 / S1 / S2 / S3 strategy comparison.
4. Visualizations.
5. Year-by-year detail of the winning strategy.
6. Tornado sensitivity.
7. Recommended actions + takeaways.
8. Monte Carlo (lognormal + bootstrap).
9. Widow's-penalty stress test.
10. TCJA-sunset stress test.
11. Smile-shaped spending + lump events.
12. Asset location (bonds in pretax, equities in Roth).
13. Market-model deep dive (Lognormal + correlation, CAPE conditioning, Historical replay, CMA presets).
14. Tier C feature cheat sheet (ACA, step-up, healthcare costs, per-spouse SS).
15. "What changed" recap.

**`retirement_planning_demo.ipynb`** is shorter and more decision-shaped.
It loads `scenarios/example02.json` once and threads a single household
through 13 numbered sections — household snapshot, first-year sanity
check, 4-strategy comparison (with an "optimizer is maximizing terminal
after-tax NW" callout), year-by-year detail, tornado, Monte Carlo,
widow's / sunset stress tests, Tier C feature dial, full action report
(including the v6 cross-model robustness check), and a v5 → v6
changelog summary. Start here if you want a worked example to copy
rather than a feature tour.

## Decision variables

The model distinguishes four kinds of inputs, from narrowest to broadest:

### 1. Optimizer search variables (S3)

The variables `optimize_household` searches over. Everything else is
fixed. As of v5 the decision vector is constructed dynamically: the
three core axes are always present, and additional axes turn on when
their feature flags are set on `Config` / `Inputs`.

| Variable | Lives on | Domain | Always on? | Meaning |
|---|---|---|---|---|
| `spouse_a_roth_401k_pct` | `Inputs` | `[0, 1]` | yes | Fraction of Spouse A's 401(k) deferrals routed to Roth. |
| `spouse_b_roth_401k_pct` | `Inputs` | `[0, 1]` | yes | Same, for Spouse B. |
| `roth_conversion_target_bracket` | `Config` | `{0%, 12%, 22%, 24%, 32%}` | yes | Bracket the converter fills *up to* during gap years. |
| `spouse_*_after_tax_401k_pct` *(v5)* | `Inputs` | `[0, 0.10]` | when `inputs.spouse_*_mega_backdoor_enabled = True` | After-tax 401(k) deferral routed straight to a mega-backdoor Roth in-plan conversion. One axis per spouse with the flag set. |
| `ss.start_age_a`, `ss.start_age_b` *(v5)* | `Inputs` | `{62, 63, …, 70}` | when `cfg.optimize_ss_claim_age = True` | Per-spouse Social Security claim age. |

Three optimizer objectives:

- `'terminal'` — point-estimate terminal after-tax NW (deterministic only).
- `'cvar'` — average terminal NW across the worst α% of Monte Carlo paths.
- `'p_success'` — probability the plan never runs out of money.

The Monte Carlo objectives accept a `mc_seed=` keyword on
`optimize_household` so each candidate is evaluated against the *same*
path set — without it, `differential_evolution` would see objective
noise from path resampling and converge unevenly.

### 2. Where each knob lives — `Config` vs. `Inputs`

The split is intentional and keeps the same `Config` reusable across households:

- **`Inputs`** — the *household*. Spouse ages, retire ages, salaries, contribution rates, Roth-401(k) splits, starting balances, Social Security amounts (incl. claim age `inputs.ss.start_age`), pension data (incl. start age `inputs.pension.start_age`), expected expenses, etc. See `tax_optimizer/inputs.py`.
- **`Config`** — the *simulation / strategy*. Macro assumptions (`nominal_growth_rate`, `inflation`, `wage_growth`), the one remaining age-gated policy event (`rmd_start_age`), the withdrawal / conversion strategy, and the modular assumption blocks below. See `tax_optimizer/config.py`.

The optimizer's decision variables follow the same split: the Roth-401(k) splits, after-tax-401(k) splits, and SS claim ages land on `Inputs` (household-level choices); the conversion bracket target lands on `Config` (a strategy-level rule). `optimize_household` therefore returns `(best_cfg, best_inputs, x_opt)`, and each of the four canonical strategies in the CLI carries its own `(cfg, inputs)` pair packaged as a `StrategyResult`. The old name `optimize_s3` is kept as a backward-compatible alias.

### 3. Modular assumption blocks (v2)

Each is a separate dataclass — set `cfg.<field>` to swap behavior:

| Block | What it does |
|---|---|
| `cfg.tax_regime` (`TaxRegime`) | Active bracket / IRMAA / NIIT / std-deduction tables. Bundled: `TCJA_EXTENDED`, `PRE_TCJA_2017`, `SUNSET_2026`. |
| `cfg.regime_change_year_offset` + `cfg.regime_change_target` | Mid-simulation regime swap. Use to model TCJA sunset, future legislation, etc. |
| `cfg.mortality` (`Mortality`) | Per-spouse year of death + survivor pension election. Switches filing status to single. |
| `cfg.market` (`MarketModel`) | `DeterministicModel` (constant returns), `LognormalModel` (μ/σ draws), or `BootstrapModel` (1928-2023 historical resampling). |
| `cfg.asset_location` (`AssetLocation`) | Per-account equity/bond mix. Default puts bonds in pretax, equities in Roth. |
| `cfg.spending` (`SpendingProfile`) | Smile-shaped age multipliers + lump events + LTC shock. |

### 4. Tornado sensitivity perturbations

`tornado_sensitivity(cfg, inputs)` perturbs each knob ± a small step and ranks them by how much terminal NW moves. Use it to spot the highest-leverage changes for your scenario.

### Not a decision variable

Federal bracket numbers, IRS Uniform Lifetime divisors, pension-formula coefficients, Social-Security provisional-income thresholds. These are *parameters* of the model and live in package data files; treat them as fixed.

## What's modeled

- **Federal income tax** — ordinary brackets, qualified-dividend / LTCG brackets, NIIT, standard deduction.
- **State income tax** — CA / NY / IL / MA presets (progressive or flat, with retirement-income carve-outs and SS exemptions); easily extensible via `StateTaxRegime(...)`.
- **Social-Security taxation** — IRC §86 provisional-income calculation with the right thresholds for filing status, plus year-of-death MFJ handling and survivor benefits from age 60.
- **FICA** — per-W-2 SS + Medicare withholding plus the Form-8959 household reconciliation for the Additional Medicare 0.9% surcharge at the MFJ threshold. §125 cafeteria-plan deductions (HSA + medical/dental/vision premiums) reduce FICA wages when `cfg.section125_reduces_fica_wages = True` (v6.6 default, matches real Box 3 / Box 5 payroll).
- **§125 health-insurance premiums** — `inputs.health_premiums` quotes the employee share of medical, dental, and vision premiums per spouse. Reduces federal Box 1, FICA, and state wages (per-spouse gating + gross-wage clamp). See `scenarios/README.md` for setup.
- **IRMAA** — current 2026 tier table for both MFJ and single filers; auto-switches when filing status changes; two-year MAGI lookback.
- **Medicare premiums** — base Part B + Part D, pre-Medicare ACA premium with the IRA-2022 8.5%-of-MAGI premium-tax-credit cap.
- **RMDs** — IRS Uniform Lifetime divisors for ages 72-110, computed per spouse on their own pretax balances.
- **Roth conversions** — RMD-aware, per-spouse capped, bracket-target driven; mega-backdoor support via dedicated decision-vector axis.
- **Mega-backdoor auto-spillover** *(v6.6)* — when `spouse_*_mega_backdoor_enabled = True`, any elective-deferral target above the §402(g) cap auto-routes into the after-tax 401(k) bucket up to the §415(c) ceiling (Vanguard "Spillover After-Tax" analog). Surfaces via `excess_deferral_*` / `mega_backdoor_spillover_*` / `after_tax_target_uncovered_*` diagnostic columns.
- **IRA contribution paths** — Traditional / direct Roth / backdoor Roth (pro-rata aware on IRA-only sub-balance).
- **Employer 401(k) match** — match-rate × match-cap per spouse, layered on top of employee deferrals.
- **Cost-basis tracking** — taxable brokerage basis fraction is dynamic, not constant; full step-up to FMV on first spouse's death (community-property model).
- **Sequence-of-returns risk** — Monte Carlo with `LognormalModel` (with equity-bond correlation + optional CAPE conditioning), `BootstrapModel` (1928–2023 block bootstrap), or `HistoricalSequenceModel` (one contiguous slice of real history per path).
- **Calibration presets** — Vanguard, JPM, Horizon CMAs via `lognormal_from_cma(...)`.
- **Asset location** — separate equity/bond allocation per account type.
- **Mortality** — single-spouse death event with proper survivor SS / pension / filing-status transitions (year-of-death MFJ, survivor benefits at 60, step-up in basis).
- **Spending phases** — Blanchett-style retirement smile + LTC shock + arbitrary lump events.
- **Tax legislation change** — switchable regimes (TCJA / pre-TCJA / sunset), mid-simulation regime swap.

## Documentation

| Document | Purpose |
|---|---|
| [`CHANGELOG.md`](CHANGELOG.md) | Every feature, behavior change, and bug fix shipped (and the rules for adding new entries). The first stop when "what's new?" or "when did we add X?" comes up. The `v6` block documents the action-report polish covered in [What the action report contains](#what-the-action-report-contains) above. |
| [`docs/architecture.md`](docs/architecture.md) | Per-module deep dive: what each `tax_optimizer/*.py` file does, how the modules layer together, and Mermaid diagrams for the cross-cutting flows (year-loop sequence, contribution cascade, withdrawal cascade, tax pipeline, Roth-conversion liquidity guard, optimizer/MC relationship). Start here when extending or auditing the package. |
| [`docs/roth_conversion.md`](docs/roth_conversion.md) | Mechanism-focused walkthrough of Roth conversion sizing (fixed vs bracket-fill), the v6.5 liquidity guard (`tax_paying_capacity` formula + bisection), Roth protection in the deficit cascade, and the seven knobs that control it all. Includes a numerical worked example and audit recipes for the diagnostic columns. |
| [`docs/scenario_guide.md`](docs/scenario_guide.md) | Reference for every field in a scenario JSON — `config.*`, `inputs.*`, all knobs and their defensible ranges. |
| [`docs/market_models.md`](docs/market_models.md) | Landscape of retirement Monte Carlo market models, the industry segments that use each one, and the design rationale behind which models we ship (vs. deliberately skip). Includes the v6 `cross_model_check` API for surfacing model-choice risk in the action report. |

## What's not modeled (yet)

- State income tax for states beyond CA / NY / IL / MA (the bundled
  presets cover ~40% of the US population by income; add via
  `StateTaxRegime(...)`).
- ACA premium tax credit beyond the post-IRA-2022 8.5%-of-MAGI cap
  (the simplified path ships in v5; full Form 8962 with FPL ×
  household-size lookup is in the Tier D backlog).
- Common-law half step-up in basis (the v5 step-up models the
  community-property full step-up; common-law is in Tier D).
- Estate-tax planning beyond the `heir_marginal_rate` haircut in
  `terminal_after_tax_nw`.
- Inherited IRA / 10-year-rule intra-period drawdown timing.
- Multi-asset-class market models (TIPS / international / REIT as
  separate buckets).
- **Roth 5-year clock** on conversions: Roth withdrawals are
  modeled as fully tax-free. In real life, *converted* Roth dollars
  must season for 5 years before they can be withdrawn penalty-free
  if the account holder is under 59½, and the first contribution
  itself must be 5+ years old before the Roth account satisfies the
  qualified-distribution rule. For optimizer users **already 59½ or
  older with an existing Roth >5 years old**, this is a no-op. For
  optimizer users pre-59½ who plan to spend a recent conversion
  early, the simulator's tax-free Roth treatment will overstate
  their plan's net by the avoided 10% penalty. Workaround: ear-mark
  pre-59½ liquidity as taxable / HSA spending instead of Roth.
  **v6.5 partial mitigation**: with
  `cfg.protect_roth_in_conversion_years=True` (default), the
  deficit cascade refuses to withdraw from Roth in any year a
  conversion fires, so the simulator no longer silently raids the
  just-converted bucket to pay the conversion's own tax bill.
  Liquidity overshoot now surfaces as `unfunded` instead.
- **Pre-existing nondeductible IRA basis (Form 8606)**: backdoor-
  Roth pro-rata math treats *all* existing pretax-IRA dollars as
  zero-basis. If you have historical 8606 basis from prior
  nondeductible Traditional contributions, your real backdoor tax
  bill will be lower than the model's estimate. Conservative bias.
- **Dual cash-balance pensions**: the `PensionInputs` / projector /
  simulator pension column track spouse A only. Households where
  both spouses have cash-balance pensions should enter the *combined*
  monthly-at-NRD as a single spouse-A figure; the SS-taxability and
  state-pension-exclusion paths key off spouse A's age.
- **Arithmetic-vs-true-lognormal return draws**: `LognormalModel`
  samples returns from a `Normal(mu, sigma)` distribution (returns
  unbounded below). Real-world annual returns are bounded by -100%
  and have fat left tails. For 30y/50% equity portfolios the
  difference is small (<1% in terminal-NW percentiles); for highly
  leveraged or short-horizon analyses it can matter. Workaround:
  use `BootstrapModel` or `HistoricalSequenceModel` which sample
  empirical returns and are bounded by definition.

## Acronyms

| Acronym | Full Form | Context |
|---------|-----------|---------|
| **MFJ** | Married Filing Jointly | Tax filing status for couples |
| **RMD** | Required Minimum Distribution | IRS-mandated annual retirement account withdrawals starting at age 72 |
| **LTCG** | Long-Term Capital Gains | Investment gains taxed at preferential rates (vs. ordinary income) |
| **NIIT** | Net Investment Income Tax | 3.8% tax on investment income for high earners |
| **IRMAA** | Income-Related Monthly Adjustment Amount | Medicare premium surcharge based on modified adjusted gross income |
| **SS** | Social Security | Federal retirement / survivor benefits program |
| **IRC** | Internal Revenue Code | U.S. federal tax law |
| **NW** | Net Worth | Total assets minus liabilities |
| **CVaR** | Conditional Value at Risk | Expected value in the worst α% of outcomes (risk metric) |
| **LTC** | Long-Term Care | Extended care / nursing-home expenses |
| **TCJA** | Tax Cuts and Jobs Act | 2017 U.S. tax reform (many provisions sunset in 2026) |
| **CLI** | Command-Line Interface | Text-based program interface |
| **NPV** | Net Present Value | Present-day value of future cash flows |
| **ACA** | Affordable Care Act | U.S. healthcare law affecting pre-Medicare premium subsidies |

## Disclaimer

This software is for **educational and illustrative purposes only**. It is not tax, legal, or investment advice. Tax law changes frequently and individual situations vary — consult a qualified professional before acting on any output.

## License

See [LICENSE](LICENSE).
