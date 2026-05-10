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
│   └── plots.py                      # matplotlib helpers
├── docs/
│   ├── scenario_guide.md             # reference for every scenario JSON field
│   └── market_models.md              # market-model landscape & design rationale
├── CHANGELOG.md                      # feature / fix log (update on every change)
├── tax_optimizer_standalone.ipynb    # demo notebook (imports from the package)
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

The Python API exposes the same renderer plus the new HTML/PDF helpers:

```python
from tax_optimizer import (
    Config, Inputs, simulate, summarize, build_action_report,
    optimize_s3, tornado_sensitivity,
)
from tax_optimizer.render import write_html, write_pdf, render_terminal
from pathlib import Path

cfg, inputs = Config(), Inputs()
results = {
    "S0_baseline": (cfg, simulate(cfg, inputs), summarize(simulate(cfg, inputs))),
}
opt_cfg, _ = optimize_s3(cfg, inputs, objective="terminal")
df_opt = simulate(opt_cfg, inputs)
results["S3_optimized"] = (opt_cfg, df_opt, summarize(df_opt))

sens, base_terminal = tornado_sensitivity(cfg, inputs)
report_md = build_action_report(cfg, inputs, results, sens, base_terminal)

render_terminal(report_md)                       # pretty stdout
write_html(report_md, Path("action_report.html"))
# write_pdf(report_md, Path("action_report.pdf"))  # needs [pdf] extra
```

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
    "ss":       { "monthly_spouse_a": 3100, "monthly_spouse_b": 2500 },
    "annual_expenses": 95000
  }
}
```

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
    Config, Inputs, simulate, simulate_paths, optimize_s3,
    LognormalModel, Mortality, SpendingProfile, SUNSET_2026,
)
from dataclasses import replace

inputs = Inputs()           # defaults match the README scenario
cfg = Config(market=LognormalModel())

# Single deterministic path:
df = simulate(cfg, inputs)

# 1000-path Monte Carlo:
mc = simulate_paths(cfg, inputs, n_paths=1000)
print(mc.summary())          # prob_success, p5/p50/p95 terminal, CVaR(10%), ...

# Stress-test the widow's penalty:
cfg_widow = replace(cfg, mortality=Mortality(year_of_death_a=25))

# Optimize for probability of success instead of point-estimate NW:
cfg_opt, x_opt = optimize_s3(cfg, inputs, objective='p_success', n_paths=500)
```

### Notebook

```bash
jupyter lab tax_optimizer_standalone.ipynb
```

The notebook imports from the package and walks through:

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

## Decision variables

The model distinguishes four kinds of inputs, from narrowest to broadest:

### 1. Optimizer search variables (S3)

The only variables `optimize_s3` searches over. Everything else is fixed.

| Variable | Lives on | Domain | Meaning |
|---|---|---|---|
| `spouse_a_roth_401k_pct` | `Inputs` | `[0, 1]` | Fraction of Spouse A's 401(k) deferrals routed to Roth. |
| `spouse_b_roth_401k_pct` | `Inputs` | `[0, 1]` | Same, for Spouse B. |
| `roth_conversion_target_bracket` | `Config` | `{0%, 12%, 22%, 24%, 32%}` | Bracket the converter fills *up to* during gap years. |

Three optimizer objectives:

- `'terminal'` — point-estimate terminal after-tax NW (deterministic only).
- `'cvar'` — average terminal NW across the worst α% of Monte Carlo paths.
- `'p_success'` — probability the plan never runs out of money.

### 2. Where each knob lives — `Config` vs. `Inputs`

The split is intentional and keeps the same `Config` reusable across households:

- **`Inputs`** — the *household*. Spouse ages, retire ages, salaries, contribution rates, Roth-401(k) splits, starting balances, Social Security amounts (incl. claim age `inputs.ss.start_age`), pension data (incl. start age `inputs.pension.start_age`), expected expenses, etc. See `tax_optimizer/inputs.py`.
- **`Config`** — the *simulation / strategy*. Macro assumptions (`nominal_growth_rate`, `inflation`, `wage_growth`), the one remaining age-gated policy event (`rmd_start_age`), the withdrawal / conversion strategy, and the modular assumption blocks below. See `tax_optimizer/config.py`.

The optimizer's three decision variables follow the same split: the Roth-401(k) splits land on `Inputs` (a household-level deferral choice) and the conversion bracket target lands on `Config` (a strategy-level rule). `optimize_s3` therefore returns `(best_cfg, best_inputs, x_opt)`, and each of the four canonical strategies in the CLI carries its own `(cfg, inputs)` pair packaged as a `StrategyResult`.

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
- **Social-Security taxation** — IRC §86 provisional-income calculation with the right thresholds for filing status.
- **IRMAA** — current 2026 tier table for both MFJ and single filers; auto-switches when filing status changes.
- **RMDs** — IRS Uniform Lifetime divisors for ages 72-110, computed per spouse on their own pretax balances.
- **Cost-basis tracking** — taxable brokerage basis fraction is dynamic, not constant.
- **Sequence-of-returns risk** — Monte Carlo with lognormal or block-bootstrap historical resampling.
- **Asset location** — separate equity/bond allocation per account type.
- **Mortality** — single-spouse death event with proper survivor SS / pension / filing-status transitions.
- **Spending phases** — Blanchett-style retirement smile + LTC shock + arbitrary lump events.
- **Tax legislation change** — switchable regimes, mid-simulation regime swap.

## Documentation

| Document | Purpose |
|---|---|
| [`CHANGELOG.md`](CHANGELOG.md) | Every feature, behavior change, and bug fix shipped (and the rules for adding new entries). The first stop when "what's new?" or "when did we add X?" comes up. |
| [`docs/scenario_guide.md`](docs/scenario_guide.md) | Reference for every field in a scenario JSON — `config.*`, `inputs.*`, all knobs and their defensible ranges. |
| [`docs/market_models.md`](docs/market_models.md) | Landscape of retirement Monte Carlo market models, the industry segments that use each one, and the design rationale behind which models we ship (vs. deliberately skip). |

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
