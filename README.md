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
│   ├── market.py                     # Deterministic / Lognormal / Bootstrap + AssetLocation
│   ├── tax/
│   │   ├── regimes.py                # TaxRegime + TCJA_EXTENDED / PRE_TCJA_2017 / SUNSET_2026
│   │   ├── federal.py                # regime + filing-status aware federal_tax
│   │   └── irmaa.py                  # MFJ + Single IRMAA tiers
│   ├── withdrawals.py                # withdraw_for_need + per-strategy solvers
│   ├── conversion.py                 # planned_roth_conversion (gap-year)
│   ├── simulator.py                  # single-path year loop
│   ├── monte_carlo.py                # simulate_paths + MonteCarloResult
│   ├── metrics.py                    # terminal-NW, lifetime-tax-NPV, summarize
│   ├── optimizer.py                  # optimize_s3 (terminal / cvar / p_success)
│   ├── sensitivity.py                # tornado + plain-English actions / takeaways
│   └── plots.py                      # matplotlib helpers
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

```bash
# Deterministic 4-strategy comparison + tornado + recommendations.
python -m tax_optimizer

# Monte Carlo (sequence-of-returns risk).
python -m tax_optimizer --market lognormal --monte-carlo 1000

# Widow's-penalty stress test (Spouse A dies year 25).
python -m tax_optimizer --widow 25

# TCJA sunsets in year 5.
python -m tax_optimizer --regime sunset --regime-change-year 5

# Smile-shaped retirement spending + LTC shock.
python -m tax_optimizer --spending smile

# Optimize for CVaR(10%) instead of point-estimate terminal NW.
python -m tax_optimizer --market lognormal --monte-carlo 500 --mc-objective cvar
```

`python -m tax_optimizer --help` lists all flags.

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

| Variable | Domain | Meaning |
|---|---|---|
| `spouse_a_roth_401k_pct` | `[0, 1]` | Fraction of Spouse A's 401(k) deferrals routed to Roth. |
| `spouse_b_roth_401k_pct` | `[0, 1]` | Same, for Spouse B. |
| `roth_conversion_target_bracket` | `{0%, 12%, 22%, 24%, 32%}` | Bracket the converter fills *up to* during gap years. |

Three optimizer objectives:

- `'terminal'` — point-estimate terminal after-tax NW (deterministic only).
- `'cvar'` — average terminal NW across the worst α% of Monte Carlo paths.
- `'p_success'` — probability the plan never runs out of money.

### 2. Policy knobs in `Config`

Edit any field on `Config` to model an alternate plan: contribution rates, retire age, SS start age, withdrawal strategy, conversion target, etc. See `tax_optimizer/config.py` for the full list.

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

## What's not modeled (yet)

- State income tax (federal-only today; PRs welcome).
- ACA premium-tax-credit cliffs in pre-Medicare years.
- Estate-tax planning beyond the 22% step-up assumption in `terminal_after_tax_nw`.
- Inherited IRA / 10-year-rule treatment.

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
