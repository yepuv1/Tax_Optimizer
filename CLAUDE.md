# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Install (editable):**
```bash
pip install -e ".[dev,notebook]"
# or faster:
uv pip install -e ".[dev,notebook]"
```

**Run CLI:**
```bash
python -m tax_optimizer                                         # terminal report
python -m tax_optimizer --scenario scenarios/example.json       # custom scenario
python -m tax_optimizer --report report.html                    # HTML output
python -m tax_optimizer --market lognormal --monte-carlo 1000   # Monte Carlo
python -m tax_optimizer --print-defaults > my_plan.json         # dump defaults
```

**Test:**
```bash
pytest                          # all tests
pytest tests/test_tier_a.py     # single file
pytest --cov=tax_optimizer      # with coverage
```

**Lint / format:**
```bash
ruff check tax_optimizer/ tests/
black tax_optimizer/ tests/
mypy tax_optimizer/
```

Line length is 100, target is Python 3.10 (configured in `pyproject.toml`). Coverage omits `__main__.py`, `plots.py`, and `render.py`.

## Architecture

### Config vs. Inputs

Two root dataclasses divide responsibilities:

- **`Inputs`** (`inputs.py`) — household facts: spouse ages, salaries, contribution rates, starting balances, Social Security, pension. Contains nested dataclasses `StartingBalances`, `CurrentIncome`, `CurrentContrib`, `PensionInputs`, `SocialSecurity`.
- **`Config`** (`config.py`) — simulation strategy and assumptions: growth rates, inflation, withdrawal strategy, tax regime, market model, asset location, spending profile, healthcare costs, optimizer scope. Key helpers: `effective_regime(year_offset)`, `resolved_market()`.

Never mutate these objects — the optimizer and strategy runner use `dataclasses.replace()` throughout.

### Six Pluggable Blocks

These can be swapped independently in `Config`:

| Block | Module | Built-ins |
|-------|--------|-----------|
| `TaxRegime` | `tax/regimes.py` | `TCJA_EXTENDED`, `PRE_TCJA_2017`, `SUNSET_2026` |
| `StateTaxRegime` | `tax/state.py` | `STATELESS`, `CA`, `NY`, `IL`, `MA` |
| `MarketModel` | `market.py` | `DeterministicModel`, `LognormalModel`, `BootstrapModel`, `HistoricalSequenceModel` |
| `AssetLocation` | `market.py` | `AssetMix` per account (pretax/roth/taxable/hsa) |
| `SpendingProfile` | `spending.py` | `flat()`, `retirement_smile()`, manual with phases/lumps/LTC |
| `Mortality` | `mortality.py` | Per-spouse year-of-death, MFJ→single transition |

`MarketModel` is a `typing.Protocol` — any class with `begin_path()` and `returns()` works without inheritance.

### Simulation Engine (`simulator.py`)

The year-by-year loop processes these steps in order:
1. Mortality check (alive status, filing status, spousal IRA rollover, step-up in basis)
2. Contributions (401k cap, employer match, mega-backdoor Roth, HSA, IRA with pro-rata)
3. Income (W-2, FICA, pension, Social Security, portfolio yield)
4. RMDs (per spouse, before conversions)
5. Roth conversions (bracket-fill or fixed-dollar)
6. Spending need (SpendingProfile smile + lump + LTC, HSA paydown)
7. Withdrawals (three strategies: conventional, proportional, bracket_fill)
8. Deficit cascade (taxable → Roth → HSA-after-65 → pretax)
9. Tax (federal ordinary + LTCG + NIIT, state, IRMAA with 2-year lookback, ACA credit)
10. Growth (market returns via asset location)

Returns a `pd.DataFrame` with ~80 columns per year.

**Binary-search gross-up:** `_solve_pretax_for_net` and `_solve_taxable_for_net` use 50-iteration bisection to find the gross withdrawal needed to produce a target net-of-tax amount.

### Monte Carlo and Optimizer Layers

- **`monte_carlo.py`** — `simulate_paths()` runs N independent paths with different RNG seeds. Returns `MonteCarloResult` with `prob_success()`, `cvar_terminal()`, percentiles.
- **`optimizer.py`** — `optimize_household()` uses `scipy.optimize.differential_evolution`. Decision vector always includes Roth 401(k) pct (both spouses) and conversion bracket index; optionally mega-backdoor pct and SS claim ages. Three objectives: terminal NW, CVaR, probability of success.

### Four Canonical Strategies

The CLI evaluates these side-by-side:
- **S0_baseline** — current allocations, no conversions
- **S1_all_roth_401k** — 100% Roth 401(k) both spouses
- **S2_bracket_fill_22** — fill to 22% bracket with Roth conversions
- **S3_optimized** — differential evolution result

### Reporting Pipeline

1. **`report.py`** (1,495 lines) — generates 9-section markdown action plan (TL;DR, snapshot, levers, outcomes, risk, tornado, year-1 actions, year-by-year timeline, caveats). Also `compare_scenarios()` and `cross_model_check()`.
2. **`render.py`** — renders to terminal (Rich), HTML (built-in CSS), or PDF (WeasyPrint).
3. **`sensitivity.py`** — tornado perturbation analysis via `dataclasses.fields()` introspection.

### Scenario System (`scenario.py`)

Loads JSON into `Config`/`Inputs`, validates (unknown keys raise errors), supports `--set DOTTED.PATH=VALUE` CLI overrides, and handles migration errors for fields that moved between versions.
