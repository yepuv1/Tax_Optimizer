# Changelog

All notable feature additions, modeling enhancements, and bug fixes
to `tax_optimizer` are tracked here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The package version constant lives in `tax_optimizer/__init__.py`.
Section labels (`v2`, `v3 — Tier A`, `v3 — Tier B`, `v4`) match the
"What changed" recap at the bottom of the standalone notebooks.

> **Maintenance rule:** **Every** new feature, behavior change, or
> bug fix added to this repo MUST get a one-line entry in the
> `[Unreleased]` section below before the change is committed.
> Entries graduate from `[Unreleased]` to a versioned section when
> a release is cut.
>
> When in doubt, prefer a smaller, more specific entry over a vague
> sweeping one. The point is for someone six months from now (or
> the next contributor) to be able to scan this file and know
> exactly what shipped, when, and where to look for it in code.

Categories used:

- **Added** — new user-visible feature or knob
- **Changed** — change to existing behavior (often a default-value
  shift or API rename)
- **Fixed** — bug fix
- **Tests** — new test coverage worth calling out
- **Docs** — documentation/notebook updates

---

## [Unreleased]

### Added
- `docs/market_models.md` — standalone reference document covering
  the full landscape of retirement Monte Carlo market models, the
  five industry segments and their conventional choices, the
  empirical-value-per-line analysis behind which models we ship,
  and which models we deliberately skipped (AR(1), regime-switching,
  GARCH, Wilkie). Linked from `docs/scenario_guide.md`.
- `CHANGELOG.md` (this file) — feature/fix tracking.

### Changed
- `tax_optimizer_standalone.ipynb` and `tax_optimizer_standalone_vijay.ipynb`:
  - Imports cell now pulls in `HistoricalSequenceModel`, `CMA_PRESETS`,
    `lognormal_from_cma`.
  - §8 Monte Carlo: broadened from a 2-way comparison (lognormal vs.
    bootstrap) to a 4-way comparison adding the Vanguard 2025 CMA preset
    and `HistoricalSequenceModel`.
  - New §14 "Market-model deep dive" — CMA preset cheat sheet, CAPE
    conditioning impact table, 5-way deterministic bake-off.
  - New §15 "Tier B feature cheat sheet" — single-cell reference for
    every Tier B `Config` / `Inputs` knob.
  - Recap markdown cell now versioned **v2 → v4** with rows for every
    Tier A bug fix, every Tier B modeling gap, and every v4 market
    enhancement.
- `docs/scenario_guide.md`: added `historical_sequence` and `cma`
  shortcut examples; renamed "The other two market kinds" →
  "The other three market kinds"; cross-link to `market_models.md`.

### Fixed
- Wrong column name in notebook §14c bake-off (used `irmaa_cost`,
  actual column is `irmaa`).

---

## [v4] — Market-modeling deep dive (2026-05-09, commit `3ac192f`)

### Added
- **Equity-bond correlation in `LognormalModel`** — new field
  `equity_bond_corr: float = 0.10`. `begin_path()` now draws from a
  bivariate normal via `rng.multivariate_normal`, so equity and bond
  returns are no longer treated as independent. Validates `[-1, 1]`
  in `__post_init__`. Marginals (μ, σ) are unchanged by correlation
  — verified empirically with N=100k draws.
- **CAPE conditioning** on `LognormalModel.equity_mu` — new fields
  `cape_today: float | None = None` and `cape_long_run: float = 16.5`,
  plus public `effective_equity_mu()` for diagnostics. When set,
  scales `equity_mu` by `cape_long_run / cape_today` (Shiller's
  starting-yield approximation). Vol is intentionally left untouched.
- **`HistoricalSequenceModel`** (4th `MarketModel`) — picks one
  contiguous N-year slice of 1928–2023 history per Monte Carlo path
  (Bengen / FIRECalc methodology). Distinct from `BootstrapModel`,
  which stitches short blocks. Yields 67 distinct possible 30-year
  paths from 96 years of data. Raises a clean `ValueError` when
  horizon exceeds available history.
- **`CMA_PRESETS` table** + **`lognormal_from_cma(name, **overrides)`**
  helper — five canned forward-looking assumption profiles:
  `vanguard_2025`, `jpm_ltcma_2025`, `horizon_2025`,
  `historical_1928_2023`, `historical_1985_2023`. Each preset bundles
  μ_eq, σ_eq, μ_b, σ_b, ρ(eq,b). Per-knob overrides via kwargs.
- Public API exports in `tax_optimizer/__init__.py`:
  `HistoricalSequenceModel`, `CMA_PRESETS`, `lognormal_from_cma`.
- Scenario JSON loader/serializer (`tax_optimizer/scenario.py`):
  - `kind: "historical_sequence"` for the new model.
  - `cma: "vanguard_2025"` shortcut (implies `kind: "lognormal"`,
    pulls preset, allows per-knob overrides).
  - Round-trips `equity_bond_corr`, `cape_today`, `cape_long_run`.
  - Helpful error for unknown CMA presets and `cma` paired with a
    non-lognormal `kind`.

### Tests
- 33 new tests in `tests/test_market_models.py` covering correlation
  marginals + propagation, CAPE math (including no-op cases),
  historical-sequence contiguity & bounds, preset table integrity,
  and full scenario JSON round-trips for every new field.

---

## [v3 — Tier B] — High-ROI modeling gaps (2026-05-09, commit `b64bbbb`)

### Added
- **State income tax module** (`tax_optimizer/tax/state.py`):
  - `StateTaxRegime` dataclass — encapsulates state-specific tax
    parameters (brackets MFJ/Single, std deductions, LTCG treatment,
    SS taxability, retirement-income exclusions, HSA deductibility).
  - `state_tax(...)` — computes state income tax for a given year,
    handling state-specific adjustments.
  - Bundled presets: `STATELESS`, `CA`, `NY`, `IL`, `MA`.
  - `lookup(name)` for retrieving regimes by name.
  - `Config.state_regime`, `state_regime_change_year_offset`,
    `state_regime_change_target` for state-to-state moves mid-horizon.
  - State brackets/deductions inflate annually via
    `bracket_indexing_rate`.
- **Bequest tax in terminal-NW objective** —
  `Config.heir_marginal_rate: float = 0.22` discounts terminal pretax
  AND HSA balances when projecting to heirs (HSA inherited by
  non-spouses is taxed like pretax). Threaded through
  `metrics.py`, `optimizer.py`, `sensitivity.py`, `monte_carlo.py`,
  and `__main__.py`. `terminal_after_tax_nw()` keeps a `marginal_rate=`
  alias for backward compatibility.
- **IRA contributions** (`tax_optimizer/ira.py`) — three priority-
  ordered paths per spouse, capped by the annual IRA limit:
  - Deductible Traditional IRA (reduces `wages_box1`).
  - Direct Roth IRA (auto-phases out by MAGI).
  - Backdoor Roth (non-deductible Traditional + immediate conversion,
    pro-rata aware over the existing pretax balance).
  - New `Inputs` fields: `spouse_a/b_traditional_ira_contrib`,
    `_roth_ira_contrib`, `_backdoor_roth`.
  - New `limits.py` constants: `IRA_CONTRIBUTION_LIMIT`,
    `IRA_CATCH_UP_50`, `IRA_CATCH_UP_AGE`, plus Roth IRA MAGI
    phase-out thresholds and helpers `ira_contribution_cap(age)`,
    `roth_ira_phaseout_factor(magi, filing_status)`.
- **Mega-backdoor Roth** (after-tax 401k → in-plan Roth conversion):
  - `Inputs.spouse_a/b_mega_backdoor_enabled` and `_after_tax_401k_pct`.
  - Capped by §415(c) overall annual additions limit
    (`SECTION_415C_LIMIT = 70_000.0` in `limits.py`), minus other
    401(k) contributions.
  - Routes directly to `state.roth`; reduces `cash_inflow`.
- **HSA after 65** — once either spouse hits 65, HSA becomes a 4th
  retirement bucket. The deficit cascade now draws taxable →
  Roth → HSA (post-65, ordinary income, no penalty) → pretax.
  `cover_deficit()` accepts `hsa_already` and `hsa_unlocked` kwargs.
- Example scenarios `scenarios/example01.json` and `example02.json`
  populated with all new Tier B knobs.
- Documentation: full sections added to `docs/scenario_guide.md`
  for every new config field.

### Tests
- 44 new tests in `tests/test_tier_b.py` covering state regimes
  (lookup, per-state rules, simulator integration, regime change),
  bequest tax with various heir rates, IRA cap arithmetic, IRA
  allocation (eligibility, cap priority, phase-out, pro-rata
  backdoor), simulator integration of all IRA paths, mega-backdoor
  enablement and §415(c) cap, HSA-after-65 deficit-cascade behavior,
  and a kitchen-sink integration test with all knobs enabled.
- Total test count: 211 → 255.

### Fixed
- Illinois `state_tax` was double-zeroing `ss_taxable_federal` in the
  retirement-income full-exclusion branch when SS was already exempt
  by `regime.ss_taxable_fraction == 0`. Now uses `state_ss_taxable`
  in the exclusion arithmetic.

---

## [v3 — Tier A] — Correctness fixes & contribution-cap accuracy (2026-05-09, commits `0d191e4` + `ef4e345` + `00db500`)

### Added
- **`limits.py` module** (commit `00db500`) — IRS 401(k) elective
  deferral limit, age-50 catch-up, age-60 super-catch-up (SECURE 2.0),
  and HSA family-coverage limit + age-55 catch-up. Helper
  `elective_deferral_cap(age)` and `hsa_contribution_cap(age)`.
- **Tax-efficient deficit cascade** (commit `00db500`) — new
  `cover_deficit()` in `withdrawals.py` draws from accounts in
  order: taxable → Roth → pretax (Roth conversion math is
  pro-rata-aware). Replaces the prior "drain pretax first" behavior
  which over-stated lifetime tax in many scenarios.
- **Employer 401(k) match modeling** (commit `ef4e345`) — new
  `Inputs` fields per spouse: `employer_match_rate` and
  `employer_match_max_pct` (i.e. "100% match up to 6% of salary").
  Match is added to the pretax balance during working years; action
  report displays the per-spouse breakdown.
- **Social Security annual COLA** (commit `0d191e4`) — `Config.ss_cola_rate`
  inflates each spouse's PIA annually after their claim year.
- **Per-spouse SS claim age & FRA** (commit `0d191e4`) — `Inputs.ss`
  upgraded from a single `claim_age` / `fra` to per-spouse fields.
  Actuarial scaling for early/delayed claims applies to each spouse
  independently.
- **Bracket indexing rate** (commit `0d191e4`) — `Config.bracket_indexing_rate`
  inflates federal ordinary-income brackets, LTCG brackets, std
  deduction, AND IRMAA tier thresholds annually. Closes the bug
  where retirees in 2055 were being taxed off 2026 nominal brackets.
- **Explicit yields on taxable-account holdings** (commit `0d191e4`)
  — `Config.taxable_equity_div_yield` and
  `Config.taxable_bond_interest_yield` make the tax drag on a taxable
  brokerage realistic instead of relying on price-only returns.
- **FICA / payroll tax modeling** (commit `0d191e4`) — employee-side
  OASDI (capped at SS wage base, indexed annually), Medicare, and
  Additional Medicare on W-2 wages. Threaded through `simulate()`
  and the action report.
- **Spousal pretax rollover on death** (commit `0d191e4`) — when a
  spouse dies, their `state.spouse_*_pretax` balance rolls over to
  the surviving spouse's pretax bucket, preserving the asset and
  rebasing future RMDs to the survivor's age.
- **Roth conversion eligibility expansion** (commit `0d191e4`) — the
  bracket-fill conversion strategy is no longer restricted to the
  pre-RMD "gap years" between retirement and SS/RMD start. The
  optimizer now sees conversions as available across the full
  horizon (still RMD-first within any given year).

### Changed
- `simulator.py` enforces all contribution caps from `limits.py`
  rather than silently letting the user over-contribute.
- README updated with new example-scenario paths and CLI usage notes
  (commit `00db500`).

### Tests
- New tests for `limits.py`, the deficit cascade, employer match,
  SS COLA + per-spouse claim, bracket indexing, and FICA. Total
  pre-Tier-B test count: ~211.

---

## [v2] — Stochastic baseline (pre-existing)

The capabilities below were already in the package before this
changelog started; they're the foundation that everything above
sits on top of. Kept here as a reference for the notebook recap.

### Already present
- `LognormalModel` (univariate, pre-correlation), `BootstrapModel`,
  `DeterministicModel` market models with `simulate_paths()` Monte
  Carlo and CVaR / `prob_success` metrics.
- `Mortality(year_of_death_a, year_of_death_b, ...)` with widow's-
  penalty filing-status flip, SS / pension scaling on death.
- `AssetLocation` per-account equity/bond split (bonds-in-pretax,
  equities-in-Roth) and `AssetLocation.uniform(equity_pct=...)`.
- `SpendingProfile.retirement_smile(...)` with per-age multipliers,
  `LumpEvent`s, and `LongTermCareShock`.
- `TCJA_EXTENDED` / `PRE_TCJA_2017` / `SUNSET_2026` regime swaps via
  `Config.tax_regime`, `regime_change_year_offset`,
  `regime_change_target`.
- `optimize_s3(cfg, inputs, objective='terminal' | 'cvar' | 'p_success')`
  via differential evolution over 401(k) Roth split + Roth
  conversion target bracket + bracket-fill target.
- `tornado_sensitivity()` for one-knob-at-a-time leverage analysis.
- `build_action_report()` Markdown/HTML/PDF report builder shared
  between the notebook and the `python -m tax_optimizer` CLI.

---

## How to update this changelog

1. **When you add a feature or fix a bug:** add a one-liner under
   the appropriate `### Added` / `### Changed` / `### Fixed` /
   `### Tests` / `### Docs` heading inside `[Unreleased]`. Mention
   the new symbol/field/file by name and (in one phrase) what it does.
2. **When cutting a release:** rename the `[Unreleased]` section
   header to `[vX.Y.Z] — <short-name> (YYYY-MM-DD, commit <sha>)`,
   then create a fresh empty `[Unreleased]` block at the top.
3. **Don't reorder old entries.** History should read in commit
   order (newest at top).
4. **Be specific about *where* the change lives** — file path or
   module name. The point is for a future reader to grep their way
   to the implementation in seconds.
