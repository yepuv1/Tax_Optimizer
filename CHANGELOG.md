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

(no entries — last release `v5` is current.)

---

## [v5] — Tier C (correctness + healthcare + optimizer scope) (2026-05-10)

A 17-item batch organized into three sub-tiers:
- **C-A** (9 correctness bugs)
- **C-B** (5 high-ROI modeling additions, including ACA premium tax credit)
- **C-C** (3 optimizer-scope extensions)

Total ≈ 1,100 LoC of source + ≈ 700 LoC of tests. Test count: **288 → 318
passing** (the 30 new tests live in [`tests/test_tier_c.py`](tests/test_tier_c.py)).

### Fixed *(Tier C-A — correctness bugs)*
- **TC-1 — Year-of-death MFJ filing status**
  ([`tax_optimizer/mortality.py`](tax_optimizer/mortality.py)). The
  IRS treats the calendar year a spouse dies as MFJ on the joint
  return. Pre-Tier-C, `Mortality.filing_status()` flipped to single
  immediately at `year_of_death`. New helper
  `Mortality.is_year_of_death()` lets `filing_status()` keep MFJ for
  exactly the year of death.
- **TC-2 — Survivor SS at age 60 from deceased's record**
  ([`tax_optimizer/simulator.py`](tax_optimizer/simulator.py) ~370–
  395). SSA lets a widow(er) collect survivor benefits on the
  deceased's record starting at age 60, regardless of their own
  claim age. Pre-Tier-C, `ssn_income` was forced to $0 if the
  survivor hadn't reached their own claim age — silently dropping
  5–10 years of benefits in any scenario where the higher-earner
  died first. Now uses `max(own_benefit_if_eligible,
  survivor_of_deceased_if_60+)`.
- **TC-3 — Deficit cascade tax base**
  ([`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)
  ~720–740). `cover_deficit` is now passed `final_kwargs` (with
  primary withdrawals + conversions stacked) instead of
  `base_kwargs`. The internal `_solve_pretax_for_net` /
  `_solve_taxable_for_net` solvers compute marginal tax via a
  base-tax delta; passing pre-stack kwargs **understated** the
  marginal bracket on every cascade leg in any year with a primary
  RMD or conversion.
- **TC-4 — Additional Medicare 0.9% MFJ joint threshold**
  ([`tax_optimizer/payroll.py`](tax_optimizer/payroll.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)). New
  `fica_household(wages_a, wages_b, *, filing_status)` reconciles
  the Form-8959 Additional Medicare 0.9% surcharge at the
  household level using the correct filing-status threshold
  ($250k MFJ vs. $200k single). Pre-Tier-C, two $180k W-2s
  produced $0 of Additional Medicare (each below the per-W-2
  $200k withholding threshold) when the actual joint liability
  was 0.9% × ($360k - $250k) = $990. `fica_employee()` is
  preserved for employer-withholding modeling.
- **TC-5 — Roth IRA MAGI estimate uses prior-year AGI**
  ([`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)
  ~245–255, [`tax_optimizer/state.py`](tax_optimizer/state.py)).
  New `state.prior_agi` field carries last year's AGI forward.
  The IRA MAGI phase-out estimate now uses
  `max(state.prior_agi, current_year_wages)` instead of wages
  alone, fixing the case where a high-portfolio retiree's
  $300k+ MAGI silently allowed direct Roth contributions.
- **TC-6 — Backdoor Roth pro-rata is IRA-aggregate, not IRA + 401(k)**
  ([`tax_optimizer/state.py`](tax_optimizer/state.py),
  [`tax_optimizer/ira.py`](tax_optimizer/ira.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)). New
  `state.spouse_*_pretax_ira` IRA-only sub-balances correctly model
  IRC §408(d)(2): pro-rata aggregates only IRA balances, NOT 401(k).
  Pre-Tier-C, a spouse with $0 pretax IRA but $500k 401(k) had a
  99% taxable backdoor; now correctly $0 taxable. Drains
  (RMDs, conversions, withdrawals) reduce the IRA-only sub-balance
  pro-rata to its share of the total pretax bucket.
- **TC-7 — Fixed Roth conversion reserves RMD bucket**
  ([`tax_optimizer/conversion.py`](tax_optimizer/conversion.py)).
  `planned_roth_conversion()` now caps each spouse's conversion
  at `pretax_balance - rmd_amount` so a fixed
  `roth_conversion_amount` can no longer leave the pretax bucket
  too low for `withdraw_for_need` to satisfy the RMD. New `rmd_a`
  / `rmd_b` kwargs.
- **TC-8 — Action report shows per-spouse SS start_age**
  ([`tax_optimizer/report.py`](tax_optimizer/report.py) ~95–105).
  Household snapshot now uses `inputs.ss.effective_start_age_a` /
  `effective_start_age_b` instead of the legacy
  `inputs.ss.start_age` (which was the same value for both
  spouses).
- **TC-9 — Caveats block doesn't claim "federal-only"**
  ([`tax_optimizer/report.py`](tax_optimizer/report.py) ~415–435).
  Updated to reflect Tier B (state tax shipped) and Tier C-B
  (Medicare + ACA + step-up shipped).

### Added *(Tier C-B — high-ROI modeling)*
- **TC-10 — Base Medicare Part B + Part D premiums**
  ([`tax_optimizer/config.py`](tax_optimizer/config.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)). New
  `cfg.medicare_base_b_d_premium` (default $2,500/yr today's
  dollars per enrolled spouse). Inflated by `cfg.inflation`. New
  DataFrame column `medicare_base_premium`.
- **TC-11 — IRMAA 2-year MAGI lookback**
  ([`tax_optimizer/state.py`](tax_optimizer/state.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)). New
  `cfg.irmaa_lookback_years` (default `2`, the SSA-published
  rule). The simulator maintains `state.prior_agi` /
  `state.agi_lag_2` and reads the appropriate lag for IRMAA
  computation. Set to `0` to recover the pre-Tier-C (current-year
  AGI) behavior. New DataFrame column `irmaa_lookback_agi`.
- **TC-12 — Pre-Medicare healthcare cost knob**
  ([`tax_optimizer/config.py`](tax_optimizer/config.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)). New
  `cfg.health_pre65_today` (default 0). Charged each year either
  spouse is alive AND below 65; per-spouse share is
  `health_pre65_today × (n_pre_medicare / 2)`. Bypassed when ACA
  is enabled (the benchmark premium replaces it). New DataFrame
  column `health_pre65`.
- **TC-13 — ACA premium tax credit**
  ([`tax_optimizer/config.py`](tax_optimizer/config.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)). New
  `cfg.aca_enabled` (default `false`),
  `cfg.aca_benchmark_premium_per_adult` (default $14,000/yr),
  `cfg.aca_max_contrib_pct` (default 0.085 — post-IRA-2022 8.5%
  cap). Models the **post-IRA-2022 enhanced subsidies**: cap
  premium contribution at 8.5% of MAGI; credit
  `max(0, benchmark - cap)` against cash outflow. Cliff-free for
  all incomes ≥ 150% FPL. Approximations: MAGI ≈ AGI; no FPL ×
  household-size lookup (Tier D). New DataFrame columns
  `aca_benchmark_premium`, `aca_apt_credit`.
- **TC-14 — Step-up in basis on first spouse's death**
  ([`tax_optimizer/config.py`](tax_optimizer/config.py),
  [`tax_optimizer/simulator.py`](tax_optimizer/simulator.py)). New
  `cfg.stepup_at_first_death` (default `false`). When true,
  resets `state.cumulative_basis = state.taxable` at the start of
  the year of first death (community-property full step-up;
  CA/WA/ID/NM/AZ/LA/NV/TX/WI). Common-law half-step-up is in
  Tier D. New DataFrame column `cumulative_basis`.

### Added *(Tier C-C — optimizer scope)*
- **TC-15 — Mega-backdoor 401(k) % in optimizer decision vector**
  ([`tax_optimizer/optimizer.py`](tax_optimizer/optimizer.py)). New
  `_build_decision_vector_meta(cfg, inputs)` discovers active
  decision axes dynamically. When
  `inputs.spouse_*_mega_backdoor_enabled=True`, the decision vector
  grows with `mega_backdoor_pct_a` / `mega_backdoor_pct_b`
  (continuous, 0..1 each).
- **TC-16 — Per-spouse SS claim-age axis**
  ([`tax_optimizer/optimizer.py`](tax_optimizer/optimizer.py),
  [`tax_optimizer/config.py`](tax_optimizer/config.py)). New
  `cfg.optimize_ss_claim_age` flag (default `false`). When true,
  adds `ss_claim_age_a` / `ss_claim_age_b` axes (discrete grid
  `{62, 65, 67, 70}`). Major lever for asymmetric couples.
- **TC-17 — `mc_seed` thread-through in MC objectives**
  ([`tax_optimizer/optimizer.py`](tax_optimizer/optimizer.py)). New
  `mc_seed` parameter on `make_objective`, `optimize_household`
  (formerly hard-coded to `42` inside `_cvar_objective` /
  `_p_success_objective`). Pin a single value for reproducibility;
  sweep to detect overfitting to one MC draw.

### Changed
- `Config` gained 8 new fields (see Tier C-B/C entries above).
  Scenario JSON loader auto-recognizes them via the existing
  `fields(Config)` reflection — no `scenario.py` changes needed.
- `optimize_s3` is preserved as a backward-compat alias to the
  newly-renamed `optimize_household`.
- `simulator.py` row dict gained columns: `medicare_base_premium`,
  `health_pre65`, `aca_benchmark_premium`, `aca_apt_credit`,
  `irmaa_lookback_agi`, `cumulative_basis`,
  `fica_oasdi`, `fica_medicare`, `fica_additional_medicare`.
  Removed: `fica_a`, `fica_b` (replaced by household-level
  reconciliation; per-W-2 data still available via
  `fica_employee()`).

### Tests
- `tests/test_tier_c.py` (NEW) — 30 tests covering every Tier C-A
  bug fix, every Tier C-B feature, and every Tier C-C optimizer
  axis.
- `tests/test_mortality.py` updated for TC-1 (year-of-death MFJ).
- `tests/test_simulator.py` updated for TC-1 (year-of-death MFJ).

### Docs
- `docs/scenario_guide.md`: 7 new sections covering all Tier C
  config knobs (Medicare base premium, pre-65 healthcare, IRMAA
  lookback, ACA suite, step-up, optimize_ss_claim_age).
- `scenarios/example01.json` and `scenarios/example02.json`:
  added Tier C config block (set to backward-compat defaults so
  existing scenarios behave identically).

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
