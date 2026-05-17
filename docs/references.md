# References

This page maps every assumption, simplification, and audited finding
in `tax_optimizer` to the underlying statute, regulation, IRS
publication, or state guidance it derives from. Use it when:

- You want to verify that the simulator's tax math agrees with the
  Internal Revenue Code or a state tax code.
- You're reviewing an audit finding (the per-finding map at the
  bottom links every flagged issue to its primary citation and the
  file/lines where the gap lives in the codebase).
- You want to know which simplifications were made deliberately vs.
  which are open gaps.

The page closes with a "How to update this file" note: when a finding
is fixed, mark it Done and link to the test that locks the new
behavior.

---

## Contents

- [1. Federal income tax](#1-federal-income-tax)
- [2. State income tax](#2-state-income-tax)
- [3. Healthcare, IRMAA, and ACA](#3-healthcare-irmaa-and-aca)
- [4. Annuity-specific rules](#4-annuity-specific-rules)
- [5. Pension lump-sum and rollovers](#5-pension-lump-sum-and-rollovers)
- [6. RMDs and SECURE 2.0](#6-rmds-and-secure-20)
- [7. Stepped-up basis and inheritance](#7-stepped-up-basis-and-inheritance)
- [8. Filing status detail](#8-filing-status-detail)
- [9. Per-finding citation map](#9-per-finding-citation-map)
- [10. Open questions and interpretive choices](#10-open-questions-and-interpretive-choices)
- [How to update this file](#how-to-update-this-file)

---

## 1. Federal income tax

| Topic | Primary citation | Secondary / explanatory | Relevant code |
|---|---|---|---|
| Ordinary brackets | IRC §1(a)–(d), §1(j) (TCJA rates) | Rev. Proc. 2024-40 (2025 brackets), Rev. Proc. 2025-32 (2026 brackets) | `tax_optimizer/tax/regimes.py` |
| Standard deduction | IRC §63(c) | Rev. Proc. 2024-40 §3.16; Rev. Proc. 2025-32 | `tax_optimizer/tax/regimes.py` |
| Capital gains preferential rates (0% / 15% / 20%) | IRC §1(h) | Pub 550 ch. 4 | `federal_tax` LTCG stacking in `tax_optimizer/tax/federal.py` |
| Net Investment Income Tax (3.8%) | IRC §1411 | Reg §1.1411-1 to §1.1411-10; Form 8960 | `niit_*` block in `tax_optimizer/tax/federal.py` |
| AMT | IRC §55–§59; IRC §55(d) (exemption); §55(b)(1) (rates 26% / 28%) | Pub 17 ch. 12; Form 6251 | `amt_*` block in `tax_optimizer/tax/federal.py`, `regimes.py` AMT params |
| SS taxability — provisional income tiers | IRC §86(a)–(c) | Pub 915 worksheet 1; SSA POMS GN 05002 | `social_security_taxable` in `tax_optimizer/tax/federal.py` |
| SS provisional MAGI add-backs (tax-exempt interest, foreign-earned-income exclusion, half of SS) | IRC §86(b)(2) | Pub 915 worksheet 1 line 3 (tax-exempt interest) | currently missing — see finding `ss-provisional-magi` |
| Annuity exclusion ratio | IRC §72(b); §72(c) ("investment in the contract") | Pub 575, Pub 939 ("Simplified Method"); Reg §1.72-4, §1.72-5 | `tax_optimizer/annuity.py`, simulator annuity block |
| Early distribution penalty (10%) — qualified plans | IRC §72(t) | Pub 575 ch. "Tax on Early Distributions"; Form 5329 | `early_distribution_penalty` in `tax_optimizer/tax/federal.py` |
| Early distribution penalty (10%) — non-qualified annuities | IRC §72(q) | Pub 575 | same — currently a flat 10% with no §72(t)(2) exception list |
| FICA — OASDI 6.2% to wage base | IRC §3101(a), §3121(a)(1) | SSA Press Release "2026 Social Security Changes" — wage base $176,100 | `tax_optimizer/payroll.py::OASDI_WAGE_BASE_2026` |
| FICA — Medicare 1.45% (uncapped) | IRC §3101(b)(1) | Pub 15 §11 | `tax_optimizer/payroll.py::MEDICARE_RATE` |
| Additional Medicare 0.9% above $250k MFJ / $200k Single | IRC §3101(b)(2); IRC §3102(f) | Form 8959; Reg §1.1411-4 (NIIT interaction) | `tax_optimizer/payroll.py` Form 8959 block |
| 401(k) elective deferral limit | IRC §402(g)(1)(B) | IRS Notice 2024-80 (2025 = $23,500); 2026 limits notice | `tax_optimizer/limits.py::ELECTIVE_DEFERRAL_LIMIT` |
| Defined-contribution annual addition limit (415(c)) | IRC §415(c)(1)(A) | Notice 2024-80 (2025 = $70,000) | `tax_optimizer/limits.py::SECTION_415C_LIMIT` |
| IRA contribution limit | IRC §219(b) | Notice 2024-80 (2025 = $7,000 / $8,000 catch-up) | `tax_optimizer/limits.py` |
| Roth IRA — eligibility and pro-rata rule on conversions | IRC §408A; IRC §408(d)(2) (aggregation) | Pub 590-A ch. 2; Pub 590-B ch. 1 | `tax_optimizer/ira.py` backdoor + pro-rata logic |
| HSA contribution limit | IRC §223(b) | Rev. Proc. 2024-25 (2025 limits); Pub 969 | `tax_optimizer/limits.py::hsa_family_cap` |
| §401(a)(31) direct rollover | IRC §401(a)(31); IRC §402(c) | Reg §1.401(a)(31)-1; Pub 590-A ch. 2 | pension `lump_sum_mode = "rollover_pretax"` in simulator |

Federal-tax module entry point: `tax_optimizer/tax/federal.py::federal_tax`.

---

## 2. State income tax

The package supports CA, NY, IL, MA out-of-the-box and a `STATELESS`
no-op regime. Each state's `StateTaxRegime` (in
`tax_optimizer/tax/state.py`) carries its own bracket table, LTCG
preference flag, SS-taxability fraction, and retirement-income
exclusion.

| State | Brackets | LTCG | SS taxability | Pension/IRA exclusion | Citation |
|---|---|---|---|---|---|
| CA | RTC §17041 progressive (1%–12.3%) | Taxed as ordinary income (no preferential rate) | Fully exempt | None — pension and IRA fully taxed | RTC §17041; FTB Pub 1005; FTB Pub 1001 |
| NY | Tax Law §601 progressive (4%–10.9%) | Taxed as ordinary income | Fully exempt (Tax Law §612(c)(3)(i)) | $20,000 per filer (per spouse on a joint return) at age 59½+ for pension + IRA + Roth-conversion income | NY Tax Law §612(c)(3-a); TSB-M-12(2)I |
| IL | 35 ILCS 5/201 flat 4.95% | Taxed as ordinary | Fully exempt | Full subtraction for federally-qualified pension, IRA, Roth-conv, SS | 35 ILCS 5/203(a)(2)(F); IL Pub 120 |
| MA | Ch. 62 §2 flat 5% (plus 4% surtax on income > $1M) | Long-term taxed at 5% (short-term at 8.5%) | Fully exempt (Ch. 62 §2(a)(2)(E)) | None for IRA / private pension; full exclusion for federal/MA pensions | M.G.L. Ch. 62 §2; DOR TIR 02-21 |

State payroll / disability:

| Item | Rate | Citation |
|---|---|---|
| CA SDI (employee) | 1.1% (2024); 1.2% (2025); **0.9% (2026)** | CA UI Code §984; CA EDD DB-1101 / "Voluntary Plan ER Year-End Adjustment Form"; SB 951 (2022) removed the SDI taxable wage cap effective 2024 |
| CA SDI wage cap | None (uncapped post-SB 951) | SB 951 (2022) |

Currently the code hard-codes CA SDI at 1.1% with no year schedule —
see finding `ca-sdi-refresh`.

State-tax module entry point:
`tax_optimizer/tax/state.py::state_tax`.

---

## 3. Healthcare, IRMAA, and ACA

| Topic | Primary citation | Secondary | Relevant code |
|---|---|---|---|
| Medicare Part B premium | 42 USC §1395r(a) | CMS 2025 Medicare Parts A & B Premiums announcement | `cfg.medicare_base_b_d_premium` |
| IRMAA Part B surcharge | 42 USC §1395r(i) | CMS 2025 IRMAA tier announcement; SSA POMS HI 01101.020 | `tax_optimizer/tax/irmaa.py` |
| IRMAA Part D surcharge | 42 USC §1395w-113(a)(7) | CMS 2025 IRMAA tier announcement | `tax_optimizer/tax/irmaa.py` |
| IRMAA two-year MAGI lookback | 42 USC §1395r(i)(4) | SSA POMS HI 01101.030 | `cfg.irmaa_lookback_years` (default 2) in `Config` |
| ACA Premium Tax Credit | IRC §36B | Form 8962; Reg §1.36B-3 | `aca_*` block in simulator |
| HSA — qualified medical expenses | IRC §223(d)(2) | Pub 502; Pub 969 | HSA pay-down block in simulator |
| HSA — qualified post-65 | IRC §223(f)(4)(C) (no 20% penalty after 65) | Pub 969 | `hsa_unlocked = max(a_age, b_age) >= 65` in simulator |
| LTC insurance premiums (deductible threshold) | IRC §213(d)(10) | Pub 502 | not currently modeled — LTC shock is treated as out-of-pocket spending |
| Disability income exclusion (rare) | IRC §104(a)(3) | Pub 525 | not modeled |

---

## 4. Annuity-specific rules

| Topic | Primary citation | Secondary | Relevant code |
|---|---|---|---|
| Exclusion ratio (general rule) | IRC §72(b)(1) | Pub 939 "General Rule"; Reg §1.72-4 | `tax_optimizer/annuity.py::exclusion_ratio` |
| Simplified method (post-Nov 1996 qualified plans) | IRC §72(d) | Pub 575 "Simplified Method" worksheet | not used directly — model takes user-supplied `expected_payout_years` |
| Investment in the contract | IRC §72(c) | Pub 939 | `inputs.annuity.cost_basis` |
| Distribution before annuity start date (partial surrender) | IRC §72(e) | Pub 575; Reg §1.72-11 | partial-surrender distribution prior to `start_age` not modeled (only full lump sum at `start_age`) |
| 10% additional tax on premature non-qualified distributions | IRC §72(q) | Pub 575 | `early_distribution_taxable` kwarg in `federal_tax`; flat 10% |
| 10% additional tax — qualified plan early distribution | IRC §72(t) | Pub 575 ch. "Tax on Early Distributions"; Form 5329 | same code path; §72(t)(2) exception list (SOSEPP, disability, first-home, etc.) not implemented |
| Variable annuity unit accounting | Reg §1.72-2(b)(3) | Pub 575 | not modeled (treated as fixed annuity with deterministic growth) |
| Period certain / refund feature adjustment | Reg §1.72-7 | Pub 939 | not modeled |
| Prohibition on rolling NQ annuity into qualified plan | IRC §408(d)(3); IRC §72(e)(11) | Pub 590-A | enforced by `Inputs.__post_init__` (raises if `tax_kind="non_qualified"` + `lump_sum_mode="rollover_pretax"`) |

The model uses the user-supplied `expected_payout_years` rather than
the IRS Pub 939 actuarial tables. This front-loads the tax-free
return of basis if `expected_payout_years` is set materially lower
than the contract's true payout horizon (e.g. life annuity).

---

## 5. Pension lump-sum and rollovers

| Topic | Primary citation | Secondary | Relevant code |
|---|---|---|---|
| Eligible rollover distribution | IRC §402(c)(4); §402(c)(8) | Pub 575 | `lump_sum_mode = "rollover_pretax"` |
| Direct trustee-to-trustee transfer | IRC §401(a)(31) | Reg §1.401(a)(31)-1 | model assumes direct rollover (no withholding) |
| Mandatory 20% withholding on indirect distributions | IRC §3405(c)(1) | Pub 575 | not modeled — `lump_sum_mode = "cash"` is treated as voluntary withholding |
| 60-day rollover window | IRC §402(c)(3) | Pub 590-A | not modeled |
| Cash-balance lump-sum value (interest-rate floor) | IRC §417(e)(3); §415(b)(2) | Pub 560; Reg §1.417(e)-1 | `tax_optimizer/pension.py::project_pension_balance` uses the configured `interest_rate` (or §417(e) effective rate) |

---

## 6. RMDs and SECURE 2.0

| Topic | Primary citation | Secondary | Relevant code |
|---|---|---|---|
| Required Beginning Date — age 73 (2023–2032) | IRC §401(a)(9)(C); SECURE 2.0 §107 | Pub 590-B | `tax_optimizer/rmd.py::rmd_amount` |
| RBD shifts to 75 starting 2033 | SECURE 2.0 §107(b) | Pub 590-B 2033 update | not yet schedule-aware in code |
| Uniform Lifetime Table (married, sole-bene spouse not >10 yrs younger) | Reg §1.401(a)(9)-9 Table III (2022 version) | Pub 590-B Appendix B | hard-coded in `tax_optimizer/rmd.py` |
| Joint Life and Last-Survivor table (spouse-bene >10 yrs younger) | Reg §1.401(a)(9)-9 Table II | Pub 590-B Appendix B | not implemented — currently always uses Uniform Lifetime |
| Single Life Expectancy (post-death) | Reg §1.401(a)(9)-9 Table I | Pub 590-B Appendix B | not implemented |
| Roth-401(k) RMD elimination starting 2024 | SECURE 2.0 §325 | Pub 575 | model ignores Roth-401(k) for RMDs (correct on/after 2024) |
| Inherited-IRA 10-year rule | IRC §401(a)(9)(H); SECURE Act 2019 §401 | Notice 2022-53; Notice 2024-35; T.D. 9930 (final regs Aug 2024) | not modeled — household ends at last surviving spouse |
| Eligible designated beneficiary categories (incl. disability, chronically ill) | IRC §401(a)(9)(E) | Pub 590-B; T.D. 9930 | not modeled |
| Surviving-spouse election to be treated as IRA owner | IRC §401(a)(9)(B)(iv) | Pub 590-B; SECURE 2.0 §327 | handled implicitly — surviving spouse keeps the inherited pretax balance and continues RMDs in `simulator.py` mortality block |

Joint Life RMD coverage is the largest open gap — see the missing-features list.

---

## 7. Stepped-up basis and inheritance

| Topic | Primary citation | Secondary | Relevant code |
|---|---|---|---|
| Step-up in basis at death | IRC §1014(a) | Pub 551 | step-up applied to `state.taxable` cost basis at death of the holder spouse, in `simulator.py` |
| Community property double step-up | IRC §1014(b)(6) | Pub 555; CA Family Code §760 | applied unconditionally — simplification (assumes community-property treatment regardless of state) |
| Estate tax exemption | IRC §2010(c) | Form 706 instructions; Notice 2024-80 (2025 = $13,990,000) | not modeled (heir balance is reported pre-estate-tax) |
| Portability of unused exclusion (DSUE) | IRC §2010(c)(4) | Reg §20.2010-2 | not modeled |
| §1014(b)(6) state list | community-property states only (AZ, CA, ID, LA, NV, NM, TX, WA, WI; AK by election) | n/a | model applies double step-up unconditionally — see "Open questions" |

---

## 8. Filing status detail

| Status | Citation | Currently supported? |
|---|---|---|
| Single | IRC §1(c) | yes |
| Married Filing Jointly | IRC §1(a); IRC §6013 | yes (default) |
| Married Filing Separately | IRC §1(d) | no |
| Head of Household | IRC §1(b); §2(b) | no |
| Qualifying Surviving Spouse | IRC §1(a) (uses MFJ brackets); §2(a) | no — model collapses survivor to Single immediately |

The post-mortality "widow(er)'s penalty" is real: the model
correctly switches from MFJ to Single brackets, but it does not
honor the two-year QSS window in IRC §2(a) when there is a
qualifying dependent. This is a known simplification.

---

## 9. Per-finding citation map

Each row below corresponds to an audit finding from the
codebase review. The "Status" column tracks fix progress; when a
finding flips to Done, the row links to the regression test that
locks the new behavior.

### Critical

| ID | Primary citation | Secondary | Where in code | Status |
|---|---|---|---|---|
| `state-tax-annuity-routing` | IRC §61(a)(9) (gross income includes annuities); state-by-state conformity (CA RTC §17071, NY Tax Law §612) | Pub 575; FTB Pub 1005; NY TSB-M-12(2)I | `tax_optimizer/simulator.py` lines 868–886, 1374–1398, 1051–1069; `tax_optimizer/tax/state.py::state_tax` | Pending |
| `annuity-survivor-balance` | IRC §72(b)(2) (joint-and-survivor exclusion); contract terms (`pension_survivor_pct`) | Pub 575 | `tax_optimizer/simulator.py` lines 643–650 | Pending |
| `working-year-cashflow` | n/a — internal accounting (cash available to pay taxes) | n/a | `tax_optimizer/simulator.py` lines 1258–1266 (working-year `cash_inflow`); ~lines 1020–1069 (`tax_paying_capacity`) | Pending |
| `conversion-capacity-double-count` | n/a — internal accounting | n/a | `tax_optimizer/simulator.py` lines 1051–1069 | Pending |
| `ny-pension-exclusion-per-spouse` | NY Tax Law §612(c)(3-a) ($20k retirement exclusion); §612(b)(3) (pension/annuity sourcing) | NY TSB-M-12(2)I; NY IT-225 instructions | `tax_optimizer/tax/state.py` lines 247–270 | Pending |

### High

| ID | Primary citation | Secondary | Where in code | Status |
|---|---|---|---|---|
| `ltc-anchor` | n/a — internal modeling choice | model assumption: LTC shock fires at end-of-life | `tax_optimizer/simulator.py` lines 893–947 | Pending |
| `single-filer-balances` | IRC §1(c) (single brackets); model invariant: spouse_b_* fields ignored when `household_kind="single"` | n/a | `tax_optimizer/inputs.py::Inputs.__post_init__` (line 480+) | Pending |
| `liquidity-guard-negative` | n/a — internal modeling choice | n/a | `tax_optimizer/conversion.py::planned_roth_conversion`; `tax_optimizer/simulator.py` line 1069 | Pending |
| `scenarioerror-mapping` | n/a — UX | n/a | `tax_optimizer/scenario.py`; `dash_app/state.py` | Pending |
| `ca-sdi-refresh` | CA UI Code §984; SB 951 (2022) | CA EDD DB-1101 | `tax_optimizer/payroll.py::state_sdi`; `tax_optimizer/tax/state.py` CA regime | Pending |
| `ss-provisional-magi` | IRC §86(b)(2)(B) (tax-exempt interest add-back); IRC §86(b)(2)(A) (foreign-earned-income / Puerto Rico) | Pub 915 worksheet 1 line 3 | `tax_optimizer/tax/federal.py::federal_tax` ~lines 175–183 | Pending |
| `negative-balance-clamp` | n/a — numerical hygiene | n/a | `tax_optimizer/simulator.py` deficit cascade | Pending |

### Medium and Low (deferred this round)

| ID | Primary citation | Note |
|---|---|---|
| `early-dist-age` | IRC §72(t)(1) ("attained age of 59½") | code uses `< 60` rather than `< 59.5` — off by half a year |
| `state-marginal-strict` | n/a — numerical hygiene | state marginal rate uses `>` instead of `>=`, so zero-income returns slab 2 |
| `report-annuity-timeline` | n/a — UX | annuity / lump-sum / early-dist events not surfaced in markdown action plan |
| `dash-figures-annuity` | n/a — UX | Dash income tab doesn't break out annuity component |
| `dash-report-stale` | n/a — UX | downloaded report can lag behind current form values |
| `dash-objective-knobs` | n/a — UX | optimizer objective + Monte Carlo knob counts not exposed in Dash |
| `tests-fix-column-name` | n/a — test hygiene | known-bad column reference in `tests/test_simulator_single_household.py` |
| `niit-magi-vs-agi` | IRC §1411(d) (defines MAGI); Reg §1.1411-2(c) | NIIT currently keyed off AGI, not MAGI (the difference is the foreign-earned-income exclusion add-back, which is rarely relevant) |
| `amt-depth` | IRC §55–§59 | model handles AMT exemption + phaseout but not preference items beyond exempt-interest from PABs and certain LTCG-vs-AMTI edge cases |

### Missing features (deferred)

| ID | Primary citation | Note |
|---|---|---|
| `joint-life-rmd` | Reg §1.401(a)(9)-9 Table II | spouse-beneficiary > 10 yrs younger needs joint table |
| `inherited-ira-10y` | IRC §401(a)(9)(H); T.D. 9930 | needed for non-spouse heir modeling |
| `mfs-hoh-qss` | IRC §1(b), §1(d), §2 | extra filing statuses |
| `step-up-half-only` | IRC §1014(b)(6) | only community-property states get full double step-up |
| `qcd` | IRC §408(d)(8) | $108k limit (2025) — Qualified Charitable Distribution |
| `sosepp-72t` | IRC §72(t)(2)(A)(iv) | substantially-equal-periodic-payment exception to 10% penalty |
| `state-tax-aca-conformity` | IRC §36B; state ACA bridges (CA AB 133) | state ACA subsidies |
| `social-security-bend-points` | 42 USC §415 | model uses user-supplied PIA, not bend-point projection |

---

## 10. Open questions and interpretive choices

These are deliberate simplifications. Each one has a real-world
edge case that would warrant a future enhancement, but the present
choice is documented and bounded.

1. **Flat 10% early-distribution penalty.** Both §72(t) (qualified)
   and §72(q) (non-qualified) carry exception lists (SOSEPP under
   §72(t)(2)(A)(iv); disability §72(t)(2)(A)(iii); first-home up to
   $10k §72(t)(2)(F); medical expenses > 7.5% AGI §72(t)(2)(B);
   unemployed health insurance §72(t)(2)(D); birth-or-adoption
   distribution up to $5k under SECURE 2.0 §314). The model applies
   a flat 10% with no exception modeling — users wanting a SOSEPP
   plan should compute outside the simulator.

2. **IRMAA tier rounding.** Tier boundaries are honored exactly,
   but partial-year proration (the SSA's "life-changing event" form
   SSA-44, e.g. retirement-year MAGI) is not modeled. Households
   with a one-time MAGI spike will overstate IRMAA in the
   following two years.

3. **AMT depth.** The model implements §55(d) exemption, the
   phase-out, and the 26% / 28% rate split, but does not model
   preference items beyond what's already in AGI. Specified
   private-activity-bond interest (§57(a)(5)) and certain
   accelerated-depreciation adjustments are not included.

4. **§1014(b)(6) double step-up applied unconditionally.** The
   model applies a full step-up on the deceased spouse's basis
   share regardless of state. In separate-property states only
   the deceased spouse's half steps up; the surviving spouse's
   half retains its original basis. Users in non-community-property
   states will see a slightly optimistic taxable-account terminal.

5. **Annuity exclusion ratio uses user-supplied
   `expected_payout_years`.** Pub 939's General Rule uses an
   actuarial life-expectancy lookup; the model simplifies to a
   single user-supplied integer. This is conservative (basis
   recovery completes faster than IRS tables would allow) for any
   `expected_payout_years` shorter than the true horizon.

6. **§72(t)(2) exception for separated employees age 55+.** Not
   modeled. A retiree separating from service at 55+ avoids the
   10% penalty on workplace 401(k) (but not IRA) distributions;
   the model imposes 10% if `a_age < 60` regardless.

7. **NIIT MAGI add-back for FEIE.** §1411(d) requires adding back
   the §911 foreign-earned-income exclusion. Domestic households
   are unaffected. Users with overseas wages should note the gap.

8. **State LTCG conformity.** CA / NY / IL / MA all conform to
   federal-AGI then back out the LTCG preference (i.e. tax LTCG
   at ordinary rate). MA is more nuanced (§63 LTCG vs §63D
   short-term) — the simplified flat 5% treatment will slightly
   misstate MA returns with material short-term gains.

9. **SS bend-point projection.** The model takes a fixed PIA
   today (`monthly_spouse_a` / `monthly_spouse_b`). Households
   with a long pre-claim work history could see their PIA grow
   (or shrink) with continued earnings; we do not re-project
   bend points.

10. **Cash-balance pension §417(e) lump-sum floor.** Pension
    `lump_sum_mode = "cash"` uses the projected cash-balance
    account value at NRD, not the §417(e) actuarially-equivalent
    value. For low-interest-rate plans this slightly understates
    the lump-sum offer.

---

## How to update this file

When a finding from §9 is fixed:

1. Flip its **Status** column from `Pending` to `Done`.
2. Add a parenthetical link to the regression test that locks the
   behavior, e.g. `Done ([test_simulator_annuity.py::test_annuity_survivor_drain](../tests/test_simulator_annuity.py))`.
3. Add the matching `Fixed` entry under `[Unreleased]` in
   [`CHANGELOG.md`](../CHANGELOG.md).
4. If the fix exposes a new public knob (kwarg, scenario-JSON
   field, Dash control), cross-link to the knob in the relevant
   guide ([`scenario_guide.md`](scenario_guide.md),
   [`modeling_guide.md`](modeling_guide.md), or
   [`dashboard.md`](dashboard.md)).

When a new audit finding surfaces:

1. Append a row to §9 under the appropriate severity table.
2. Cite the primary statute/regulation and at least one secondary
   source (IRS Pub, state guidance, etc.).
3. Pin the row to a file path + line range so the next reviewer
   can find the gap quickly.
