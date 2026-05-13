# Scenarios

This directory holds JSON scenario files consumed by `tax_optimizer`.
A scenario is just a `{ "config": {...}, "inputs": {...} }` dict that
overrides defaults on `Config` / `Inputs`. The CLI loads them with
`--scenario PATH`, and the library API uses `apply_scenario(cfg,
inputs, scenario_dict)`.

## Files

| File              | Purpose                                                                                              |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| `template.json`   | **Canonical reference**: every available knob, with its default value. Copy this to start a new plan. |
| `example01.json`  | Mid-career dual-income household with a widow stress test.                                            |
| `example02.json`  | Late-career household with mega-backdoor Roth, pension, and aggressive savings.                       |

Looking specifically for the Roth-conversion or withdrawal-strategy
knobs? Jump to
[Roth conversion & withdrawal knobs](#roth-conversion--withdrawal-knobs).

## Maintenance contract for new knobs

**Whenever a new field is added to `Config`, `Inputs`, or one of the
nested input dataclasses (`StartingBalances`, `CurrentIncome`,
`CurrentContrib`, `PensionInputs`, `SocialSecurity`) or to `Mortality`,
the contributor MUST also:**

1. Add the field (with its default value) to `template.json`.
2. If the new knob is meaningful for the two narrative scenarios,
   reflect it in `example01.json` / `example02.json` too (else they
   silently fall back to defaults).
3. Re-run the drift tests:

   ```
   pytest tests/test_scenario_template.py
   ```

   These tests **will fail loudly** if `template.json` is missing a
   new dataclass field or has a value that disagrees with the
   field's default. The failure message points at the exact field
   and which side to update.

`scenarios/example02.local.json` is a developer's personal scenario,
listed in `.gitignore`. Do not commit personal financial data.

## Polymorphic blocks

`config.market` and `config.spending` are tagged-union shapes. The
template renders ONE representative form (lognormal + smile) so the
file remains a useful copy-paste starting point. Other shapes the
loader accepts:

### `config.market`

```jsonc
// kind: "deterministic" — constant returns
{ "kind": "deterministic", "equity": 0.07, "bond": 0.04 }

// kind: "lognormal" — independent yearly draws (default; CAPE-aware)
{ "kind": "lognormal", "equity_mu": 0.07, "equity_sigma": 0.18,
  "bond_mu": 0.04, "bond_sigma": 0.06, "equity_bond_corr": 0.10,
  "cape_today": null, "cape_long_run": 16.5 }

// kind: "lognormal" with a CMA preset shortcut
{ "kind": "lognormal", "cma": "vanguard_2025" }

// kind: "bootstrap" — block-bootstrap from history
{ "kind": "bootstrap", "block_size": 5 }

// kind: "historical_sequence" — replay contiguous N-year slices
{ "kind": "historical_sequence" }
```

### `config.spending`

```jsonc
// kind: "flat" — constant real spending
{ "kind": "flat", "base_spending": 85000, "inflation": 0.025 }

// kind: "smile" — Blanchett "retirement smile" + LTC shock at end of life
{ "kind": "smile", "base_spending": 85000, "inflation": 0.025,
  "ltc_years": 3, "ltc_annual_today": 80000 }

// kind: "custom" — arbitrary phases / lump events
{ "kind": "custom", "base_spending": 85000, "inflation": 0.025,
  "phases": [{ "age_lo": 65, "age_hi": 200, "multiplier": 1.0,
               "label": "retirement" }],
  "lump_events": [{ "year_offset": 5, "amount_today": 50000,
                    "label": "kitchen-remodel" }],
  "ltc_shock": { "years": 3, "annual_cost_today": 80000 } }
```

### `config.state_regime`

String code: `"stateless"` (default), `"CA"`, `"NY"`, `"IL"`, `"MA"`.

### `config.tax_regime` / `config.regime_change_target`

String code: `"tcja"` (alias `"tcja_extended"`), `"pre_tcja"` (alias
`"pre_tcja_2017"`), `"sunset"` (alias `"sunset_2026"`).

### `config.asset_location`

Either `{ "uniform_equity_pct": 0.7 }` (every account allocated the
same way) OR the per-bucket form shown in `template.json`.

## Roth conversion & withdrawal knobs

A focused reference for the two most-tuned strategy axes. All defaults
below match `template.json`; file references point at the canonical
definition.

### A. Roth-conversion sizing (`Config`)

#### A1. Pick a mode (exactly one)

| Knob                              | Default | What it does                                                                                                                                                                                                                                                  |
| --------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `roth_conversion_target_bracket`  | `0.0`   | **Bracket-fill mode.** Convert just enough to fill the named ordinary bracket (e.g. `0.22`, `0.24`, `0.32`). Self-regulates to zero in high-income working years; fires in any year with positive pretax balance.                                             |
| `roth_conversion_amount`          | `0.0`   | **Fixed-dollar mode.** Convert exactly this many of today's dollars per year, capped at available pretax minus RMD. Gated to the gap window (retired but pre-`rmd_start_age`) so it doesn't detonate working-year tax bills.                                  |

Set ONE of these `> 0` to enable conversions. If both are `0`, no
conversions happen. See `tax_optimizer/conversion.py`
(`planned_roth_conversion`).

#### A2. v6.5 liquidity guards (default-on)

| Knob                                  | Default | What it does                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cap_conversion_by_liquidity`         | `true`  | The simulator bisects the chosen conversion total DOWN if its marginal federal + state tax would exceed the household's tax-paying capacity (earned cash + pension + SS + RMD net of FICA/SDI/spending/healthcare/IRA & MBR contributions, plus the taxable-balance slice). Without this, an aggressive bracket target or oversized fixed amount silently raids the just-funded Roth.   |
| `conversion_taxable_use_ratio`        | `0.5`   | Fraction of the taxable-brokerage balance the sizer is allowed to earmark for conversion tax. `0.0` → tax must come entirely from earned/retirement-income surplus. `1.0` → use full taxable.                                                                                                                                                                                           |
| `protect_roth_in_conversion_years`    | `true`  | The deficit cascade refuses to touch Roth in any year a conversion fires. Liquidity overshoots surface as `unfunded` instead of silently draining Roth dollars.                                                                                                                                                                                                                         |

See `tax_optimizer/config.py` (Roth-conversion liquidity-guards
block).

#### A3. Diagnostic columns (read-only, every row)

- `roth_conversion`, `roth_conversion_a`, `roth_conversion_b` — what got converted.
- `roth_conv_capped_by_liquidity` — `True` when A2 bound and reduced the size.
- `roth_conv_bracket_target` — what the bracket / fixed mode wanted before the liquidity cap.
- `roth_conv_tax_capacity` — the per-year capacity number.

### B. Withdrawal strategy (`Config`)

#### B1. Strategy selection

| Knob                  | Default          | Values                                                  |
| --------------------- | ---------------- | ------------------------------------------------------- |
| `withdrawal_strategy` | `"conventional"` | `"conventional"` / `"proportional"` / `"bracket_fill"`  |

- **`conventional`** (Bengen / Schwab default): RMD first → taxable → pretax → Roth.
- **`proportional`**: pull from taxable / pretax-room / Roth in proportion to balances.
- **`bracket_fill`**: fill bracket headroom with pretax before touching taxable, then Roth as the residual.

See `tax_optimizer/withdrawals.py` (`withdraw_for_need`).

#### B2. Strategy parameters

| Knob                       | Default | What it does                                                                                                                                                       |
| -------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `bracket_fill_target`      | `0.22`  | Bracket cap (as a decimal rate) the `bracket_fill` strategy fills pretax to. Also the implicit ceiling on incidental pretax draws in `proportional`.               |
| `rmd_start_age`            | `75`    | First age the IRS Uniform Lifetime Table is applied. Validated `>= 72`. Lowering to `73` models the current SECURE 2.0 rule.                                       |
| `cap_gains_basis_fraction` | `0.5`   | Initial cost-basis / FMV ratio for the taxable brokerage. The simulator tracks `cumulative_basis` live thereafter (this knob is just the seed).                    |

#### B3. Deficit cascade (no direct knob — the safety net under every strategy)

When `cash_inflow - taxes - net_need < 0`, `cover_deficit` pulls in
this order:

**taxable** (LTCG-rate gross-up) → **Roth** (skipped in conversion
years per A2) → **HSA after age 65** (ordinary gross-up, no penalty)
→ **pretax** (ordinary gross-up).

Anything left over shows up as `unfunded` on that row.

See `tax_optimizer/withdrawals.py` (`cover_deficit`).

### C. Knobs that indirectly shape both

| Knob                                                                                                                | Why it matters for conversion / withdrawal                                                                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tax_regime` / `regime_change_year_offset` / `regime_change_target`                                                 | TCJA-extended vs. SUNSET shifts marginal rates ~3–4 pts above ~$100k AGI. A conversion that fills the 22% bracket today fills the 25% bracket post-sunset — bracket-fill math adjusts automatically once the regime swaps.                            |
| `state_regime` / `state_regime_change_*`                                                                            | High-tax states (CA, NY) add 9.3% / 6.85% marginal bite to conversion tax. The v6.5 capacity bisection includes state tax. CA → FL move post-retirement is a classic conversion-timing optimization.                                                  |
| `bracket_indexing_rate`                                                                                             | Brackets / std-deduction / IRMAA tiers index by this rate. `0.0` freezes them (worst-case "Congress freezes brackets" stress).                                                                                                                        |
| `taxable_equity_div_yield`, `taxable_bond_interest_yield`, `taxable_equity_qualified_fraction`                      | Portfolio yields hit AGI every year and eat into bracket headroom available for conversions.                                                                                                                                                          |
| `medicare_base_b_d_premium`, `health_pre65_today`, `irmaa_lookback_years`, `aca_enabled` / `aca_*`                  | Healthcare costs are subtracted from `tax_paying_capacity`. IRMAA tier hops are a real cost of aggressive same-year conversions (2-year lookback means they show up later).                                                                           |
| `heir_marginal_rate`                                                                                                | The optimizer weights pretax → Roth conversions against this rate. Higher heir rate (32%+) → more aggressive conversions. `0.0` → roughly indifferent.                                                                                                |
| `stepup_at_first_death`                                                                                             | If `true`, taxable basis resets at first spouse's death, making "leave it taxable" cheaper for heirs → mildly de-emphasizes conversions.                                                                                                              |
| `mortality.year_of_death_a/b` + `mortality.pension_survivor_pct`                                                    | Conversion timing changes when one spouse dies (filing status flips to single, brackets compress) — sized automatically against the active regime.                                                                                                    |
| `optimize_ss_claim_age`                                                                                             | When `true`, the optimizer also chooses SS claim ages; later claim → longer gap window with low ordinary income → more bracket headroom for conversions.                                                                                              |

### D. Inputs that gate conversion / withdrawal timing

| Field (on `Inputs` or nested)                            | Effect                                                                                                                                                            |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `spouse_a_retire_age` / `spouse_b_retire_age`            | Fixed-amount conversions only fire when the eligible spouse is retired AND pre-RMD. Bracket-fill ignores this gate.                                               |
| `ss.start_age` / `ss.start_age_a` / `ss.start_age_b`     | Claim age changes the bracket headroom available in the gap. Later claim = wider conversion window.                                                               |
| `pension.start_age`                                      | Pension annuity income reduces bracket headroom once it starts.                                                                                                   |
| `starting.spouse_*_pretax_*`                             | Hard cap on conversions (each spouse's conversion is capped at their pretax balance minus their RMD).                                                             |
| `starting.spouse_*_pretax_ira`                           | The backdoor pro-rata rule (IRC §408(d)(2)) taxes the Roth-conversion piece pro-rata against the pretax-IRA sub-balance. Tracked separately from pretax-401(k).   |

### E. Common patterns

```jsonc
// Aggressive gap-year conversions in a CA-then-FL household
"config": {
  "roth_conversion_target_bracket": 0.24,
  "cap_conversion_by_liquidity": true,
  "conversion_taxable_use_ratio": 0.75,    // willing to spend taxable on conv tax
  "state_regime": "CA",
  "state_regime_change_year_offset": 11,   // year retirement starts
  "state_regime_change_target": "stateless"
}

// Pure bracket-fill withdrawal, no Roth conversions
"config": {
  "withdrawal_strategy": "bracket_fill",
  "bracket_fill_target": 0.22,
  "roth_conversion_target_bracket": 0.0
}

// Fixed $50k/yr gap-year conversions, conservative liquidity stance
"config": {
  "roth_conversion_amount": 50000.0,
  "cap_conversion_by_liquidity": true,
  "conversion_taxable_use_ratio": 0.0,    // earned-income surplus only
  "protect_roth_in_conversion_years": true
}
```

## Health insurance premiums (§125 cafeteria plan)

Employer-sponsored medical / dental / vision premiums paid through a
§125 cafeteria plan ("premium-only plan") are pre-tax in three ways
that the simulator now models explicitly (v6.6+):

  * **Reduce federal Box 1 wages** → lower federal income tax
  * **Reduce FICA wages** (OASDI + Medicare) → lower payroll tax
  * **Reduce state wages** in every conforming state → lower state
    tax (CA, NY, IL, MA, etc. — flow-through automatic)

This is the key difference from a traditional 401(k): a 401(k)
deferral reduces Box 1 but **not** FICA wages. §125 premiums reduce
**both**. The same treatment also retroactively fixes the
long-standing HSA-FICA approximation (pre-v6.6 the simulator
documented but didn't model HSA's FICA exemption).

### `inputs.health_premiums` — per-spouse annual employee share

Quote **annual** dollars (NOT monthly) of the employee-paid portion.
The employer share is irrelevant for tax purposes (already excluded
from W-2 income).

```jsonc
{
  "inputs": {
    "health_premiums": {
      "spouse_a_medical": 6500.0,
      "spouse_a_dental":   450.0,
      "spouse_a_vision":   180.0,
      "spouse_b_medical": 1200.0,
      "spouse_b_dental":   240.0,
      "spouse_b_vision":   120.0
    }
  }
}
```

For a typical paystub: open the year-end summary, find lines marked
something like `MEDICAL (pre-tax)`, `DENTAL (pre-tax)`,
`VISION (pre-tax)` — sum each over 12 months (or grab the annual
"YTD" column). Do NOT include the HSA contribution here; that's
already on `inputs.contrib.hsa_family`.

### `cfg.section125_reduces_fica_wages` — default-on FICA treatment

```jsonc
{
  "config": {
    "section125_reduces_fica_wages": true
  }
}
```

  * `true` (default) → FICA + CA SDI are computed on **post-§125**
    wages: premiums and HSA both reduce the FICA base. This matches
    real Box 3 / Box 5 payroll. A household with $10k of M/D/V +
    $8,550 HSA saves an extra ~$1,420 of FICA per year.
  * `false` → FICA + SDI are computed on **gross** wages (pre-v6.6
    behavior). Box 1 / federal tax / state tax reductions are
    unchanged. Useful for reproducing pre-v6.6 numbers or for
    sensitivity analysis.

### Gating and clamps

  * A spouse's premium only applies during their working years
    (`age < retire_age` AND alive). Once retired, the simulator
    zeroes the premium — retiree healthcare belongs on
    `cfg.health_pre65_today` (post-retirement pre-Medicare) or
    `cfg.medicare_base_b_d_premium` (Medicare-eligible).
  * A non-working spouse's premium is **ignored** even if set
    (no W-2 to deduct from). Common scenario: spouse B retires
    early; the working spouse A carries both on their plan —
    quote the full combined family premium on `spouse_a_medical`.
  * Per-spouse combined deduction (M + D + V) is clamped at that
    spouse's gross W-2 wages. You can't §125-deduct more than you
    earn; the simulator silently caps rather than allowing
    negative Box 1.

### Diagnostic columns (every row of the simulation DataFrame)

  * `health_premium_a` — §125 deduction for spouse A in this year
  * `health_premium_b` — §125 deduction for spouse B in this year
  * `health_premium_total` — sum (matches the household §125
    Box 1 / FICA / state reduction)

## Mega-backdoor Roth (auto-spillover)

For high-earner households whose 401(k) plan supports after-tax
contributions plus in-plan Roth conversions ("mega-backdoor"), the
simulator can route any **excess elective-deferral target** —
dollars that pre-v6.6 silently leaked into taxable cash — into the
after-tax bucket up to the §415(c) ceiling. Real-world analog:
Vanguard's "Spillover After-Tax" feature and similar Fidelity /
Schwab record-keeper offerings.

### Two per-spouse knobs

```jsonc
{
  "inputs": {
    "spouse_a_mega_backdoor_enabled": true,
    "spouse_a_after_tax_401k_pct": 0.10
  }
}
```

  * **`spouse_*_mega_backdoor_enabled`** (default `false`) — gate.
    Whether your 401(k) plan supports after-tax contributions and
    same-day in-plan Roth conversions. Most plans don't; check
    your Summary Plan Description.
  * **`spouse_*_after_tax_401k_pct`** (default `0.0`) — explicit
    fraction of salary you'd LIKE routed to the after-tax bucket
    on top of any auto-spillover. Capped at the §415(c) room that
    remains after the auto-spillover has fired.

### How the auto-spillover works

Each working year, per spouse:

  1. Target deferral  =  `salary × total_contrib_pct`
  2. Actual elective deferral  =  `min(target, §402(g) cap)`
     (split into pretax / Roth via `roth_401k_pct`)
  3. Excess  =  `max(0, target − §402(g) cap)`
  4. §415(c) room  =  `§415(c) limit − base deferral − employer match`
     (catch-up is excluded — sits outside §415(c))
  5. **Auto-spillover**  =  `min(excess, §415(c) room)` ← new in v6.6
  6. Remaining room  =  §415(c) room − auto-spillover
  7. Explicit  =  `min(salary × after_tax_401k_pct, remaining room)`
  8. Total mega-backdoor  =  auto-spillover + explicit  →  Roth

Excess beyond what fits in §415(c) stays as taxable cash (same as
the pre-v6.6 silent-cap behavior). When the auto-spillover crowds
out the explicit pct, those uncovered dollars also stay as cash.

### Worked example

Spouse A, age 45, salary $300k:

| Setting | Value |
|---|---|
| `total_contrib_pct` | 0.30  (target $90,000) |
| `roth_401k_pct`     | 0.5 |
| `mega_backdoor_enabled` | true |
| `after_tax_401k_pct` | 0.10  (target $30,000) |
| Employer match | 100% match up to 7% of pay |

Numbers per year:

  * §402(g) cap (age <50, no catch-up) = $23,500
  * Elective deferral = $23,500 (split → $11,750 pretax + $11,750 Roth)
  * Excess = $90,000 − $23,500 = **$66,500**
  * Employer match: effective deferral pct $23.5k/$300k = 7.83%,
    capped at 7% → match = $300k × 7% × 1.0 = $21,000
  * §415(c) room = $70,000 − $23,500 − $21,000 = **$25,500**
  * Auto-spillover = min($66,500, $25,500) = **$25,500**  (fills room)
  * Remaining room = $0
  * Explicit target $30,000, fits into $0 of remaining room → **$0**
  * `after_tax_target_uncovered_a` = **$30,000** (crowded out)
  * Excess beyond §415(c) = $66,500 − $25,500 = $41,000 → taxable cash

End result: $25,500/yr more Roth than pre-v6.6 silently delivered.

### Diagnostic columns (every row of the simulation DataFrame)

  * `excess_deferral_a` / `excess_deferral_b` — raw target − §402(g)
    cap. Always emitted, even when mega-backdoor is off — surfaces
    today's silent-cap behavior for the first time.
  * `mega_backdoor_spillover_a` / `mega_backdoor_spillover_b` —
    auto-routed portion of after-tax. Subset of `mega_backdoor_{a,b}`.
  * `after_tax_target_uncovered_a` / `after_tax_target_uncovered_b` —
    explicit-pct dollars that didn't fit in remaining §415(c) room.
    Non-zero means the auto-spillover crowded the explicit pct out.
  * `mega_backdoor_a` / `mega_backdoor_b` — total after-tax routed
    to Roth (auto + explicit). Existing column, unchanged semantics.

### How to disable

  * Plan doesn't support after-tax + in-plan Roth conversions:
    set `spouse_*_mega_backdoor_enabled = false`. Disables BOTH the
    explicit pct and the auto-spillover in one knob flip; excess
    falls back to taxable cash exactly as in pre-v6.6.
  * Plan supports it but you're choosing not to contribute beyond
    the elective limit: keep the gate on, set
    `total_contrib_pct ≤ §402(g) cap / salary` so there's no excess,
    and `after_tax_401k_pct = 0`.

## Useful CLI patterns

Generate a fresh defaults dump (newer than `template.json` if you've
been editing on a feature branch):

```
tax-optimizer --print-defaults > /tmp/defaults.json
```

Run a scenario:

```
tax-optimizer --scenario scenarios/example01.json
```

Override a single value without editing the file:

```
tax-optimizer --scenario scenarios/example01.json \
  --set config.roth_conversion_target_bracket=0.22
```
