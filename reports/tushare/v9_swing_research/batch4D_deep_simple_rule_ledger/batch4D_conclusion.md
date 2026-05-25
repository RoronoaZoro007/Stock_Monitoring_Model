# Batch 4D Deep Simple Rule Ledger

## Scope

- Built an overlapping 5-day capital ledger for `low_stock_ret_60d` and `combo_low_liquidity_weak_momentum`.
- No model training, no new rule search, no parameter tuning, no feature/label/horizon change, and no v7_locked modification.

## Core Findings

- Best research-holdout base ledger rule: `low_stock_ret_60d`, cumulative return `0.3859`, max drawdown `-0.1645`, PF `1.1818`, annualized turnover `78.33`.
- Best strict capacity ledger rule: `low_stock_ret_60d`, cumulative return `0.3813`, avg fill `0.9981`, max drawdown `-0.1645`.
- Research holdout has no weak trend-regime signal days; weak-market robustness is not proven on the newest period.
- Validation weak-regime best rule under base ledger: `low_stock_ret_60d`, avg net return on target `0.0241`, PF `1.7190`.

## Interpretation

- The two simple rules remain more defensible than Batch4B models because the ledger is explicit about overlap, turnover, cost and fill.
- This still does not clear forward tracking because weak-market coverage in the newest holdout is missing and exit-delay risk is only flagged, not simulated.
- `low_stock_ret_60d` is the more liquid and less concentrated candidate; `combo_low_liquidity_weak_momentum` has lower concentration but needs stronger drawdown review.

## Gate

- Gate result: `review_required_no_forward_tracking`.
- Next step, if accepted, should be a full rule audit with exit-delay simulation, monthly cash ledger review and independent weak-market sample validation.
- Do not resume complex model training from this result.
