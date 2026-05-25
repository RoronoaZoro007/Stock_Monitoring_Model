# Batch 4E Full Simple Rule Audit

## Scope

- Extended Batch 4D with exit-delay simulation, monthly cash ledger review and weak-market sample validation.
- No model training, no new rule search, no parameter tuning, no feature/label/horizon change, and no v7_locked modification.

## Core Findings

- Best delayed-exit research-holdout base rule: `low_stock_ret_60d`, cumulative return `0.3343`, max drawdown `-0.1679`, PF `1.1555`.
- Best delayed-exit strict capacity rule: `low_stock_ret_60d`, cumulative return `0.3297`, avg fill `0.9981`, max drawdown `-0.1679`.
- Exit-delay events found: `531`; unresolved after 5 trading days: `55`.
- Validation weak-sample best delayed base rule: `low_stock_ret_60d`, cumulative return `0.4633`, PF `1.4694`, trades `1940`.
- Research-holdout still has no weak trend-regime signal days; recent weak-market robustness remains unproven.
- Best research-holdout delayed base monthly profile: `combo_low_liquidity_weak_momentum`, monthly win rate `0.7059`, worst month `-0.0856`, longest negative-month streak `2`.

## Interpretation

- Exit-delay risk is now simulated rather than only flagged, but it is still based on daily bars and not intraday order-book liquidity.
- The rules remain more defensible than the failed Batch4B models, but the evidence is not sufficient for live readiness.
- Weak-market evidence is historical/validation stress evidence; it is not a newest-period unseen weak-market test.
- High-cost stress and monthly drawdown behavior must remain gating evidence before any forward paper tracking.

## Gate

- Gate result: `review_required_no_forward_tracking`.
- Do not resume complex model training from this result.
- Do not start live trading or broker integration.
- If accepted, the next step is a decision review: either keep historical research paused and wait for future weak-market data, or run only a paper-only forward shadow with explicit caveats.
