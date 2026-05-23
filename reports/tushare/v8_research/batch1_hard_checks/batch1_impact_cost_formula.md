# Batch 1 Impact Cost Formula

Batch 1 uses the same simplified execution-cost formula introduced by the Batch
0 unified evaluator.

Formula:

`side_impact = min(0.0050, 0.0015 * sqrt(realized_participation))`

Answers to required audit questions:

| Question | Answer |
|---|---|
| Is buy impact included? | Yes, when the scenario enables impact cost. |
| Is sell impact included? | Yes, when the scenario enables impact cost. |
| Is impact linear in participation? | No. It uses square-root participation. |
| Is impact bucketed by participation bands? | No. |
| Does it distinguish buy vs sell direction? | Only by each side's realized participation; the same formula is used on both sides. |
| Does it distinguish capital size? | Yes indirectly: larger capital raises order notional and realized participation. |
| Does it distinguish turnover/amount size? | Yes indirectly: realized participation divides order notional by the relevant 5-minute bar amount. |
| Is this a simplified formula? | Yes. It does not use order book depth, queue position, active buy/sell imbalance, intrabar path, or Level-2 liquidity. |

This is an audit stress formula, not an optimized trading-cost model.
