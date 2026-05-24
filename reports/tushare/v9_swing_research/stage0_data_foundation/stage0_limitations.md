# v9 Stage 0 Data Foundation Limitations

## What Is Locked

- Raw daily-level files for `daily`, `adj_factor`, `daily_basic`, `suspend_d`, and `stk_limit` are hashed per file.
- Bootstrap files for `trade_cal`, `stock_basic`, and `namechange` are hashed.
- Coverage is checked against open trading dates from `trade_cal` for 2018-01-02 to 2026-05-22.

## Point-In-Time Risks

- `daily`, `daily_basic`, `adj_factor`, `suspend_d`, and `stk_limit` are keyed by trade date and should only be used after the relevant date is observable. They are suitable for T-close to T+1 swing research, not intraday pre-close decisions.
- `stock_basic.industry` and `stock_basic.name` are current snapshot fields. They must not be treated as point-in-time industry/name history until independently validated.
- `namechange` can create an ST/name-history proxy by effective interval, but name text alone is not a complete exchange status ledger. Announcement timing must be audited before using it as a training filter.
- `adj_factor` can reflect back-adjusted history. The downstream label audit must keep raw-vs-adjusted return diagnostics.
- `suspend_d` contains returned suspension records; absence of a row is not a full per-stock tradability ledger by itself.
- No concept/theme membership data is locked in this stage. Any concept/industry crowding research must document its own point-in-time source before training.

## Coverage Status

| endpoint    |   expected_open_dates |   available_dates |   missing_dates |   first_available |   last_available | aggregate_sha256                                                 |
|:------------|----------------------:|------------------:|----------------:|------------------:|-----------------:|:-----------------------------------------------------------------|
| daily       |                  2032 |              2032 |               0 |          20180102 |         20260522 | b8a3d6a0ba9198059785e20ed8899defee805217cd1b0454e1a4b8ef332e7122 |
| adj_factor  |                  2032 |              2032 |               0 |          20180102 |         20260522 | 2a05fb34d3f92676631858c963dd92f96fca6a54900d995f09b2cf3661629d5b |
| daily_basic |                  2032 |              2032 |               0 |          20180102 |         20260522 | 402111c1bcb6b73a67b779e1e663a5cf96e4f6009bd73f935caa6592655a8b2d |
| suspend_d   |                  2032 |              2032 |               0 |          20180102 |         20260522 | c7d4e2b5437ca727cab6e32f1de80c77a14e882b95fc163e8bd3e029f023a26c |
| stk_limit   |                  2032 |              2032 |               0 |          20180102 |         20260522 | a16324d9fcd8870f74a4f384c0ce409d4b3c3c356b25e28dc375a32528d26f44 |

## Bootstrap Files

| endpoint    |   files |   rows | sha256                                                           | known_risk                                                                                                              |
|:------------|--------:|-------:|:-----------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------|
| trade_cal   |       1 |   3064 | 651a9d117adcc8d862ea5f9e25a927549106e6e87eda249dc24c4d2e71e75f66 | Future known exchange holidays are fine for scheduling, but labels must not use future prices.                          |
| stock_basic |       1 |   5847 | a8a937c964c82c448f6dcec049fc5f5fb6a73652ed3452f372a0f022468877ce | Current industry/name can leak future classification; industry should be treated as non-point-in-time until validated.  |
| namechange  |       1 |   5250 | 0511a853c86a58829da5953b633e0c00be3b7e93605cd6a63c6fdd7061f94872 | ST derived from name text is only a proxy; announcement timing and exchange status should be validated before training. |
