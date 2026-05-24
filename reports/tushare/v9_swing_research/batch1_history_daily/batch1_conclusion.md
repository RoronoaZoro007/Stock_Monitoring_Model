# v9 Swing Research - Batch 1 History Daily Data

## Scope

- Requested range: 20180101 to 20260522.
- This batch only validates and downloads daily-level historical data needed for swing-model research.
- It does not train a model, tune parameters, or modify v7_locked.

## Interface Smoke Test

| api_name    | params_mode   | status   |   rows |   seconds | columns                                                                                                                              | error   |
|:------------|:--------------|:---------|-------:|----------:|:-------------------------------------------------------------------------------------------------------------------------------------|:--------|
| trade_cal   | range/static  | ok       |      1 |     0.136 | exchange,cal_date,is_open,pretrade_date                                                                                              |         |
| stock_basic | range/static  | ok       |   5522 |     0.427 | ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date                                                  |         |
| namechange  | range/static  | ok       |   5250 |     1.62  | ts_code,name,start_date,end_date,ann_date,change_reason                                                                              |         |
| daily       | trade_date    | ok       |   5504 |     0.494 | ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount                                                           |         |
| adj_factor  | trade_date    | ok       |   5522 |     0.185 | ts_code,trade_date,adj_factor                                                                                                        |         |
| daily_basic | trade_date    | ok       |   5504 |     1.029 | ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,total_share,float_share,free_share,total_mv,circ_mv |         |
| suspend_d   | trade_date    | ok       |     25 |     0.07  | ts_code,trade_date,suspend_timing,suspend_type                                                                                       |         |
| stk_limit   | trade_date    | ok       |   7613 |     0.435 | trade_date,ts_code,up_limit,down_limit                                                                                               |         |

## Coverage Before Download

| api_name    | purpose                                                 |   expected_open_dates |   available_dates |   missing_dates |   first_missing |   last_missing |   first_available |   last_available |
|:------------|:--------------------------------------------------------|----------------------:|------------------:|----------------:|----------------:|---------------:|------------------:|-----------------:|
| daily       | daily OHLCV and amount                                  |                  2032 |               522 |            1510 |        20180227 |       20240520 |          20180102 |         20260522 |
| adj_factor  | adjustment factor                                       |                  2032 |                37 |            1995 |        20180301 |       20260522 |          20180102 |         20180228 |
| daily_basic | market cap, float shares, turnover and valuation fields |                  2032 |                35 |            1997 |        20180124 |       20260522 |          20180102 |         20180227 |
| suspend_d   | suspension status                                       |                  2032 |                35 |            1997 |        20180124 |       20260522 |          20180102 |         20180227 |
| stk_limit   | daily limit up/down prices                              |                  2032 |                35 |            1997 |        20180124 |       20260522 |          20180102 |         20180227 |

## Coverage After Download

| api_name    | purpose                                                 |   expected_open_dates |   available_dates |   missing_dates | first_missing   | last_missing   |   first_available |   last_available |
|:------------|:--------------------------------------------------------|----------------------:|------------------:|----------------:|:----------------|:---------------|------------------:|-----------------:|
| daily       | daily OHLCV and amount                                  |                  2032 |              2032 |               0 |                 |                |          20180102 |         20260522 |
| adj_factor  | adjustment factor                                       |                  2032 |              2032 |               0 |                 |                |          20180102 |         20260522 |
| daily_basic | market cap, float shares, turnover and valuation fields |                  2032 |              2032 |               0 |                 |                |          20180102 |         20260522 |
| suspend_d   | suspension status                                       |                  2032 |              2032 |               0 |                 |                |          20180102 |         20260522 |
| stk_limit   | daily limit up/down prices                              |                  2032 |              2032 |               0 |                 |                |          20180102 |         20260522 |

## Download Summary

| api_name    | status     |   files |     rows |   seconds |
|:------------|:-----------|--------:|---------:|----------:|
| adj_factor  | downloaded |    1995 |  9426441 |   567.007 |
| adj_factor  | exists     |      37 |   135038 |     0     |
| daily       | downloaded |    1510 |  6564215 |   923.514 |
| daily       | exists     |     522 |  2742705 |     0     |
| daily_basic | downloaded |    1997 |  9128930 |  1867.88  |
| daily_basic | exists     |      35 |   113255 |     0     |
| stk_limit   | downloaded |    1997 | 11328304 |   671.457 |
| stk_limit   | exists     |      35 |   121795 |     0     |
| suspend_d   | downloaded |    1997 |   100407 |   377.793 |
| suspend_d   | exists     |      35 |     9368 |     0     |

## Data Mapping

| Need | Endpoint | Local path pattern | Notes |
|---|---|---|---|
| Daily OHLCV | daily | data_tushare/raw/daily/trade_date=YYYYMMDD.parquet | Base daily label and return data |
| Adjustment | adj_factor | data_tushare/raw/adj_factor/trade_date=YYYYMMDD.parquet | Used to build adjusted prices |
| Market cap / turnover | daily_basic | data_tushare/raw/daily_basic/trade_date=YYYYMMDD.parquet | total_mv, circ_mv, turnover_rate, turnover_rate_f |
| Suspension | suspend_d | data_tushare/raw/suspend_d/trade_date=YYYYMMDD.parquet | Empty file means no returned suspended records for that date |
| Limit prices | stk_limit | data_tushare/raw/stk_limit/trade_date=YYYYMMDD.parquet | Used with OHLCV to infer limit-up/down blocks |
| ST / name history | namechange | data_tushare/raw/bootstrap/namechange_START_END.parquet | Point-in-time ST proxy; must be converted to daily flags before modeling |

## Important Constraints

- Multi-day range calls for full-market daily-like endpoints can be row-capped by the provider. Batch 1 therefore downloads these endpoints by single trade_date.
- ST status is not a single clean daily field here; it must be derived from namechange intervals and validated before use.
- This completes raw daily-level data staging only. Feature joins, market-regime labels, and factor tests belong to later batches.