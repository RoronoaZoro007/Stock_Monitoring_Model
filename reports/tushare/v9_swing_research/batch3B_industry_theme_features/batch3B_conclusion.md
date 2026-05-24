# Batch 3B Industry / Theme Feature Engineering

## Scope

- This batch builds industry-level strength, breadth, amount-share and crowding diagnostics.
- It builds stock-relative industry return and stock share-in-industry diagnostics.
- It does not run factor IC, simple-rule backtests, model training, or parameter selection.

## Output Coverage

- Industry daily feature rows: `225541`
- Stock-industry feature rows: `9306920`
- Trade dates: `2032`
- Industries: `111`

## Industry Crowding State Distribution

| industry_crowding_state   |   rows |
|:--------------------------|-------:|
| crowding_normal           | 116855 |
| crowding_low              |  57143 |
| crowding_high             |  44883 |
| insufficient_history      |   6660 |

## Top Crowding Industries

| industry   |   days |   avg_amount_share |   p95_amount_share |   crowding_high_days | latest_crowding_state   |
|:-----------|-------:|-------------------:|-------------------:|---------------------:|:------------------------|
| 电气设备       |   2032 |        0.0701217   |         0.118461   |                  615 | crowding_normal         |
| 专用机械       |   2032 |        0.0300557   |         0.0468779  |                  572 | crowding_high           |
| 半导体        |   2032 |        0.0515523   |         0.0933294  |                  562 | crowding_high           |
| 通信设备       |   2032 |        0.0388603   |         0.0728089  |                  557 | crowding_normal         |
| 化工原料       |   2032 |        0.0329493   |         0.053237   |                  515 | crowding_normal         |
| 元器件        |   2032 |        0.0615983   |         0.099096   |                  512 | crowding_high           |
| 陶瓷         |   2032 |        0.000658901 |         0.00145687 |                  506 | crowding_high           |
| 摩托车        |   2032 |        0.00146658  |         0.00377601 |                  497 | crowding_low            |
| 玻璃         |   2032 |        0.00378786  |         0.00781994 |                  486 | crowding_high           |
| 新型电力       |   2032 |        0.00645708  |         0.0153686  |                  482 | crowding_normal         |
| 电器仪表       |   2032 |        0.00869156  |         0.0137587  |                  479 | crowding_high           |
| 水运         |   2032 |        0.00389087  |         0.00879011 |                  479 | crowding_normal         |
| 纺织机械       |   2032 |        0.000870102 |         0.0019006  |                  476 | crowding_high           |
| 机械基件       |   2032 |        0.0103638   |         0.0210639  |                  474 | crowding_high           |
| 农药化肥       |   2032 |        0.00862506  |         0.0199786  |                  471 | crowding_low            |

## Theme / Concept Status

| feature_group            | status          | reason                                                                                           | allowed_use                                                                                                                  |
|:-------------------------|:----------------|:-------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------|
| concept_theme_membership | disabled        | No point-in-time concept/theme membership dataset has been locked in Stage 0.                    | Do not use for IC or training until a separate data-foundation batch locks source, coverage, fields and point-in-time rules. |
| industry_snapshot        | diagnostic_only | stock_basic.industry is a current snapshot field and may contain classification look-ahead risk. | Allowed for diagnostics and grouping; training use requires explicit acceptance or point-in-time replacement.                |

## Gate Result

- Gate result: `pass`.
- Reason: industry diagnostics are built, concept/theme features are explicitly disabled, and point-in-time limitations are documented.

## Next Step

- Execute `Batch 3C factor IC and decile diagnostics` after accepting current industry limitations.
- Do not train models or run simple-rule baseline yet.
