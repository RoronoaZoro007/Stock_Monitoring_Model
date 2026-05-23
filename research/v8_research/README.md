# v8 Research Boundary

This directory is reserved for research after the locked v7 baseline.

## Hard Boundaries

- Do not overwrite `locked_artifacts/v7_cap20_strong_label_003`.
- Do not modify the locked v7 model file, feature list, label definition,
  train/validation split, Top10 signal file, or audit outputs.
- Any change to universe construction, event cleaning, extreme-sample handling,
  features, labels, hyperparameters, TopN, or exit logic must be versioned as
  v8 or later.
- v8/v9 results must always be compared against `v7_locked`.
- The period `20260224-20260520` has already been used as the v7 validation
  period and must not be reused as the final test set for new models.
- Final evidence for v8/v9 must come from future unseen data or an independent
  walk-forward test that keeps each test window strictly out of sample.

## Allowed v8 Research Topics

- Static Top3000 replacement with point-in-time rolling liquidity universe.
- Event/corporate-action cleaning and explicit limit/suspension rules.
- Extreme target-return handling with dirty-data-only removal.
- Walk-forward training and validation.
- Execution-aware labels and execution-aware backtesting.
- Capacity, participation-rate and impact-cost modeling.

## Required Output For Any v8 Candidate

- Immutable config file.
- Feature list and label definition.
- Train/validation/test split report.
- Data lineage and no-future-data audit.
- Comparison table versus `v7_locked`.
- Statement identifying whether `20260224-20260520` was used only for legacy
  comparison and not for final model selection.
