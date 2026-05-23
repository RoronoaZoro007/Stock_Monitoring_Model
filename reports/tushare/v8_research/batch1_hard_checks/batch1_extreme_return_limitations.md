# Batch 1 Extreme Return Limitations

This audit attributes `abs(target_return) > 20%` using only local data available
in the current project: repaired/raw feature frames, clean daily data, minute-
aggregated daily bars, raw 5-minute bars loaded for the extreme rows, and stock
basic metadata.

Fields marked `unknown` are intentionally left unknown. Batch 1 does not fetch
corporate-action details or authoritative suspension announcements, so it does
not guess ex-right/ex-dividend causes unless local minute-vs-daily price evidence
is present.

C0/C1/C2 are audit policies only. No model is retrained and no v7 feature,
label, TopN, universe or exit rule is changed.
