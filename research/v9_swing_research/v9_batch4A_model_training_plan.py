#!/usr/bin/env python3
"""Batch 4A: freeze the v9 swing model training plan.

This script does not train models. It reads the completed Batch 3A/3B/3C/3D
artifacts and writes a frozen, auditable plan for a later Batch 4B training run.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "reports" / "tushare" / "v9_swing_research"
REPORT_DIR = REPORT_ROOT / "batch4A_model_training_plan"
DATA_DIR = ROOT / "data_tushare" / "clean" / "v9"

B3A_DIR = REPORT_ROOT / "batch3A_market_regime"
B3B_DIR = REPORT_ROOT / "batch3B_industry_theme_features"
B3C_DIR = REPORT_ROOT / "batch3C_factor_ic_decile"
B3D_DIR = REPORT_ROOT / "batch3D_simple_rule_baseline"

GLOBAL_HANDOFF_PATH = REPORT_ROOT / "v9_current_handoff.md"
ROADMAP_PATH = REPORT_ROOT / "v9_execution_roadmap.md"
STAGE_STATUS_PATH = REPORT_ROOT / "v9_stage_status.csv"

PRIMARY_HORIZON = "5d"
DIAGNOSTIC_HORIZONS = ["3d", "10d"]
TRAIN_START = "20180102"
TRAIN_END = "20221230"
VALIDATION_START = "20230103"
VALIDATION_END = "20241231"
TEST_START = "20250102"
TEST_END = "20260522"

BLOCKED_INDUSTRY_STATUS = "diagnostic_industry_snapshot"
USABLE_STATUS = "usable_after_t_close"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(v: float | int | None) -> str:
    if v is None or pd.isna(v):
        return ""
    return f"{float(v) * 100:.2f}%"


def read_inputs() -> dict[str, pd.DataFrame]:
    return {
        "b3c_candidates": pd.read_csv(B3C_DIR / "batch3C_candidate_factor_screen.csv"),
        "b3c_coverage": pd.read_csv(B3C_DIR / "batch3C_coverage_summary.csv"),
        "b3c_factor_dictionary": pd.read_csv(B3C_DIR / "batch3C_factor_dictionary.csv"),
        "b3d_rules": pd.read_csv(B3D_DIR / "batch3D_rule_backtest_summary.csv"),
        "b3d_random": pd.read_csv(B3D_DIR / "batch3D_random_baseline_summary.csv"),
        "b3d_coverage": pd.read_csv(B3D_DIR / "batch3D_coverage_summary.csv"),
    }


def build_horizon_decision(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cand = dfs["b3c_candidates"].copy()
    rules = dfs["b3d_rules"].copy()
    coverage = dfs["b3d_coverage"].copy()

    rows: list[dict[str, object]] = []
    for horizon in ["3d", "5d", "10d"]:
        usable = cand[
            cand["horizon"].eq(horizon)
            & cand["candidate_flag"].astype(bool)
            & cand["point_in_time_status"].eq(USABLE_STATUS)
        ].copy()
        eligible = rules[rules["horizon"].eq(horizon) & rules["eligible_for_training_plan"].astype(bool)].copy()
        cov = coverage[coverage["horizon"].eq(horizon)].iloc[0].to_dict()
        least_bad_drawdown = eligible["max_drawdown"].max() if not eligible.empty else float("nan")
        best_rule = eligible.sort_values(["profit_factor", "rule_cumret_percentile"], ascending=[False, False]).head(1)
        rows.append(
            {
                "horizon": horizon,
                "decision_role": "primary_training_label" if horizon == PRIMARY_HORIZON else "diagnostic_only",
                "screened_valid_target_ratio": cov["screened_valid_target_ratio"],
                "usable_candidate_factor_count": int(len(usable)),
                "usable_candidate_abs_ic_mean": float(usable["abs_mean_ic"].mean()) if not usable.empty else float("nan"),
                "usable_candidate_spread_mean": float(usable["direction_adjusted_mean_spread"].mean()) if not usable.empty else float("nan"),
                "eligible_simple_rule_count": int(len(eligible)),
                "best_rule_id": "" if best_rule.empty else str(best_rule.iloc[0]["rule_id"]),
                "best_rule_profit_factor": float(best_rule.iloc[0]["profit_factor"]) if not best_rule.empty else float("nan"),
                "best_rule_daily_win_rate": float(best_rule.iloc[0]["daily_win_rate"]) if not best_rule.empty else float("nan"),
                "least_bad_eligible_max_drawdown": float(least_bad_drawdown),
                "decision_reason": horizon_reason(horizon),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "batch4A_horizon_decision.csv", index=False)
    return out


def horizon_reason(horizon: str) -> str:
    if horizon == "5d":
        return (
            "Frozen as primary: stronger IC/spread than 3d, materially less extreme than 10d, "
            "and consistent with earlier Batch2 label audit compromise."
        )
    if horizon == "3d":
        return "Kept diagnostic: better drawdown profile but thinner spread and lower IC than 5d."
    return "Kept diagnostic only: strongest gross cohort results but extreme drawdown and higher label tail risk."


def build_feature_freeze(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cand = dfs["b3c_candidates"].copy()
    dictionary = dfs["b3c_factor_dictionary"].copy()
    primary = cand[cand["horizon"].eq(PRIMARY_HORIZON)].copy()
    merged = dictionary.merge(
        primary[
            [
                "factor",
                "candidate_flag",
                "mean_ic",
                "icir",
                "positive_ic_rate",
                "direction_adjusted_mean_spread",
                "abs_mean_ic",
            ]
        ],
        on="factor",
        how="left",
    )
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        factor = str(row["factor"])
        status = str(row["point_in_time_status"])
        candidate = bool(row["candidate_flag"]) if not pd.isna(row.get("candidate_flag")) else False
        if status == USABLE_STATUS and candidate:
            if factor == "stock_amount_share_in_industry":
                training_status = "conditional_ablation_only"
                role = "liquidity_crowding_diagnostic"
                reason = (
                    "Batch3C marks it usable after close, but it depends on industry grouping; "
                    "use only in a separate ablation unless point-in-time industry membership is accepted."
                )
            else:
                training_status = "primary_training_feature"
                role = "primary_feature"
                reason = "Candidate in Batch3C for 5d and available after T close without concept/theme data."
        elif status == BLOCKED_INDUSTRY_STATUS:
            training_status = "blocked_diagnostic_only"
            role = "diagnostic_grouping"
            reason = "Current-snapshot industry dependency; blocked until point-in-time industry source is locked."
        else:
            training_status = "blocked_not_candidate"
            role = "not_selected"
            reason = "Did not pass Batch3C 5d candidate screen."
        rows.append(
            {
                "factor": factor,
                "factor_group": row["factor_group"],
                "description": row["description"],
                "point_in_time_status": status,
                "batch3C_5d_candidate_flag": candidate,
                "batch3C_5d_mean_ic": row.get("mean_ic"),
                "batch3C_5d_icir": row.get("icir"),
                "batch3C_5d_positive_ic_rate": row.get("positive_ic_rate"),
                "batch3C_5d_direction_adjusted_spread": row.get("direction_adjusted_mean_spread"),
                "expected_direction": "low_factor" if pd.notna(row.get("mean_ic")) and float(row["mean_ic"]) < 0 else "not_frozen",
                "training_status": training_status,
                "feature_role": role,
                "transform_freeze": "daily_cross_sectional_rank_pct_then_center" if training_status == "primary_training_feature" else "not_used_in_primary_training",
                "missing_value_policy": "cross_sectional_median_rank" if training_status == "primary_training_feature" else "not_applicable",
                "reason": reason,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "batch4A_feature_freeze.csv", index=False)
    return out


def compute_split_summary() -> pd.DataFrame:
    labels_path = DATA_DIR / "v9_swing_labels_3_5_10.parquet"
    cols = [
        "trade_date",
        "listed_days",
        "st_flag",
        "suspend_flag",
        "limit_up_close_flag",
        "limit_down_close_flag",
        "fwd_ret_3d_open",
        "fwd_ret_5d_open",
        "fwd_ret_10d_open",
    ]
    df = pd.read_parquet(labels_path, columns=cols)
    df["trade_date"] = df["trade_date"].astype(str)
    for col in ["st_flag", "suspend_flag", "limit_up_close_flag", "limit_down_close_flag"]:
        df[col] = df[col].fillna(False).astype(bool)
    df["screened_universe_flag"] = (
        df["listed_days"].ge(120)
        & ~df["st_flag"]
        & ~df["suspend_flag"]
        & ~df["limit_up_close_flag"]
        & ~df["limit_down_close_flag"]
    )
    split_specs = [
        ("train", TRAIN_START, TRAIN_END),
        ("validation", VALIDATION_START, VALIDATION_END),
        ("research_holdout", TEST_START, TEST_END),
    ]
    rows: list[dict[str, object]] = []
    for split_name, start, end in split_specs:
        mask = df["trade_date"].between(start, end)
        screened = mask & df["screened_universe_flag"]
        row = {
            "split": split_name,
            "start_date": start,
            "end_date": end,
            "rows": int(mask.sum()),
            "screened_rows": int(screened.sum()),
            "trade_days": int(df.loc[mask, "trade_date"].nunique()),
        }
        for h in [3, 5, 10]:
            row[f"screened_valid_fwd_ret_{h}d_open_rows"] = int((screened & df[f"fwd_ret_{h}d_open"].notna()).sum())
            row[f"screened_valid_fwd_ret_{h}d_open_ratio"] = float((screened & df[f"fwd_ret_{h}d_open"].notna()).sum() / max(int(screened.sum()), 1))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "batch4A_split_summary.csv", index=False)
    return out


def write_label_freeze() -> None:
    text = f"""# Batch 4A label freeze

primary_horizon: {PRIMARY_HORIZON}
diagnostic_horizons:
  - 3d
  - 10d
primary_return_column: fwd_ret_5d_open
primary_binary_column: label_win_5d
entry_price_definition: T+1 adjusted open
exit_price_definition: T+5 adjusted close
raw_formula: fwd_ret_5d_open = exit_adj_close_5d / entry_adj_open - 1
training_objective_priority:
  - cross_sectional_rank_or_regression_on_fwd_ret_5d_open
  - binary_label_win_5d_as_secondary_diagnostic
forbidden_label_changes:
  - no threshold optimization after seeing validation/test results
  - no switching primary horizon after Batch 4B starts
  - no using 10d as primary unless Batch 4A is explicitly revised and re-approved
  - no post-hoc removal of losing dates, losing industries, or low-capacity samples
label_notes:
  - 5d is selected before model training based on Batch2/3C/3D evidence.
  - Labels are gross research labels; trading costs and capacity are applied in evaluation, not in label construction.
  - Diagnostic horizons cannot be used for model selection in the same Batch 4B run.
"""
    (REPORT_DIR / "batch4A_label_freeze.yaml").write_text(text, encoding="utf-8")


def write_split_config(split_summary: pd.DataFrame) -> None:
    text = f"""# Batch 4A split config

split_method: chronological_time_series
shuffle_samples: false
fit_transform_scope: train_only
primary_train:
  start_date: {TRAIN_START}
  end_date: {TRAIN_END}
validation:
  start_date: {VALIDATION_START}
  end_date: {VALIDATION_END}
research_holdout:
  start_date: {TEST_START}
  end_date: {TEST_END}
holdout_warning: >
  The research_holdout is not a pristine final test set because Batch3C/3D
  diagnostics already looked across the full historical sample. It can be used
  for locked-plan audit, but final credibility still requires later walk-forward
  and forward paper tracking.
screened_universe:
  listed_days_min: 120
  exclude_st: true
  exclude_suspended: true
  exclude_limit_up_close: true
  exclude_limit_down_close: true
primary_horizon: {PRIMARY_HORIZON}
split_summary_file: batch4A_split_summary.csv
"""
    (REPORT_DIR / "batch4A_split_config.yaml").write_text(text, encoding="utf-8")


def build_baseline_reference(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rules = dfs["b3d_rules"].copy()
    primary_rules = rules[rules["horizon"].eq(PRIMARY_HORIZON)].copy()
    primary_rules = primary_rules.sort_values(["eligible_for_training_plan", "profit_factor"], ascending=[False, False])
    cols = [
        "rule_id",
        "rule_group",
        "horizon",
        "uses_industry_snapshot",
        "eligible_for_training_plan",
        "signal_days",
        "total_trades",
        "mean_daily_return",
        "daily_win_rate",
        "trade_win_rate",
        "profit_factor",
        "trade_profit_factor",
        "max_drawdown",
        "rule_cumret_percentile",
        "random_cumret_median",
        "random_cumret_p95",
    ]
    out = primary_rules[cols].copy()
    out["baseline_role"] = out["eligible_for_training_plan"].map({True: "minimum_to_beat", False: "diagnostic_baseline"})
    out["evaluation_note"] = "gross cohort baseline from Batch3D; Batch4B must report model-vs-rule under the same locked split."
    out.to_csv(REPORT_DIR / "batch4A_baseline_reference.csv", index=False)
    return out


def write_training_plan(
    horizon_decision: pd.DataFrame,
    features: pd.DataFrame,
    baseline: pd.DataFrame,
    split_summary: pd.DataFrame,
) -> None:
    primary_features = features[features["training_status"].eq("primary_training_feature")]["factor"].tolist()
    conditional_features = features[features["training_status"].eq("conditional_ablation_only")]["factor"].tolist()
    min_to_beat = baseline[baseline["baseline_role"].eq("minimum_to_beat")].head(6)
    lines = [
        "# Batch 4A Model Training Plan Freeze",
        "",
        "## Scope",
        "",
        "- This batch freezes the v9 swing model training plan.",
        "- No model training, model scoring, hyperparameter optimization, or live/paper trading is performed.",
        "- `v7_locked` remains untouched.",
        "",
        "## Frozen Primary Horizon",
        "",
        f"- Primary horizon: `{PRIMARY_HORIZON}`.",
        "- 3d and 10d remain diagnostic-only and cannot be used for Batch4B model selection.",
        "- Reason: 5d has stronger IC/spread than 3d and lower tail/drawdown concern than 10d.",
        "",
        horizon_decision.to_markdown(index=False),
        "",
        "## Frozen Label",
        "",
        "- Primary target: `fwd_ret_5d_open`.",
        "- Formula: `exit_adj_close_5d / entry_adj_open - 1`.",
        "- Entry: T+1 adjusted open. Exit: T+5 adjusted close.",
        "- `label_win_5d` is secondary diagnostic only.",
        "",
        "## Frozen Primary Features",
        "",
        ", ".join(f"`{x}`" for x in primary_features),
        "",
        "Conditional ablation-only features:",
        "",
        ", ".join(f"`{x}`" for x in conditional_features) if conditional_features else "None",
        "",
        "Blocked fields:",
        "",
        "- Current-snapshot industry fields and concept/theme fields are blocked from primary training.",
        "- Market regime fields are used for evaluation grouping, not as primary model features in Batch4B.",
        "",
        "## Frozen Splits",
        "",
        split_summary.to_markdown(index=False),
        "",
        "## Minimum Baselines To Beat",
        "",
        min_to_beat[
            [
                "rule_id",
                "horizon",
                "mean_daily_return",
                "daily_win_rate",
                "profit_factor",
                "max_drawdown",
                "rule_cumret_percentile",
            ]
        ].to_markdown(index=False),
        "",
        "## Batch 4B Allowed Training Families",
        "",
        "- Ridge / ElasticNet regression or rank model.",
        "- Logistic / linear probability model for `label_win_5d` as secondary diagnostic.",
        "- LightGBM rank/binary only with constrained complexity and fixed small search grid defined before execution.",
        "",
        "## Batch 4B Mandatory Evaluation",
        "",
        "- Compare against `batch4A_baseline_reference.csv` and matched random baseline.",
        "- Report by year, split, market regime, size bucket, liquidity bucket and industry diagnostic group.",
        "- Apply cost/capacity stress separately from label construction.",
        "- Include profit concentration checks: remove largest 1/3/5/10 winning days.",
        "",
        "## Blocked Actions",
        "",
        "- Do not train before this plan is reviewed.",
        "- Do not add features outside `batch4A_feature_freeze.csv`.",
        "- Do not switch to 10d because its gross return is higher.",
        "- Do not use concept/theme features until a point-in-time data foundation is locked.",
    ]
    (REPORT_DIR / "batch4A_model_training_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_conclusion(horizon_decision: pd.DataFrame, features: pd.DataFrame, baseline: pd.DataFrame) -> None:
    primary_count = int(features["training_status"].eq("primary_training_feature").sum())
    conditional_count = int(features["training_status"].eq("conditional_ablation_only").sum())
    blocked_count = int(features["training_status"].str.startswith("blocked").sum())
    min_to_beat_count = int(baseline["baseline_role"].eq("minimum_to_beat").sum())
    lines = [
        "# Batch 4A Conclusion",
        "",
        "## What Changed",
        "",
        "- Froze the v9 swing model training plan.",
        "- Froze primary horizon, label, feature set, split config, baseline references and evaluation gates.",
        "- Updated the project handoff to Batch 4B.",
        "",
        "## What Did Not Change",
        "",
        "- No model was trained.",
        "- No feature was recomputed.",
        "- No label was changed.",
        "- No historical parameter optimization was performed.",
        "- `v7_locked` was not modified or moved.",
        "",
        "## Frozen Decisions",
        "",
        f"- Primary horizon: `{PRIMARY_HORIZON}`.",
        "- Primary label: `fwd_ret_5d_open`.",
        f"- Primary training features: `{primary_count}`.",
        f"- Conditional ablation-only features: `{conditional_count}`.",
        f"- Blocked/diagnostic-only features: `{blocked_count}`.",
        f"- Minimum-to-beat Batch3D baselines for 5d: `{min_to_beat_count}`.",
        "",
        "## Data-Supported Rationale",
        "",
        "- 5d retains 7 usable Batch3C candidate factors; 6 are frozen as primary training features and 1 industry-dependent factor is kept as conditional ablation-only.",
        "- 3d remains useful as a lower-tail-risk diagnostic but has thinner IC/spread.",
        "- 10d remains diagnostic because the strongest gross returns are paired with near-total drawdowns in simple rules.",
        "- Batch4B must prove improvement over simple rules, not merely reproduce low-size/low-liquidity exposure.",
        "",
        "## Gate",
        "",
        "- Gate result: `pass_to_batch4B_after_review`.",
        "- Next step, if accepted: `Batch 4B lightweight model training` using only the frozen plan.",
        "- Do not start Batch 4B automatically from this script.",
    ]
    (REPORT_DIR / "batch4A_conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_handoff(features: pd.DataFrame, baseline: pd.DataFrame) -> None:
    primary_features = features[features["training_status"].eq("primary_training_feature")]["factor"].tolist()
    lines = [
        "# Handoff: Batch 4A to Batch 4B",
        "",
        "## Completed Batch",
        "",
        "- Completed: `Batch 4A model training plan freeze`.",
        "- Output directory: `reports/tushare/v9_swing_research/batch4A_model_training_plan/`.",
        "- No model training, hyperparameter search, prediction generation, or trading was performed.",
        "",
        "## Frozen Plan",
        "",
        f"- Primary horizon: `{PRIMARY_HORIZON}`.",
        "- Primary label: `fwd_ret_5d_open`.",
        f"- Primary features: `{', '.join(primary_features)}`.",
        "- Split: chronological train 20180102-20221230, validation 20230103-20241231, research_holdout 20250102-20260522.",
        "- Baseline: Batch3D 5d simple rules and matched random baseline.",
        "",
        "## Required Inputs For Next Step",
        "",
        "- `reports/tushare/v9_swing_research/batch4A_model_training_plan/batch4A_model_training_plan.md`",
        "- `reports/tushare/v9_swing_research/batch4A_model_training_plan/batch4A_feature_freeze.csv`",
        "- `reports/tushare/v9_swing_research/batch4A_model_training_plan/batch4A_label_freeze.yaml`",
        "- `reports/tushare/v9_swing_research/batch4A_model_training_plan/batch4A_split_config.yaml`",
        "- `reports/tushare/v9_swing_research/batch4A_model_training_plan/batch4A_baseline_reference.csv`",
        "",
        "## Blocked Actions",
        "",
        "- Do not add non-frozen features in Batch4B.",
        "- Do not switch primary horizon after seeing Batch4B results.",
        "- Do not use current-snapshot industry or concept/theme fields as primary training features.",
        "- Do not treat `research_holdout` as a pristine final test, because Batch3 diagnostics already viewed full history.",
        "- Do not proceed to live or paper tracking from Batch4A.",
    ]
    text = "\n".join(lines) + "\n"
    (REPORT_DIR / "batch4A_handoff_to_batch4B.md").write_text(text, encoding="utf-8")
    GLOBAL_HANDOFF_PATH.write_text(text, encoding="utf-8")


def update_stage_status() -> None:
    if not STAGE_STATUS_PATH.exists():
        return
    df = pd.read_csv(STAGE_STATUS_PATH)
    mask4a = df["batch"].eq("batch4A_model_training_plan")
    df.loc[mask4a, "status"] = "completed"
    df.loc[mask4a, "output_dir"] = "reports/tushare/v9_swing_research/batch4A_model_training_plan"
    df.loc[mask4a, "gate_result"] = "pass"
    df.loc[mask4a, "next_action"] = "Proceed to Batch 4B only after reviewing frozen horizon, labels, features, split and baselines; do not change plan during training."
    mask4b = df["batch"].eq("batch4B_lightweight_model_training")
    df.loc[mask4b, "status"] = "pending"
    df.loc[mask4b, "gate_result"] = "pending"
    df.loc[mask4b, "next_action"] = "Train only frozen low-complexity models from Batch 4A and compare to Batch 3D baselines."
    df.to_csv(STAGE_STATUS_PATH, index=False)


def update_roadmap() -> None:
    if not ROADMAP_PATH.exists():
        return
    text = ROADMAP_PATH.read_text(encoding="utf-8")
    text = text.replace(
        "| Stage 7 轻量模型研究 | 只在 Stage 5/6 通过后训练 | 后续 Batch4 | 等待 Batch 4A | Logistic/Ridge/LightGBM rank model | 先冻结训练计划、特征、标签、切分、基准和成本容量限制 |",
        "| Stage 7 训练计划冻结 | 冻结 horizon、label、feature、split 和 baseline | `batch4A_model_training_plan` | 已完成 | 5d 主标签、冻结特征、时间切分、Batch3D 基准 | 已通过进入 Batch 4B；禁止改计划后训练 |",
    )
    marker = "### Step 4B: 轻量模型训练"
    addition = """### Stage 7A 训练计划冻结

输出目录：`reports/tushare/v9_swing_research/batch4A_model_training_plan/`

核心冻结：

- 主训练周期：`5d`。
- 主标签：`fwd_ret_5d_open = exit_adj_close_5d / entry_adj_open - 1`。
- 主训练特征：`log_amount`、`turnover_rate`、`log_total_mv`、`stock_ret_5d`、`stock_ret_20d`、`stock_ret_60d`。
- 条件消融特征：`stock_amount_share_in_industry`，需显式说明行业分组风险。
- 禁用字段：当前快照行业字段、概念/题材字段、未通过 Batch3C 的字段。
- 切分：train 20180102-20221230，validation 20230103-20241231，research_holdout 20250102-20260522。
- 基准：Batch3D 的 5d 简单规则和 matched random baseline。

重要限制：

- `research_holdout` 不是完全未见最终测试集，因为 Batch3C/3D 已看过全历史诊断。
- Batch4B 只能使用冻结配置训练，不能因为结果好坏切换 horizon、删样本或加特征。

Handoff：

- `reports/tushare/v9_swing_research/batch4A_model_training_plan/batch4A_handoff_to_batch4B.md`
- `reports/tushare/v9_swing_research/v9_current_handoff.md`

"""
    if addition.strip() not in text:
        text = text.replace(marker, addition + marker)
    text = text.replace("- 内容来自 `Batch 3D -> Batch 4A`", "- 内容来自 `Batch 4A -> Batch 4B`")
    text = text.replace(
        "> Batch 4A: model training plan freeze",
        "> Batch 4B: lightweight model training, only if the Batch 4A frozen plan is accepted",
    )
    text = text.replace(
        "- Batch 4A 只能冻结计划，不训练模型；真正训练必须等待 Batch 4A 验收后另行执行 Batch 4B。",
        "- Batch 4A 已冻结计划；真正训练必须严格按 Batch 4A 配置另行执行 Batch 4B，不得在训练中改 horizon、特征或样本。",
    )
    ROADMAP_PATH.write_text(text, encoding="utf-8")


def write_hashes() -> None:
    rows = []
    for path in sorted(REPORT_DIR.glob("batch4A_*")):
        if path.name == "batch4A_file_sha256.csv" or not path.is_file():
            continue
        rows.append({"file": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    for path in [GLOBAL_HANDOFF_PATH, ROADMAP_PATH, STAGE_STATUS_PATH]:
        if path.exists():
            rows.append({"file": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    with (REPORT_DIR / "batch4A_file_sha256.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "sha256", "size_bytes"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print({"event": "read_inputs"}, flush=True)
    dfs = read_inputs()
    print({"event": "build_horizon_decision"}, flush=True)
    horizon_decision = build_horizon_decision(dfs)
    print({"event": "build_feature_freeze"}, flush=True)
    features = build_feature_freeze(dfs)
    print({"event": "compute_split_summary"}, flush=True)
    split_summary = compute_split_summary()
    print({"event": "write_freeze_configs"}, flush=True)
    write_label_freeze()
    write_split_config(split_summary)
    baseline = build_baseline_reference(dfs)
    write_training_plan(horizon_decision, features, baseline, split_summary)
    write_conclusion(horizon_decision, features, baseline)
    write_handoff(features, baseline)
    update_stage_status()
    update_roadmap()
    write_hashes()
    print(
        {
            "out_dir": str(REPORT_DIR),
            "primary_horizon": PRIMARY_HORIZON,
            "primary_features": int(features["training_status"].eq("primary_training_feature").sum()),
            "minimum_baselines": int(baseline["baseline_role"].eq("minimum_to_beat").sum()),
            "gate_result": "pass_to_batch4B_after_review",
            "next_step": "Batch 4B lightweight model training after user review",
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
