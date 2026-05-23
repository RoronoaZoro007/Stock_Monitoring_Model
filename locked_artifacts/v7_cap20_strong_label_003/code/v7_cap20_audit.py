#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import pickle
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from top3000_repaired_model_pipeline import BASE_PARAMS, make_model
from tushare_tail_model import FEATURE_COLUMNS, CostConfig, max_drawdown, profit_factor, topn_by_day


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "reports" / "tushare" / "full_top3000_model_compare"
V7_DIR = MODEL_ROOT / "v7_cap20_strong_label_003"
AUDIT_DIR = ROOT / "reports" / "tushare" / "v7_cap20_audit"
MODEL_FRAME = ROOT / "data_tushare" / "clean" / "features" / "tail_dataset_top3000_model_frame.parquet"
REPAIRED_DAILY = ROOT / "data_tushare" / "clean" / "daily_repaired_top3000.parquet"
MINUTE_AGG = ROOT / "data_tushare" / "clean" / "daily_from_minutes_top3000.parquet"
RANK_FILE = ROOT / "data_tushare" / "manifests" / "liquid_top3000_20251120_20260213.csv"
STOCK_BASIC = ROOT / "data_tushare" / "raw" / "bootstrap" / "stock_basic.parquet"
RAW_AUDIT = ROOT / "reports" / "tushare" / "full_top3000_data_quality" / "raw_audit.json"
VALIDATION_START = "20260224"
TOP_N = 10
RANDOM_SEED = 20260523


def pct(x: Any) -> str:
    try:
        if x is None or pd.isna(x):
            return "NA"
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "NA"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, name: str) -> Path:
    path = AUDIT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_json(obj: Any, name: str) -> Path:
    path = AUDIT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def daily_from_selected(selected: pd.DataFrame, all_dates: list[str], return_col: str = "target_return", top_n: int = TOP_N) -> pd.Series:
    ret = selected.groupby("trade_date")[return_col].mean() if not selected.empty else pd.Series(dtype=float)
    daily = pd.Series(0.0, index=pd.Index(sorted(all_dates), name="trade_date"))
    if not ret.empty:
        daily.loc[ret.index.astype(str)] = ret.to_numpy()
    return daily


def metrics_from_selected(
    pred: pd.DataFrame,
    selected: pd.DataFrame,
    label_col: str = "label_static_003",
    score_col: str = "score",
    return_col: str = "target_return",
) -> dict[str, Any]:
    all_dates = sorted(pred["trade_date"].astype(str).unique())
    daily = daily_from_selected(selected, all_dates, return_col)
    auc = np.nan
    if label_col in pred.columns and pred[label_col].nunique() > 1 and score_col in pred.columns:
        auc = float(roc_auc_score(pred[label_col], pred[score_col]))
    return {
        "rows": int(len(pred)),
        "days": int(len(all_dates)),
        "selected_trades": int(len(selected)),
        "selected_days": int(selected["trade_date"].nunique()) if len(selected) else 0,
        "avg_candidates_per_day": float(pred.groupby("trade_date").size().mean()),
        "min_candidates_per_day": int(pred.groupby("trade_date").size().min()),
        "auc": auc,
        "trade_win_rate": float((selected[return_col] > 0).mean()) if len(selected) else np.nan,
        "daily_win_rate": float((daily > 0).mean()) if len(daily) else np.nan,
        "avg_trade_return": float(selected[return_col].mean()) if len(selected) else np.nan,
        "avg_daily_return": float(daily.mean()) if len(daily) else np.nan,
        "cumulative_return": float((1 + daily).prod() - 1) if len(daily) else np.nan,
        "max_drawdown": max_drawdown(daily),
        "profit_factor": profit_factor(selected[return_col]) if len(selected) else np.nan,
    }


def select_top(pred: pd.DataFrame, score_col: str, n: int = TOP_N, ascending: bool = False) -> pd.DataFrame:
    return pred.sort_values(["trade_date", score_col], ascending=[True, ascending]).groupby("trade_date", group_keys=False).head(n)


def gross_from_net(net: pd.Series, buy_slippage: float = 0.00050, sell_slippage: float = 0.00050) -> pd.Series:
    c = CostConfig(buy_slippage=buy_slippage, sell_slippage=sell_slippage)
    return (1 + net) * (1 + c.buy_slippage + c.buy_commission) / (1 - c.sell_slippage - c.sell_commission - c.stamp_tax) - 1


def net_from_gross(gross: pd.Series, slippage: float) -> pd.Series:
    c = CostConfig(buy_slippage=slippage, sell_slippage=slippage)
    return (1 + gross) * (1 - c.sell_slippage - c.sell_commission - c.stamp_tax) / (1 + c.buy_slippage + c.buy_commission) - 1


def add_cost_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    gross = gross_from_net(out["target_return"])
    out["return_no_cost"] = gross
    out["return_basic_cost_no_slippage"] = net_from_gross(gross, 0.0)
    out["return_5bp_slippage"] = net_from_gross(gross, 0.0005)
    out["return_10bp_slippage"] = net_from_gross(gross, 0.0010)
    out["return_20bp_slippage"] = net_from_gross(gross, 0.0020)
    return out


def consecutive_max(bool_values: list[bool]) -> int:
    best = cur = 0
    for v in bool_values:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def auc_safe(df: pd.DataFrame, label_col: str = "label_static_003", score_col: str = "score") -> float:
    if len(df) == 0 or df[label_col].nunique() < 2:
        return np.nan
    return float(roc_auc_score(df[label_col], df[score_col]))


def bootstrap_auc_ci(df: pd.DataFrame, n: int = 300) -> dict[str, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    y = df["label_static_003"].to_numpy()
    s = df["score"].to_numpy()
    vals = []
    idx = np.arange(len(df))
    for _ in range(n):
        take = rng.choice(idx, size=len(idx), replace=True)
        yy = y[take]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(float(roc_auc_score(yy, s[take])))
    return {
        "auc": auc_safe(df),
        "bootstrap_n": len(vals),
        "ci_2_5": float(np.quantile(vals, 0.025)),
        "ci_50": float(np.quantile(vals, 0.50)),
        "ci_97_5": float(np.quantile(vals, 0.975)),
    }


def feature_lineage() -> pd.DataFrame:
    rows = [
        ("ret_to_prev_close_1450", "14:50 价格相对前收", "close_1450 / prev_close_use - 1", "当日 <=14:50 + 前一交易日 close", "N", "N", "N", "Y"),
        ("intraday_ret_1450", "日内 09:35 至 14:50 收益", "close_1450 / open_0935 - 1", "当日 09:35,14:50", "N", "N", "N", "Y"),
        ("tail_ret_1420_1450", "14:20 至 14:50 尾盘动量", "close_1450 / close_1420 - 1", "当日 14:20,14:50", "N", "N", "N", "Y"),
        ("tail_ret_1430_1450", "14:30 至 14:50 尾盘动量", "close_1450 / close_1430 - 1", "当日 14:30,14:50", "N", "N", "N", "Y"),
        ("late_ret_1445_1450", "14:45 至 14:50 短动量", "close_1450 / close_1445 - 1", "当日 14:45,14:50", "N", "N", "N", "Y"),
        ("afternoon_ret_1305_1450", "午后动量", "close_1450 / close_1305 - 1", "当日 13:05,14:50", "N", "N", "N", "Y"),
        ("vwap_pos_1450", "14:50 相对当日已发生 VWAP", "close_1450 / VWAP(<=14:50) - 1", "当日 <=14:50", "N", "N", "N", "Y"),
        ("close_to_high_sofar", "14:50 距离已发生最高价", "close_1450 / max(high <=14:50) - 1", "当日 <=14:50", "N", "N", "不是全天 high", "Y"),
        ("close_to_low_sofar", "14:50 距离已发生最低价", "close_1450 / min(low <=14:50) - 1", "当日 <=14:50", "N", "N", "不是全天 low", "Y"),
        ("range_sofar", "已发生日内振幅", "max(high <=14:50) / min(low <=14:50) - 1", "当日 <=14:50", "N", "N", "不是全天 high/low", "Y"),
        ("tail_amount_share", "14:30-14:50 成交额占比", "sum(amount 14:30-14:50) / sum(amount <=14:50)", "当日 <=14:50", "N", "N", "不是全天 amount", "Y"),
        ("tail_amount_vs_prev20", "尾盘成交额相对前 20 日均量", "tail_amount / (amount20_prev * 5/48) - 1", "当日 14:30-14:50 + 前 20 交易日 amount", "N", "N", "N", "Y"),
        ("amount_sofar_log", "截至 14:50 成交额", "log1p(sum(amount <=14:50))", "当日 <=14:50", "N", "N", "不是全天 amount", "Y"),
        ("open_gap", "开盘跳空", "open_0935 / prev_close_use - 1", "当日 09:35 + 前收", "N", "N", "N", "Y"),
        ("prev_ret_1d", "前一日收益", "close[t-1]/close[t-2]-1", "t-1 及以前日线", "N", "N", "前日 full day", "Y"),
        ("prev_ret_3d", "前三日收益", "close[t-1]/close[t-4]-1", "t-1 及以前日线", "N", "N", "前日 full day", "Y"),
        ("prev_ret_5d", "前五日收益", "close[t-1]/close[t-6]-1", "t-1 及以前日线", "N", "N", "前日 full day", "Y"),
        ("prev_volatility_20d", "前 20 日波动率", "std(pct_change(close), shift 1, rolling 20)", "t-1 及以前日线", "N", "N", "历史 full day", "Y"),
        ("price_vs_ma5_prev", "14:50 相对前 MA5", "close_1450 / ma5_prev - 1", "当日 14:50 + t-1 及以前日线", "N", "N", "历史 full day", "Y"),
        ("price_vs_ma10_prev", "14:50 相对前 MA10", "close_1450 / ma10_prev - 1", "当日 14:50 + t-1 及以前日线", "N", "N", "历史 full day", "Y"),
        ("price_vs_ma20_prev", "14:50 相对前 MA20", "close_1450 / ma20_prev - 1", "当日 14:50 + t-1 及以前日线", "N", "N", "历史 full day", "Y"),
        ("prev_amount_ratio_5_20", "前 5/20 日成交额比", "amount5_prev / amount20_prev - 1", "t-1 及以前日线", "N", "N", "历史 full day amount", "Y"),
        ("open_auction_ret", "当日开盘集合竞价收益", "open_auction_close / prev_close_use - 1", "当日开盘集合竞价", "N", "N", "N", "Y"),
        ("open_auction_amount_log", "当日开盘集合竞价成交额", "log1p(open_auction_amount)", "当日开盘集合竞价", "N", "N", "N", "Y"),
        ("prev_close_auction_ret", "前日收盘集合竞价收益", "prev_close_auction_close / prev_close_use - 1", "t-1 收盘集合竞价", "N", "N", "前日收盘", "Y"),
        ("prev_close_auction_amount_log", "前日收盘集合竞价成交额", "log1p(prev_close_auction_amount)", "t-1 收盘集合竞价", "N", "N", "前日收盘", "Y"),
        ("prev_net_mf_amount_ratio", "前日主力净流入强度", "prev_net_mf_amount / amount20_prev", "t-1 资金流 + 历史成交额", "N", "N", "前日 full day", "Y"),
        ("prev_elg_net_amount_ratio", "前日超大单净流入强度", "(buy_elg_amount-sell_elg_amount) / amount20_prev", "t-1 资金流 + 历史成交额", "N", "N", "前日 full day", "Y"),
        ("prev_ths_net_amount_ratio", "前日同花顺资金净流入强度", "prev_ths_net_amount / amount20_prev", "t-1 资金流 + 历史成交额", "N", "N", "前日 full day", "Y"),
        ("market_ret_median_1450", "当日市场 14:50 截面中位收益", "median(ret_to_prev_close_1450 over candidates)", "当日全候选 <=14:50", "N", "N", "N", "Y, 需全市场实时分钟数据"),
        ("market_breadth_positive", "当日市场 14:50 上涨比例", "mean(ret_to_prev_close_1450>0 over candidates)", "当日全候选 <=14:50", "N", "N", "N", "Y, 需全市场实时分钟数据"),
        ("market_tail_ret_median", "当日市场尾盘动量中位数", "median(tail_ret_1420_1450 over candidates)", "当日全候选 <=14:50", "N", "N", "N", "Y, 需全市场实时分钟数据"),
        ("market_tail_breadth_positive", "当日市场尾盘上涨比例", "mean(tail_ret_1420_1450>0 over candidates)", "当日全候选 <=14:50", "N", "N", "N", "Y, 需全市场实时分钟数据"),
        ("market_amount_log", "候选池截至 14:50 总成交额", "log1p(sum(exp(amount_sofar_log)-1))", "当日全候选 <=14:50", "N", "N", "不是全天 amount", "Y, 需全市场实时分钟数据"),
    ]
    return pd.DataFrame(rows, columns=["field", "meaning", "formula", "data_time_range", "uses_after_1500_today", "uses_next_day", "uses_full_day_high_low_close_amount", "available_before_signal"])


def drawdown_window(daily: pd.Series) -> tuple[str, str, float]:
    equity = (1 + daily.fillna(0)).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    end = dd.idxmin()
    start = equity.loc[:end].idxmax()
    return str(start), str(end), float(dd.loc[end])


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(V7_DIR / "validation_predictions.csv", dtype={"trade_date": str, "ts_code": str})
    selected = pd.read_csv(V7_DIR / "top10_validation.csv", dtype={"trade_date": str, "ts_code": str})
    selected = add_cost_columns(selected)
    pred = add_cost_columns(pred)
    all_dates = sorted(pred["trade_date"].unique())
    model_bundle = pickle.load((V7_DIR / "model.pkl").open("rb"))
    feature_cols = model_bundle["feature_columns"]
    label_col = model_bundle["label_col"]

    stock = pd.read_parquet(STOCK_BASIC)
    stock["is_st"] = stock["name"].astype(str).str.contains("ST", case=False, na=False)
    rank = pd.read_csv(RANK_FILE, dtype={"ts_code": str})
    daily = pd.read_parquet(REPAIRED_DAILY)
    daily["trade_date"] = daily["trade_date"].astype(str)

    # P0 execution contract.
    execution_contract = pd.DataFrame(
        [
            ("model_type", "尾盘隔夜选股模型", "用 14:50 前可得特征打分，假设 14:55 5分钟 bar VWAP 买入，次日上午择机卖出。"),
            ("signal_time", "14:50 bar 完成后、14:55 执行前", "特征最多使用到 14:50；entry_vwap 使用 14:55 bar，只作为回测成交假设。"),
            ("buy_price", "当日 14:55 5分钟 bar VWAP", "vwap(amount, vol, close, low, high)，见 tushare_tail_model.py:373。"),
            ("sell_price", "次日 09:40/09:50/10:05/10:30 对应 bar VWAP", "由 09:35/09:45/10:00 检查止盈止损后执行；未触发则 10:30。"),
            ("target_return", "净收益", "sell_vwap*(1-sell_slippage-sell_commission-stamp_tax)/(entry_vwap*(1+buy_slippage+buy_commission))-1。"),
            ("top_n", "10", "每天按 score 降序取 Top10。"),
            ("weighting", "等权", "日收益=Top10 target_return 算术平均。"),
            ("candidate_lt_10", "按实际入选数量等权", "当前验证期最少候选数 > 10，因此未触发。"),
            ("limit_suspend_handling", "部分处理", "14:50 接近涨跌停过滤；停牌无分钟线则不生成样本；未显式模拟涨停买不进/跌停卖不出。"),
            ("cost_included", "已包含基础交易成本", "佣金、印花税、基础滑点已计入；过户费/冲击成本未计入。"),
        ],
        columns=["item", "current_setting", "evidence"],
    )
    write_csv(execution_contract, "p0_execution_contract.csv")

    # Universe audit.
    universe = pd.DataFrame(
        [
            ("pool_size", int(rank["ts_code"].nunique()), "liquid_top3000_20251120_20260213.csv"),
            ("ranking_type", "静态", "不是每日滚动股票池。"),
            ("rank_start", str(rank["rank_start"].min()), "排名文件字段 rank_start。"),
            ("rank_end", str(rank["rank_end"].max()), "排名文件字段 rank_end。"),
            ("validation_start", VALIDATION_START, "验证集起点。"),
            ("uses_after_validation_start", False, "rank_end=20260213 < validation_start=20260224。"),
            ("uses_future_for_early_training_dates", True, "对 2025-11-20 以前训练样本，静态池使用了其后的流动性信息。"),
            ("liquidity_metric", "avg_amount", "排名文件字段 avg_amount。"),
            ("liquidity_window_days", int(rank["days"].median()), "排名文件字段 days，当前为 60。"),
            ("st_handling", "未从股票池剔除；特征生成时按 ST 5% 涨跌幅阈值过滤近涨跌停样本", "code_limit_threshold。"),
            ("new_delist_suspend_handling", "上市前/停牌无分钟线不生成样本；退市状态未做独立过滤", "样本生成依赖分钟线和 daily。"),
        ],
        columns=["item", "value", "evidence"],
    )
    write_csv(universe, "p0_universe_audit.csv")

    # Feature lineage.
    lineage = feature_lineage()
    write_csv(lineage, "p0_feature_lineage.csv")

    # Target and thresholds.
    costs = CostConfig()
    write_json(
        {
            "target_return_formula": "sell_vwap*(1-sell_slippage-sell_commission-stamp_tax)/(entry_vwap*(1+buy_slippage+buy_commission))-1",
            "entry_vwap": "当日 14:55 5分钟 bar VWAP",
            "sell_vwap": "次日出场执行 bar VWAP: 09:40/09:50/10:05/10:30",
            "cost_config": costs.__dict__,
            "label_v7": "label_static_003 = int(target_return > 0.003)",
            "threshold_reason": "0.3% 来自预先提出的交易摩擦门槛：20bp 成本 + 约10bp 期望利润；本轮 v7 选择使用了验证结果。",
        },
        "p0_target_definition.json",
    )

    # Daily repair missing rows.
    minute_agg = pd.read_parquet(MINUTE_AGG)
    raw_daily = daily[daily["daily_source"].eq("minute_reconstructed")][
        ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "pre_close", "change", "pct_chg", "next_trade_date"]
    ].copy()
    raw_daily = raw_daily.merge(stock[["ts_code", "name", "industry", "market", "list_date", "delist_date"]], on="ts_code", how="left")
    write_csv(raw_daily.sort_values(["ts_code", "trade_date"]), "p0_daily_repair_2173_missing_days.csv")

    # Repair before/after target comparison where possible.
    repaired_raw = pd.read_parquet(ROOT / "data_tushare" / "clean" / "features" / "tail_dataset_top3000_repaired_raw.parquet", columns=["ts_code", "trade_date", "target_return"])
    old_paths = [ROOT / "data_tushare" / "features" / "stages" / "tail_dataset_top2000.parquet", ROOT / "data_tushare" / "features" / "stages" / "tail_dataset_top1000.parquet", ROOT / "data_tushare" / "features" / "stages" / "tail_dataset_top500.parquet"]
    old_parts = [pd.read_parquet(p, columns=["ts_code", "trade_date", "target_return"]) for p in old_paths if p.exists()]
    if old_parts:
        old = pd.concat(old_parts, ignore_index=True).drop_duplicates(["ts_code", "trade_date"], keep="last").rename(columns={"target_return": "target_return_before_repair"})
        cmp = repaired_raw.merge(old, on=["ts_code", "trade_date"], how="inner")
        cmp["target_return_after_repair"] = cmp["target_return"]
        cmp["delta"] = cmp["target_return_after_repair"] - cmp["target_return_before_repair"]
        write_csv(cmp[cmp["delta"].abs().gt(1e-12)].sort_values("delta"), "p0_target_return_before_after_repair_available_overlap.csv")

    # Extreme samples.
    model_frame = pd.read_parquet(MODEL_FRAME)
    cap20_df = model_frame[model_frame["target_return"].abs().le(0.20)].copy()
    cap20_df["trade_date"] = cap20_df["trade_date"].astype(str)
    extreme = model_frame[model_frame["target_return"].abs().gt(0.20)].copy()
    extreme["extreme_side"] = np.where(extreme["target_return"] > 0, "positive", "negative")
    extreme = extreme.merge(stock[["ts_code", "name", "industry", "market", "list_date", "is_st"]], on="ts_code", how="left")
    write_csv(extreme.sort_values("target_return"), "p0_extreme_target_return_391.csv")
    split_counts = []
    for split, d in [("train", model_frame[model_frame["trade_date"].astype(str) < VALIDATION_START]), ("validation", model_frame[model_frame["trade_date"].astype(str) >= VALIDATION_START])]:
        e = d[d["target_return"].abs().gt(0.20)]
        split_counts.append(
            {
                "split": split,
                "rows": int(len(d)),
                "extreme_abs_gt20": int(len(e)),
                "positive_extreme": int((e["target_return"] > 0).sum()),
                "negative_extreme": int((e["target_return"] < 0).sum()),
            }
        )
    write_csv(pd.DataFrame(split_counts), "p0_extreme_counts_by_split.csv")

    # Existing variant metrics and threshold sensitivity.
    comparison = pd.read_csv(MODEL_ROOT / "comparison_metrics.csv")
    write_csv(comparison, "p0_model_versions_v1_v8.csv")
    threshold_rows = []
    for threshold in [0.0, 0.003, 0.005, 0.010]:
        name = f"threshold_{threshold:.3f}"
        label_tmp = f"label_tmp_{threshold:.3f}"
        cap20_df[label_tmp] = (cap20_df["target_return"] > threshold).astype(int)
        train = cap20_df[cap20_df["trade_date"] < VALIDATION_START]
        val = cap20_df[cap20_df["trade_date"] >= VALIDATION_START]
        model = make_model(BASE_PARAMS).fit(train[FEATURE_COLUMNS], train[label_tmp])
        val_tmp = val.copy()
        val_tmp["score"] = model.predict_proba(val_tmp[FEATURE_COLUMNS])[:, 1]
        top = select_top(val_tmp, "score", TOP_N)
        row = {"threshold": threshold, "positive_rate_train": float(train[label_tmp].mean()), "positive_rate_validation": float(val[label_tmp].mean())}
        row.update(metrics_from_selected(val_tmp, top, label_tmp))
        threshold_rows.append(row)
    write_csv(pd.DataFrame(threshold_rows), "p0_label_threshold_sensitivity.csv")

    # Cost / slippage stress.
    stress_rows = []
    for col, label in [
        ("return_no_cost", "no_cost"),
        ("return_basic_cost_no_slippage", "basic_cost_no_slippage"),
        ("return_5bp_slippage", "base_5bp_slippage"),
        ("return_10bp_slippage", "10bp_slippage"),
        ("return_20bp_slippage", "20bp_slippage"),
    ]:
        row = {"cost_case": label}
        row.update(metrics_from_selected(pred, selected, return_col=col))
        stress_rows.append(row)
    write_csv(pd.DataFrame(stress_rows), "p0_p1_cost_slippage_stress.csv")

    # Limit/suspend/fill diagnostics.
    limit_threshold = np.where(selected["ts_code"].str.split(".").str[0].str.startswith(("300", "301", "688")), 0.20, 0.10)
    limit_threshold = np.where(selected["ts_code"].isin(set(stock.loc[stock["is_st"], "ts_code"])), 0.05, limit_threshold)
    fill_diag = pd.DataFrame(
        [
            ("near_up_limit_1450_selected", int((selected["ret_to_prev_close_1450"] > limit_threshold - 0.008).sum()), "样本生成阶段已过滤，理论应为 0。"),
            ("near_down_limit_1450_selected", int((selected["ret_to_prev_close_1450"] < -limit_threshold + 0.008).sum()), "样本生成阶段已过滤，理论应为 0。"),
            ("missing_exit_selected", int(selected["exit_time"].isna().sum()), "有样本才说明次日存在可用退出分钟线。"),
            ("min_amount_sofar_filter", 20_000_000, "候选生成时 amount_sofar >= 2000万。"),
            ("explicit_limit_fill_model", "NO", "未建模涨停买不进/跌停卖不出；使用 VWAP 理想成交。"),
            ("impact_cost_model", "NO", "未建模冲击成本。"),
        ],
        columns=["item", "value", "evidence"],
    )
    write_csv(fill_diag, "p0_limit_suspend_fill_diagnostics.csv")

    # Random benchmark.
    rng = np.random.default_rng(RANDOM_SEED)
    groups = {d: g["target_return"].to_numpy() for d, g in pred.groupby("trade_date")}
    random_rows = []
    for i in range(1000):
        daily_returns = []
        for d in all_dates:
            arr = groups[d]
            pick = arr if len(arr) <= TOP_N else arr[rng.choice(len(arr), size=TOP_N, replace=False)]
            daily_returns.append(float(np.mean(pick)))
        s = pd.Series(daily_returns, index=all_dates)
        random_rows.append(
            {
                "run": i,
                "avg_daily_return": float(s.mean()),
                "cumulative_return": float((1 + s).prod() - 1),
                "daily_win_rate": float((s > 0).mean()),
                "max_drawdown": max_drawdown(s),
            }
        )
    random_df = pd.DataFrame(random_rows)
    model_cum = metrics_from_selected(pred, selected)["cumulative_return"]
    random_summary = random_df[["avg_daily_return", "cumulative_return", "daily_win_rate", "max_drawdown"]].agg(
        ["mean", "median", "min", "max", lambda s: s.quantile(0.05), lambda s: s.quantile(0.95)]
    )
    random_summary.index = ["mean", "median", "min", "max", "q05", "q95"]
    random_summary.loc["model_percentile_cumulative_return", :] = np.nan
    random_summary.loc["model_percentile_cumulative_return", "cumulative_return"] = float((random_df["cumulative_return"] < model_cum).mean())
    write_csv(random_df, "p0_random_top10_1000_runs.csv")
    write_csv(random_summary.reset_index(names="stat"), "p0_random_top10_summary.csv")

    # Daily returns and concentration.
    daily_net = daily_from_selected(selected, all_dates)
    daily_df = daily_net.reset_index(name="daily_return")
    daily_df["equity"] = (1 + daily_df["daily_return"]).cumprod()
    daily_df["drawdown"] = daily_df["equity"] / daily_df["equity"].cummax() - 1
    write_csv(daily_df, "p1_daily_returns_58d.csv")
    sorted_gain = daily_df.sort_values("daily_return", ascending=False)
    concentration = pd.DataFrame(
        [
            ("max_gain_day", sorted_gain.iloc[0]["trade_date"], float(sorted_gain.iloc[0]["daily_return"])),
            ("max_loss_day", daily_df.sort_values("daily_return").iloc[0]["trade_date"], float(daily_df.sort_values("daily_return").iloc[0]["daily_return"])),
            ("top3_gain_sum", "", float(sorted_gain.head(3)["daily_return"].sum())),
            ("top5_gain_sum", "", float(sorted_gain.head(5)["daily_return"].sum())),
            ("cum_without_top1_gain", "", float((1 + daily_df.drop(sorted_gain.head(1).index)["daily_return"]).prod() - 1)),
            ("cum_without_top3_gain", "", float((1 + daily_df.drop(sorted_gain.head(3).index)["daily_return"]).prod() - 1)),
            ("cum_without_top5_gain", "", float((1 + daily_df.drop(sorted_gain.head(5).index)["daily_return"]).prod() - 1)),
        ],
        columns=["metric", "date", "value"],
    )
    write_csv(concentration, "p1_return_concentration.csv")

    # Score deciles and TopN.
    pred["score_decile"] = pred.groupby("trade_date")["score"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
    decile_rows = []
    for decile, g in pred.groupby("score_decile"):
        # decile 10 is highest because rank ascending labels; score itself ranked ascending by default.
        daily_dec = g.groupby("trade_date")["target_return"].mean().reindex(all_dates).fillna(0)
        decile_rows.append(
            {
                "score_decile": int(decile),
                "samples": int(len(g)),
                "avg_target_return": float(g["target_return"].mean()),
                "win_rate": float((g["target_return"] > 0).mean()),
                "cumulative_return_equal_decile_daily": float((1 + daily_dec).prod() - 1),
                "max_drawdown": max_drawdown(daily_dec),
                "profit_factor": profit_factor(g["target_return"]),
            }
        )
    write_csv(pd.DataFrame(decile_rows), "p1_score_deciles.csv")
    topn_rows = []
    for n in [1, 3, 5, 10, 20]:
        top = select_top(pred, "score", n)
        row = {"top_n": n}
        row.update(metrics_from_selected(pred, top))
        topn_rows.append(row)
    write_csv(pd.DataFrame(topn_rows), "p1_topn_sensitivity.csv")
    corr_rows = pd.DataFrame(
        [
            {"metric": "pearson_score_target_return", "value": float(pred["score"].corr(pred["target_return"], method="pearson"))},
            {"metric": "spearman_score_target_return", "value": float(pred["score"].corr(pred["target_return"], method="spearman"))},
        ]
    )
    write_csv(corr_rows, "p1_score_return_correlation.csv")

    # AUC splits.
    auc_month = pred.groupby(pred["trade_date"].str[:6]).apply(lambda g: auc_safe(g), include_groups=False).reset_index()
    auc_month.columns = ["month", "auc"]
    write_csv(auc_month, "p1_auc_by_month.csv")
    env = pred.copy()
    env["market_direction"] = np.where(env["market_ret_median_1450"] > 0, "market_up", "market_down")
    env["tail_direction"] = np.where(env["market_tail_ret_median"] > 0, "tail_up", "tail_down")
    env["amount_bucket"] = pd.qcut(env["amount_sofar_log"], 5, labels=[f"q{i}" for i in range(1, 6)])
    env["board"] = np.select(
        [env["ts_code"].str.startswith(("300", "301")), env["ts_code"].str.startswith("688")],
        ["创业板", "科创板"],
        default="主板/其他",
    )
    auc_split_rows = []
    for group_col in ["market_direction", "tail_direction", "board", "amount_bucket"]:
        for key, g in env.groupby(group_col, observed=False):
            auc_split_rows.append({"group_type": group_col, "group": str(key), "rows": int(len(g)), "auc": auc_safe(g)})
    write_csv(pd.DataFrame(auc_split_rows), "p1_auc_splits.csv")
    write_json(bootstrap_auc_ci(pred), "p1_auc_bootstrap_ci.json")

    # Profit factor details.
    gains = selected.loc[selected["target_return"] > 0, "target_return"]
    losses = selected.loc[selected["target_return"] < 0, "target_return"]
    pf_details = {
        "gross_profit_sum": float(gains.sum()),
        "gross_loss_abs_sum": float(-losses.sum()),
        "avg_winning_trade": float(gains.mean()),
        "avg_losing_trade": float(losses.mean()),
        "win_loss_ratio_abs": float(gains.mean() / abs(losses.mean())),
        "max_single_gain": float(selected["target_return"].max()),
        "max_single_loss": float(selected["target_return"].min()),
        "max_consecutive_losing_trades_in_selected_order": consecutive_max((selected.sort_values(["trade_date", "score"], ascending=[True, False])["target_return"] < 0).tolist()),
        "max_consecutive_losing_days": consecutive_max((daily_net < 0).tolist()),
    }
    write_json(pf_details, "p1_profit_factor_details.json")

    # Market environment splits for selected trades.
    selected_env = selected.copy()
    selected_env["market_direction"] = np.where(selected_env["market_ret_median_1450"] > 0, "market_up", "market_down")
    selected_env["volume_regime"] = np.where(
        selected_env.groupby("trade_date")["market_amount_log"].transform("first") >= pred.groupby("trade_date")["market_amount_log"].first().median(),
        "market_high_amount",
        "market_low_amount",
    )
    selected_env["volatility_regime"] = np.where(
        selected_env.groupby("trade_date")["market_tail_ret_median"].transform("first").abs()
        >= pred.groupby("trade_date")["market_tail_ret_median"].first().abs().median(),
        "high_tail_vol_proxy",
        "low_tail_vol_proxy",
    )
    env_metric_rows = []
    for col in ["market_direction", "volume_regime", "volatility_regime"]:
        for key, trades in selected_env.groupby(col):
            sub_pred = pred[pred["trade_date"].isin(trades["trade_date"].unique())]
            row = {"environment_type": col, "environment": key}
            row.update(metrics_from_selected(sub_pred, trades))
            env_metric_rows.append(row)
    write_csv(pd.DataFrame(env_metric_rows), "p1_market_environment_metrics.csv")

    # Industry and stock contribution.
    selected_ind = selected.merge(stock[["ts_code", "name", "industry", "market"]], on="ts_code", how="left", suffixes=("", "_basic"))
    industry_rows = []
    for industry, g in selected_ind.groupby("industry"):
        daily_ind = g.groupby("trade_date")["target_return"].mean().reindex(all_dates).fillna(0)
        industry_rows.append(
            {
                "industry": industry,
                "signals": int(len(g)),
                "win_rate": float((g["target_return"] > 0).mean()),
                "avg_return": float(g["target_return"].mean()),
                "sum_return_contribution": float(g["target_return"].sum() / TOP_N),
                "cumulative_daily_if_industry_only": float((1 + daily_ind).prod() - 1),
                "max_drawdown": max_drawdown(daily_ind),
            }
        )
    write_csv(pd.DataFrame(industry_rows).sort_values("sum_return_contribution", ascending=False), "p1_industry_contribution.csv")
    stock_contrib = (
        selected_ind.groupby(["ts_code", "name", "industry"])
        .agg(signals=("target_return", "size"), win_rate=("target_return", lambda s: float((s > 0).mean())), avg_return=("target_return", "mean"), sum_return=("target_return", "sum"))
        .reset_index()
    )
    stock_contrib["portfolio_contribution"] = stock_contrib["sum_return"] / TOP_N
    write_csv(stock_contrib.sort_values("portfolio_contribution", ascending=False).head(30), "p1_top_profit_stocks.csv")
    write_csv(stock_contrib.sort_values("portfolio_contribution", ascending=True).head(30), "p1_top_loss_stocks.csv")

    # Time split and reproducibility.
    raw_audit = json.loads(RAW_AUDIT.read_text(encoding="utf-8"))
    time_split = pd.DataFrame(
        [
            ("raw_minute", raw_audit["coverage"]["top3000_symbols"], "20240521", "20260521", "原始分钟线审计"),
            ("raw_daily", int((daily["daily_source"] == "raw_daily").sum()), str(daily.loc[daily["daily_source"] == "raw_daily", "trade_date"].min()), str(daily.loc[daily["daily_source"] == "raw_daily", "trade_date"].max()), "raw daily within repaired file"),
            ("clean_daily", len(daily), str(daily["trade_date"].min()), str(daily["trade_date"].max()), "daily_repaired_top3000.parquet"),
            ("train", int((cap20_df["trade_date"] < VALIDATION_START).sum()), str(cap20_df.loc[cap20_df["trade_date"] < VALIDATION_START, "trade_date"].min()), str(cap20_df.loc[cap20_df["trade_date"] < VALIDATION_START, "trade_date"].max()), "v7 train"),
            ("validation", int((cap20_df["trade_date"] >= VALIDATION_START).sum()), str(cap20_df.loc[cap20_df["trade_date"] >= VALIDATION_START, "trade_date"].min()), str(cap20_df.loc[cap20_df["trade_date"] >= VALIDATION_START, "trade_date"].max()), "v7 validation"),
            ("unused_test", 0, "", "", "当前没有完全未使用测试集"),
        ],
        columns=["dataset", "rows_or_symbols", "start_date", "end_date", "note"],
    )
    write_csv(time_split, "p1_time_split.csv")
    repro = {
        "git_commit": None,
        "git_status": "not_a_git_repository",
        "python": sys.version,
        "platform": platform.platform(),
        "random_seed": RANDOM_SEED,
        "model_params": BASE_PARAMS,
        "validation_start": VALIDATION_START,
        "top_n": TOP_N,
        "feature_columns": feature_cols,
        "hashes": {
            "validation_predictions.csv": sha256_file(V7_DIR / "validation_predictions.csv"),
            "top10_validation.csv": sha256_file(V7_DIR / "top10_validation.csv"),
            "model.pkl": sha256_file(V7_DIR / "model.pkl"),
            "model_frame.parquet": sha256_file(MODEL_FRAME),
            "daily_repaired_top3000.parquet": sha256_file(REPAIRED_DAILY),
            "rank_file": sha256_file(RANK_FILE),
        },
        "reproduce_commands": [
            ".venv/bin/python top3000_repaired_model_pipeline.py",
            ".venv/bin/python v7_cap20_audit.py",
        ],
    }
    try:
        freeze = subprocess.check_output([str(ROOT / ".venv" / "bin" / "python"), "-m", "pip", "freeze"], text=True, timeout=30)
        (AUDIT_DIR / "python_freeze.txt").write_text(freeze, encoding="utf-8")
    except Exception as exc:
        repro["pip_freeze_error"] = repr(exc)
    write_json(repro, "p1_reproducibility.json")

    # Feature importance and ablations.
    train = cap20_df[cap20_df["trade_date"] < VALIDATION_START].copy()
    val = cap20_df[cap20_df["trade_date"] >= VALIDATION_START].copy()
    val_scored = pred.copy()
    sample = val.sample(n=min(50000, len(val)), random_state=RANDOM_SEED)
    perm = permutation_importance(model_bundle["model"], sample[feature_cols], sample[label_col], n_repeats=3, random_state=RANDOM_SEED, scoring="roc_auc", n_jobs=1)
    imp = pd.DataFrame({"feature": feature_cols, "importance_mean_auc_drop": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean_auc_drop", ascending=False)
    write_csv(imp, "p2_feature_importance_permutation_auc_sample50k.csv")
    top5_features = imp.head(5)["feature"].tolist()
    feature_groups = {
        "all_features_v7": feature_cols,
        "drop_top5_importance": [c for c in feature_cols if c not in top5_features],
        "volume_only": ["tail_amount_share", "tail_amount_vs_prev20", "amount_sofar_log", "prev_amount_ratio_5_20", "open_auction_amount_log", "prev_close_auction_amount_log", "prev_net_mf_amount_ratio", "prev_elg_net_amount_ratio", "prev_ths_net_amount_ratio", "market_amount_log"],
        "price_momentum_only": ["ret_to_prev_close_1450", "intraday_ret_1450", "tail_ret_1420_1450", "tail_ret_1430_1450", "late_ret_1445_1450", "afternoon_ret_1305_1450", "open_gap", "prev_ret_1d", "prev_ret_3d", "prev_ret_5d"],
        "intraday_position_only": ["vwap_pos_1450", "close_to_high_sofar", "close_to_low_sofar", "range_sofar", "price_vs_ma5_prev", "price_vs_ma10_prev", "price_vs_ma20_prev"],
        "without_market_cross_section": [c for c in feature_cols if not c.startswith("market_")],
    }
    ablation_rows = []
    for name, cols in feature_groups.items():
        cols = [c for c in cols if c in train.columns]
        model = make_model(BASE_PARAMS).fit(train[cols], train[label_col])
        tmp = val.copy()
        tmp["score"] = model.predict_proba(tmp[cols])[:, 1]
        top = select_top(tmp, "score", TOP_N)
        row = {"feature_set": name, "feature_count": len(cols)}
        row.update(metrics_from_selected(tmp, top))
        ablation_rows.append(row)
    write_csv(pd.DataFrame(ablation_rows), "p2_feature_ablation.csv")

    # Simple rule benchmarks.
    rules: list[tuple[str, str, bool]] = [
        ("ai_model_v7", "score", False),
        ("late_ret_1445_1450_top10", "late_ret_1445_1450", False),
        ("tail_ret_1430_1450_top10", "tail_ret_1430_1450", False),
        ("tail_amount_vs_prev20_top10", "tail_amount_vs_prev20", False),
        ("intraday_ret_1450_top10", "intraday_ret_1450", False),
        ("close_to_high_sofar_top10", "close_to_high_sofar", False),
        ("amount_sofar_log_top10", "amount_sofar_log", False),
    ]
    pred["combo_amount_momentum_score"] = pred["tail_ret_1430_1450"].rank(pct=True) + pred["tail_amount_vs_prev20"].rank(pct=True)
    rules.append(("amount_momentum_combo_top10", "combo_amount_momentum_score", False))
    bench_rows = []
    for name, score_col, asc in rules:
        top = select_top(pred, score_col, TOP_N, asc)
        row = {"benchmark": name, "score_col": score_col}
        row.update(metrics_from_selected(pred, top, score_col=score_col if name != "ai_model_v7" else "score"))
        bench_rows.append(row)
    random_base = pd.read_csv(AUDIT_DIR / "p0_random_top10_summary.csv")
    bench_rows.append({"benchmark": "random_top10_mean_1000", "score_col": "random", "cumulative_return": float(random_df["cumulative_return"].mean()), "avg_daily_return": float(random_df["avg_daily_return"].mean()), "daily_win_rate": float(random_df["daily_win_rate"].mean()), "max_drawdown": float(random_df["max_drawdown"].mean()), "profit_factor": np.nan, "trade_win_rate": np.nan, "auc": np.nan})
    write_csv(pd.DataFrame(bench_rows), "p2_rule_benchmarks.csv")

    # Signal and capacity.
    repeat = selected_ind.groupby(["ts_code", "name", "industry"]).size().reset_index(name="selected_count").sort_values("selected_count", ascending=False)
    write_csv(repeat.head(50), "p2_high_frequency_selected_stocks.csv")
    signal_turnover = pd.DataFrame(
        [
            ("validation_total_trades", len(selected)),
            ("days", len(all_dates)),
            ("avg_candidates_per_day", pred.groupby("trade_date").size().mean()),
            ("avg_selected_per_day", selected.groupby("trade_date").size().mean()),
            ("min_selected_per_day", selected.groupby("trade_date").size().min()),
            ("max_selected_per_day", selected.groupby("trade_date").size().max()),
            ("daily_turnover_assumption", 1.0),
            ("holding_period", "当日 14:55 至次日 09:40/09:50/10:05/10:30"),
        ],
        columns=["metric", "value"],
    )
    write_csv(signal_turnover, "p2_signal_turnover.csv")
    # Capacity uses available feature amounts; tail_5m exact buy-bar amount is not stored in feature frame.
    cap = selected[["trade_date", "ts_code", "amount_sofar_log", "tail_amount_share", "tail_amount_vs_prev20", "target_return"]].copy()
    cap["amount_sofar"] = np.expm1(cap["amount_sofar_log"])
    cap["tail_1430_1450_amount"] = cap["amount_sofar"] * cap["tail_amount_share"]
    for capital in [100_000, 500_000, 1_000_000, 5_000_000]:
        per_trade = capital / TOP_N
        cap[f"buy_{capital}_pct_amount_sofar"] = per_trade / cap["amount_sofar"]
        cap[f"buy_{capital}_pct_tail_1430_1450"] = per_trade / cap["tail_1430_1450_amount"]
    write_csv(cap, "p2_capacity_by_trade.csv")
    cap_summary_rows = []
    for capital in [100_000, 500_000, 1_000_000, 5_000_000]:
        for denom in ["amount_sofar", "tail_1430_1450"]:
            col = f"buy_{capital}_pct_{denom}"
            cap_summary_rows.append(
                {
                    "capital": capital,
                    "denominator": denom,
                    "median_pct": float(cap[col].median()),
                    "p90_pct": float(cap[col].quantile(0.90)),
                    "p99_pct": float(cap[col].quantile(0.99)),
                    "max_pct": float(cap[col].max()),
                    "count_gt_1pct": int((cap[col] > 0.01).sum()),
                    "count_gt_5pct": int((cap[col] > 0.05).sum()),
                    "count_gt_10pct": int((cap[col] > 0.10).sum()),
                }
            )
    write_csv(pd.DataFrame(cap_summary_rows), "p2_capacity_summary.csv")

    # Drawdown.
    dd_start, dd_end, dd_val = drawdown_window(daily_net)
    dd_daily = daily_df[(daily_df["trade_date"] >= dd_start) & (daily_df["trade_date"] <= dd_end)].copy()
    dd_trades = selected_ind[(selected_ind["trade_date"] >= dd_start) & (selected_ind["trade_date"] <= dd_end)].copy()
    write_csv(dd_daily, "p2_max_drawdown_daily.csv")
    write_csv(dd_trades, "p2_max_drawdown_trades.csv")
    write_json(
        {
            "drawdown_start": dd_start,
            "drawdown_end": dd_end,
            "max_drawdown": dd_val,
            "duration_trading_days": int(len(dd_daily)),
            "main_loss_stocks": dd_trades.groupby(["ts_code", "name"]).target_return.sum().sort_values().head(20).reset_index().to_dict(orient="records"),
            "market_proxy_return_sum": float(pred.drop_duplicates("trade_date").set_index("trade_date").loc[dd_daily["trade_date"], "market_ret_median_1450"].sum()),
        },
        "p2_max_drawdown_summary.json",
    )
    write_csv(pd.DataFrame([{"item": "v7_walk_forward_done", "value": False, "note": "当前 v7 是固定参数训练集拟合 + 最近三个月验证；未完成独立滚动训练前向验证。此前 top500/top1000/top2000 基础模型做过训练期 walk-forward，但不等同于 v7。"}]), "p2_walk_forward_status.csv")

    # Train/validation AUC.
    model = model_bundle["model"]
    train_scored = train.copy()
    train_scored["score"] = model.predict_proba(train_scored[feature_cols])[:, 1]
    auc_tv = pd.DataFrame(
        [
            {"split": "train_in_sample", "rows": len(train_scored), "auc": auc_safe(train_scored)},
            {"split": "validation", "rows": len(pred), "auc": auc_safe(pred)},
            {"split": "unused_test", "rows": 0, "auc": np.nan},
        ]
    )
    write_csv(auc_tv, "p1_auc_train_validation_test.csv")

    # Markdown answer index.
    answer = f"""# v7_cap20_strong_label_003 审计包

生成目录：`{AUDIT_DIR}`

## P0 硬伤结论

| 项 | 结论 | 证据文件 |
|---|---|---|
| 模型类型 | 尾盘隔夜选股模型，不是纯日内尾盘套利 | `p0_execution_contract.csv` |
| 交易成本 | 5.63% 为基础成本 + 5bp 单边滑点后的净收益 | `p0_p1_cost_slippage_stress.csv` |
| 股票池 | 静态 Top3000，排名区间 20251120-20260213；不含验证期 20260224 之后，但对早期训练期存在股票池前视 | `p0_universe_audit.csv` |
| 验证集调参 | v7 是看过 v1-v8 验证结果后选出的；当前没有完全未使用测试集 | `p0_model_versions_v1_v8.csv` |
| 未来函数 | 34 个基础因子表逐项列出；未使用当日 15:00 后或次日数据，但市场截面因子需要 14:50 前全市场数据 | `p0_feature_lineage.csv` |
| 日线修复 | 2173 个 raw daily 缺失交易日已由分钟线聚合修复 | `p0_daily_repair_2173_missing_days.csv` |
| 极端收益 | `abs(target_return)>20%` 共 {len(extreme)} 行，正 {int((extreme['target_return']>0).sum())}、负 {int((extreme['target_return']<0).sum())} | `p0_extreme_target_return_391.csv` |
| 涨跌停/成交 | 当前仅过滤 14:50 近涨跌停样本，未显式模拟涨停买不进/跌停卖不出/冲击成本 | `p0_limit_suspend_fill_diagnostics.csv` |

## P1 模拟盘准入结论

| 项 | 当前结果 |
|---|---:|
| v7 验证期累计收益 | {pct(metrics_from_selected(pred, selected)['cumulative_return'])} |
| v7 验证期最大回撤 | {pct(metrics_from_selected(pred, selected)['max_drawdown'])} |
| v7 验证期 PF | {metrics_from_selected(pred, selected)['profit_factor']:.4f} |
| 随机 Top10 累计收益 95 分位 | {pct(random_df['cumulative_return'].quantile(0.95))} |
| 模型累计收益在 1000 次随机中的百分位 | {float((random_df['cumulative_return'] < model_cum).mean()):.3f} |
| 最大回撤区间 | {dd_start} 至 {dd_end} |

模拟盘前必须补：完全未使用测试集、涨跌停可成交性、冲击成本、滚动前向验证。

## P2 优化文件索引

- 因子表：`p0_feature_lineage.csv`
- 因子重要性：`p2_feature_importance_permutation_auc_sample50k.csv`
- 因子组消融：`p2_feature_ablation.csv`
- 简单规则基准：`p2_rule_benchmarks.csv`
- 容量测算：`p2_capacity_summary.csv`
- 最大回撤拆解：`p2_max_drawdown_summary.json`
"""
    (AUDIT_DIR / "AUDIT_INDEX.md").write_text(answer, encoding="utf-8")
    print(json.dumps({"audit_dir": str(AUDIT_DIR), "files": len(list(AUDIT_DIR.glob('*')))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
