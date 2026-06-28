import numpy as np
import pandas as pd

from .config import (
    ACCEL_LOOKBACK_MONTHS,
    BACKTEST_START,
    COMBINED_ACCEL_WEIGHT,
    DAILY_PRICES_FILE,
    FILTER_TOP_N,
    MOMENTUM_LOOKBACK_MONTHS,
    OUTPUT_DIR,
    TOP_N,
)


def first_trading_day_panel(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["month"] = daily["date"].dt.to_period("M")
    firsts = (
        daily.sort_values(["ticker", "date"])
        .groupby(["ticker", "month"], as_index=False)
        .first()[["ticker", "month", "date", "adj_close"]]
        .rename(columns={"date": "first_trade_date", "adj_close": "first_day_adj_close"})
    )
    firsts = firsts.sort_values(["ticker", "month"])
    firsts["monthly_open_to_open_return"] = firsts.groupby("ticker")["first_day_adj_close"].pct_change()
    return firsts


def zscore_by_month(df: pd.DataFrame, col: str) -> pd.Series:
    def z(s: pd.Series) -> pd.Series:
        std = s.std(ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    return df.groupby("month")[col].transform(z)


def acceleration_feature_specs() -> list[dict]:
    """Return the exact acceleration features used by Pure, B, and C strategies.

    The intended ablation is deliberately narrow: four simple average
    momentum-acceleration features, plus two contrast features. We do NOT
    generate every 3v3/2v2 variant for every lookback.
    """
    specs = [
        {
            "lookback": 3,
            "feature": "avg_accel_3m",
            "method": "avg_accel",
            "label": "avg_accel_3m",
            "description": "Average month-to-month change in the 3M momentum series.",
        },
        {
            "lookback": 4,
            "feature": "avg_accel_4m",
            "method": "avg_accel",
            "label": "avg_accel_4m",
            "description": "Average month-to-month change in the 4M momentum series.",
        },
        {
            "lookback": 5,
            "feature": "avg_accel_5m",
            "method": "avg_accel",
            "label": "avg_accel_5m",
            "description": "Average month-to-month change in the 5M momentum series.",
        },
        {
            "lookback": 6,
            "feature": "avg_accel_6m",
            "method": "avg_accel",
            "label": "avg_accel_6m",
            "description": "Average month-to-month change in the 6M momentum series.",
        },
        {
            "lookback": 6,
            "feature": "accel_3v3_6m",
            "method": "recent3_minus_previous3_avg_accel",
            "label": "accel_3v3_6m",
            "description": "Recent 3-month average momentum acceleration minus previous 3-month average momentum acceleration, based on the 6M momentum series.",
        },
        {
            "lookback": 4,
            "feature": "accel_2v2_4m",
            "method": "recent2_minus_previous2_avg_accel",
            "label": "accel_2v2_4m",
            "description": "Recent 2-month average momentum acceleration minus previous 2-month average momentum acceleration, based on the 4M momentum series.",
        },
    ]
    return specs


def build_signal_panel(firsts: pd.DataFrame) -> pd.DataFrame:
    """Build momentum and momentum-acceleration features.

    Momentum_N(t) is the average of the previous N monthly open-to-open returns.
    For a decision made in month t, every signal is shifted so it only uses
    information known before the first trading day of t.

    Method 1: avg_accel_N(t)
        Average month-to-month change of Momentum_N over the previous N
        momentum observations:

        avg_accel_N(t) = mean(diff([M_N(t-N), ..., M_N(t-1)]))
                       = (M_N(t-1) - M_N(t-N)) / (N - 1)

    Method 2: accel_3v3_N(t)
        First compute monthly momentum acceleration:

        monthly_mom_accel_N(t) = M_N(t) - M_N(t-1)

        Then compare recent acceleration with earlier acceleration:

        accel_3v3_N(t) = mean(monthly_mom_accel_N(t), t-1, t-2)
                         - mean(monthly_mom_accel_N(t-3), t-4, t-5)

        In words: average momentum acceleration over the last 3 available
        months minus average momentum acceleration over the previous 3 months.

    Method 3: accel_2v2_N(t)
        accel_2v2_N(t) = mean(monthly_mom_accel_N(t), t-1)
                         - mean(monthly_mom_accel_N(t-2), t-3)

        In words: average momentum acceleration over the last 2 available
        months minus average momentum acceleration over the previous 2 months.
    """
    panel = firsts[["ticker", "month", "first_trade_date", "first_day_adj_close", "monthly_open_to_open_return"]].copy()
    panel = panel.sort_values(["ticker", "month"])
    g = panel.groupby("ticker", group_keys=False)

    for m in MOMENTUM_LOOKBACK_MONTHS:
        panel[f"momentum_{m}m"] = g["monthly_open_to_open_return"].apply(
            lambda s: s.shift(1).rolling(m, min_periods=m).mean()
        )

    # Need to recreate the groupby after adding columns.
    g = panel.groupby("ticker", group_keys=False)

    # Four simple average acceleration features: avg_accel_3m/4m/5m/6m.
    for n in [3, 4, 5, 6]:
        mom_col = f"momentum_{n}m"
        avg_col = f"avg_accel_{n}m"
        panel[avg_col] = g[mom_col].apply(lambda s: (s - s.shift(n - 1)) / (n - 1))
        panel[f"momentum_{n}m_start_for_avg_accel"] = g[mom_col].shift(n - 1)
        panel[f"momentum_{n}m_end_for_avg_accel"] = panel[mom_col]

    # Contrast feature 1: accel_3v3_6m only.
    monthly_accel_6m = "monthly_mom_accel_6m"
    panel[monthly_accel_6m] = g["momentum_6m"].diff()
    g = panel.groupby("ticker", group_keys=False)
    panel["recent3_avg_mom_accel_6m"] = g[monthly_accel_6m].apply(lambda s: s.rolling(3, min_periods=3).mean())
    panel["previous3_avg_mom_accel_6m"] = g[monthly_accel_6m].apply(lambda s: s.shift(3).rolling(3, min_periods=3).mean())
    panel["accel_3v3_6m"] = panel["recent3_avg_mom_accel_6m"] - panel["previous3_avg_mom_accel_6m"]

    # Contrast feature 2: accel_2v2_4m only.
    monthly_accel_4m = "monthly_mom_accel_4m"
    panel[monthly_accel_4m] = g["momentum_4m"].diff()
    g = panel.groupby("ticker", group_keys=False)
    panel["recent2_avg_mom_accel_4m"] = g[monthly_accel_4m].apply(lambda s: s.rolling(2, min_periods=2).mean())
    panel["previous2_avg_mom_accel_4m"] = g[monthly_accel_4m].apply(lambda s: s.shift(2).rolling(2, min_periods=2).mean())
    panel["accel_2v2_4m"] = panel["recent2_avg_mom_accel_4m"] - panel["previous2_avg_mom_accel_4m"]

    return panel


def add_forward_returns(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["ticker", "month"]).copy()
    g = panel.groupby("ticker", group_keys=False)
    for h in [1, 2, 3]:
        future_price = g["first_day_adj_close"].shift(-h)
        panel[f"future_return_{h}m"] = future_price / panel["first_day_adj_close"] - 1.0
    panel["future_max_return_1_3m"] = panel[["future_return_1m", "future_return_2m", "future_return_3m"]].max(axis=1, skipna=False)
    return panel


def select_top_n(mdf: pd.DataFrame, score_col: str, top_n: int = TOP_N) -> pd.DataFrame:
    return mdf.sort_values([score_col, "ticker"], ascending=[False, True]).head(top_n).copy()


def portfolio_row(strategy: str, family: str, month, ranked: pd.DataFrame, score_col: str, extra: dict | None = None) -> dict:
    row = {
        "strategy": strategy,
        "family": family,
        "decision_month": str(month),
        "decision_date": ranked["first_trade_date"].min().date().isoformat(),
        "selected_tickers": ",".join(ranked["ticker"].tolist()),
        "score_col": score_col,
        "avg_score": ranked[score_col].mean(),
        "portfolio_future_return_1m": ranked["future_return_1m"].mean(),
        "portfolio_future_max_return_1_3m": ranked["future_max_return_1_3m"].mean(),
    }
    if extra:
        row.update(extra)
    for i, (_, r) in enumerate(ranked.iterrows(), start=1):
        row[f"top{i}_ticker"] = r["ticker"]
        row[f"top{i}_score"] = r[score_col]
        row[f"top{i}_future_return_1m"] = r["future_return_1m"]
        row[f"top{i}_future_max_return_1_3m"] = r["future_max_return_1_3m"]
    return row


def expanded_selection_rows(row: dict, ranked: pd.DataFrame, score_col: str) -> list[dict]:
    out = []
    feature_cols = [
        "momentum_3m", "momentum_4m", "momentum_5m", "momentum_6m",
        "avg_accel_3m", "avg_accel_4m", "avg_accel_5m", "avg_accel_6m",
        "accel_3v3_6m", "accel_2v2_4m",
    ]
    for i, (_, r) in enumerate(ranked.iterrows(), start=1):
        item = {
            "strategy": row["strategy"],
            "family": row["family"],
            "decision_month": row["decision_month"],
            "decision_date": row["decision_date"],
            "rank": i,
            "ticker": r["ticker"],
            "score_col": score_col,
            "score": r[score_col],
            "accel_method": row.get("accel_method", ""),
            "accel_feature": row.get("accel_feature", ""),
            "future_return_1m": r["future_return_1m"],
            "future_max_return_1_3m": r["future_max_return_1_3m"],
        }
        for c in feature_cols:
            item[c] = r.get(c, np.nan)
        out.append(item)
    return out


def run_backtest(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = panel[panel["month"] >= pd.Period(BACKTEST_START, freq="M")].copy()
    panel = panel.dropna(subset=["future_return_1m", "future_max_return_1_3m"])

    specs = acceleration_feature_specs()

    # Cross-sectional z-scores for combined Strategy C.
    panel["z_momentum_6m"] = zscore_by_month(panel, "momentum_6m")
    for spec in specs:
        feature = spec["feature"]
        z_col = f"z_{feature}"
        combo_col = f"combined_score_{feature}"
        panel[z_col] = zscore_by_month(panel, feature)
        panel[combo_col] = panel["z_momentum_6m"] + COMBINED_ACCEL_WEIGHT * panel[z_col]

    rows = []
    selection_rows = []

    for month, mdf0 in panel.groupby("month"):
        # Strategy A: original six-month average momentum.
        mdf = mdf0.dropna(subset=["momentum_6m"]).copy()
        if not mdf.empty:
            ranked = select_top_n(mdf, "momentum_6m")
            row = portfolio_row(
                strategy=f"A_momentum_6m_top{TOP_N}",
                family="A_original_6m_momentum",
                month=month,
                ranked=ranked,
                score_col="momentum_6m",
                extra={
                    "accel_lookback_months": np.nan,
                    "accel_method": "none",
                    "accel_feature": "none",
                    "filter_top_n": np.nan,
                    "combined_accel_weight": np.nan,
                },
            )
            rows.append(row)
            selection_rows.extend(expanded_selection_rows(row, ranked, "momentum_6m"))

        for spec in specs:
            n = spec["lookback"]
            accel_col = spec["feature"]
            method = spec["method"]
            label = spec["label"]
            common_extra = {
                "accel_lookback_months": n,
                "accel_method": method,
                "accel_feature": accel_col,
            }

            # Pure acceleration strategy: standalone hypothesis test.
            mdf = mdf0.dropna(subset=[accel_col]).copy()
            if not mdf.empty:
                ranked = select_top_n(mdf, accel_col)
                row = portfolio_row(
                    strategy=f"Pure_{label}_top{TOP_N}",
                    family="Pure_acceleration",
                    month=month,
                    ranked=ranked,
                    score_col=accel_col,
                    extra={**common_extra, "filter_top_n": np.nan, "combined_accel_weight": np.nan},
                )
                rows.append(row)
                selection_rows.extend(expanded_selection_rows(row, ranked, accel_col))

            # Strategy B: six-month momentum filter, then acceleration ranking.
            mdf = mdf0.dropna(subset=["momentum_6m", accel_col]).copy()
            if not mdf.empty:
                momentum_pool = select_top_n(mdf, "momentum_6m", top_n=FILTER_TOP_N)
                ranked = select_top_n(momentum_pool, accel_col)
                row = portfolio_row(
                    strategy=f"B_mom6m_top{FILTER_TOP_N}_then_{label}_top{TOP_N}",
                    family="B_6m_momentum_filter_then_acceleration",
                    month=month,
                    ranked=ranked,
                    score_col=accel_col,
                    extra={**common_extra, "filter_top_n": FILTER_TOP_N, "combined_accel_weight": np.nan},
                )
                rows.append(row)
                selection_rows.extend(expanded_selection_rows(row, ranked, accel_col))

            # Strategy C: z(6m momentum) + lambda * z(acceleration feature).
            combo_col = f"combined_score_{accel_col}"
            mdf = mdf0.dropna(subset=[combo_col]).copy()
            if not mdf.empty:
                ranked = select_top_n(mdf, combo_col)
                row = portfolio_row(
                    strategy=f"C_z_mom6m_plus_{COMBINED_ACCEL_WEIGHT:g}_z_{label}_top{TOP_N}",
                    family="C_combined_score",
                    month=month,
                    ranked=ranked,
                    score_col=combo_col,
                    extra={**common_extra, "filter_top_n": np.nan, "combined_accel_weight": COMBINED_ACCEL_WEIGHT},
                )
                rows.append(row)
                selection_rows.extend(expanded_selection_rows(row, ranked, combo_col))

    monthly = pd.DataFrame(rows).sort_values(["family", "strategy", "decision_month"])
    selections = pd.DataFrame(selection_rows).sort_values(["family", "strategy", "decision_month", "rank"])
    summary = summarize(monthly)
    comparison = build_comparison_table(summary)
    return monthly, selections, summary, comparison


def summarize(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, g in monthly.groupby("strategy"):
        r1 = g["portfolio_future_return_1m"].dropna()
        rmax = g["portfolio_future_max_return_1_3m"].dropna()
        if r1.empty or rmax.empty:
            continue
        equity = (1 + r1).cumprod()
        running_max = equity.cummax()
        drawdown = equity / running_max - 1
        first = g.iloc[0]
        rows.append({
            "strategy": strategy,
            "family": first["family"],
            "accel_lookback_months": first.get("accel_lookback_months", np.nan),
            "accel_method": first.get("accel_method", ""),
            "accel_feature": first.get("accel_feature", ""),
            "months": int(len(g)),
            "avg_future_return_1m": r1.mean(),
            "median_future_return_1m": r1.median(),
            "win_rate_1m": (r1 > 0).mean(),
            "cumulative_return_1m_rebalanced": equity.iloc[-1] - 1,
            "max_drawdown_1m_rebalanced": drawdown.min(),
            "avg_future_max_return_1_3m": rmax.mean(),
            "median_future_max_return_1_3m": rmax.median(),
            "hit_rate_positive_max_1_3m": (rmax > 0).mean(),
            "best_month_1m": r1.max(),
            "worst_month_1m": r1.min(),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["family", "accel_method", "accel_lookback_months", "strategy"], na_position="first")


def build_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    cols = [
        "strategy",
        "months",
        "avg_future_return_1m",
        "win_rate_1m",
        "avg_future_max_return_1_3m",
        "hit_rate_positive_max_1_3m",
        "cumulative_return_1m_rebalanced",
        "max_drawdown_1m_rebalanced",
    ]
    comp = summary[[c for c in cols if c in summary.columns]].copy()
    return comp.sort_values("avg_future_return_1m", ascending=False)


def build_panel_audit_outputs(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create compact audit tables for checking the full monthly panel.

    The complete panel is saved separately as `momentum_acceleration_panel.csv`.
    These summaries make it easier to verify row counts, month coverage, ticker
    coverage, feature availability, and forward-return availability.
    """
    p = panel.copy()
    if p.empty:
        summary = pd.DataFrame([{
            "panel_rows": 0,
            "unique_tickers": 0,
            "unique_months": 0,
            "first_panel_month": "",
            "last_panel_month": "",
            "first_trade_date_min": "",
            "first_trade_date_max": "",
            "backtest_evaluable_rows": 0,
            "backtest_evaluable_months": 0,
            "backtest_evaluable_tickers": 0,
        }])
        return summary, pd.DataFrame()

    p["month_str"] = p["month"].astype(str)
    evaluable = p.dropna(subset=["future_return_1m", "future_max_return_1_3m"]).copy()
    signal_cols = [
        "momentum_3m", "momentum_4m", "momentum_5m", "momentum_6m",
        "avg_accel_3m", "avg_accel_4m", "avg_accel_5m", "avg_accel_6m",
        "accel_3v3_6m", "accel_2v2_4m",
    ]
    signal_cols = [c for c in signal_cols if c in p.columns]

    summary = pd.DataFrame([{
        "panel_rows": int(len(p)),
        "unique_tickers": int(p["ticker"].nunique()),
        "unique_months": int(p["month_str"].nunique()),
        "first_panel_month": p["month_str"].min(),
        "last_panel_month": p["month_str"].max(),
        "first_trade_date_min": pd.to_datetime(p["first_trade_date"]).min().date().isoformat() if "first_trade_date" in p.columns else "",
        "first_trade_date_max": pd.to_datetime(p["first_trade_date"]).max().date().isoformat() if "first_trade_date" in p.columns else "",
        "backtest_evaluable_rows": int(len(evaluable)),
        "backtest_evaluable_months": int(evaluable["month_str"].nunique()) if not evaluable.empty else 0,
        "backtest_evaluable_tickers": int(evaluable["ticker"].nunique()) if not evaluable.empty else 0,
    }])

    rows = []
    for month, g in p.groupby("month_str"):
        row = {
            "month": month,
            "rows": int(len(g)),
            "unique_tickers": int(g["ticker"].nunique()),
            "first_trade_date_min": pd.to_datetime(g["first_trade_date"]).min().date().isoformat() if "first_trade_date" in g.columns else "",
            "first_trade_date_max": pd.to_datetime(g["first_trade_date"]).max().date().isoformat() if "first_trade_date" in g.columns else "",
            "has_future_return_1m_rows": int(g["future_return_1m"].notna().sum()) if "future_return_1m" in g.columns else 0,
            "has_future_max_return_1_3m_rows": int(g["future_max_return_1_3m"].notna().sum()) if "future_max_return_1_3m" in g.columns else 0,
        }
        for c in signal_cols:
            row[f"has_{c}_rows"] = int(g[c].notna().sum())
        rows.append(row)
    by_month = pd.DataFrame(rows).sort_values("month")
    return summary, by_month


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(DAILY_PRICES_FILE)
    firsts = first_trading_day_panel(daily)
    signal_panel = build_signal_panel(firsts)
    panel = add_forward_returns(signal_panel)
    monthly, selections, summary, comparison = run_backtest(panel)

    panel_summary, panel_by_month = build_panel_audit_outputs(panel)

    firsts.to_csv(OUTPUT_DIR / "first_trading_day_prices.csv", index=False)
    panel.to_csv(OUTPUT_DIR / "momentum_acceleration_panel.csv", index=False)
    panel.to_csv(OUTPUT_DIR / "full_panel.csv", index=False)
    panel.head(500).to_csv(OUTPUT_DIR / "panel_head_500.csv", index=False)
    panel.tail(500).to_csv(OUTPUT_DIR / "panel_tail_500.csv", index=False)
    panel_summary.to_csv(OUTPUT_DIR / "panel_summary.csv", index=False)
    panel_by_month.to_csv(OUTPUT_DIR / "panel_by_month_summary.csv", index=False)
    monthly.to_csv(OUTPUT_DIR / "monthly_strategy_returns.csv", index=False)
    selections.to_csv(OUTPUT_DIR / "monthly_selected_tickers.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "strategy_summary.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "strategy_comparison.csv", index=False)
    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
