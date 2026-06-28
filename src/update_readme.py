import pandas as pd

from .backtest_acceleration import acceleration_feature_specs, select_top_n
from .config import (
    BACKTEST_START,
    COMBINED_ACCEL_WEIGHT,
    FILTER_TOP_N,
    OUTPUT_DIR,
    PROJECT_ROOT,
    REPORT_END_MONTH,
    REPORT_START_MONTH,
    TOP_N,
)

README_FILE = PROJECT_ROOT / "README.md"

COMPARISON_DISPLAY_COLS = [
    "strategy",
    "months",
    "avg_future_return_1m",
    "win_rate_1m",
    "avg_future_max_return_1_3m",
    "hit_rate_positive_max_1_3m",
    "cumulative_return_1m_rebalanced",
    "max_drawdown_1m_rebalanced",
]

MONTHLY_DISPLAY_COLS = [
    "decision_month",
    "decision_date",
    "selected_tickers",
    "score_col",
    "avg_score",
    "portfolio_future_return_1m",
    "portfolio_future_max_return_1_3m",
    "top1_ticker",
    "top1_future_return_1m",
    "top1_future_max_return_1_3m",
    "top2_ticker",
    "top2_future_return_1m",
    "top2_future_max_return_1_3m",
    "top3_ticker",
    "top3_future_return_1m",
    "top3_future_max_return_1_3m",
]

FAMILY_ORDER = [
    "A_original_6m_momentum",
    "Pure_acceleration",
    "B_6m_momentum_filter_then_acceleration",
    "C_combined_score",
]

FAMILY_TITLES = {
    "A_original_6m_momentum": "Strategy A: original six-month average momentum",
    "Pure_acceleration": "Pure acceleration strategies",
    "B_6m_momentum_filter_then_acceleration": "Strategy B: six-month momentum plus acceleration filter",
    "C_combined_score": "Strategy C: combined momentum and acceleration score",
}


def fmt_pct(x) -> str:
    if pd.isna(x):
        return ""
    return f"{x * 100:.2f}%"


def fmt_num(x) -> str:
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available. Run `python run_all.py` first._"
    return df.to_markdown(index=False)


def format_percent_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pct_keywords = ["return", "rate", "drawdown", "best_month", "worst_month"]
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            continue
        if any(k in c for k in pct_keywords):
            out[c] = out[c].map(fmt_pct)
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            continue
        if "score" in c or "momentum" in c or "accel" in c:
            if c not in {"accel_lookback_months"}:
                out[c] = out[c].map(fmt_num)
    return out


def compact_comparison_table(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return comparison
    return comparison[[c for c in COMPARISON_DISPLAY_COLS if c in comparison.columns]].copy()


def compact_monthly_table(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    return monthly[[c for c in MONTHLY_DISPLAY_COLS if c in monthly.columns]].copy()


def strategy_sort_key(strategy: str) -> tuple:
    if strategy.startswith("A_"):
        return (0, strategy)
    if strategy.startswith("Pure_"):
        return (1, strategy)
    if strategy.startswith("B_"):
        return (2, strategy)
    if strategy.startswith("C_"):
        return (3, strategy)
    return (9, strategy)


def monthly_returns_sections(period: pd.DataFrame) -> str:
    if period.empty:
        return "_No monthly strategy returns available for this window. Run `python run_all.py` first._"

    parts: list[str] = []
    for family in FAMILY_ORDER:
        fdf = period[period["family"] == family].copy() if "family" in period.columns else pd.DataFrame()
        if fdf.empty:
            continue
        parts.append(f"### {FAMILY_TITLES[family]}")
        strategies = sorted(fdf["strategy"].dropna().unique().tolist(), key=strategy_sort_key)
        for strategy in strategies:
            sdf = fdf[fdf["strategy"] == strategy].copy()
            parts.append(f"#### `{strategy}`")
            show = compact_monthly_table(sdf)
            show = format_percent_cols(show)
            parts.append(to_md(show))
    if not parts:
        return "_No monthly strategy returns available for this window. Run `python run_all.py` first._"
    return "\n\n".join(parts)


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def _date_only(x) -> str:
    if pd.isna(x):
        return ""
    try:
        return pd.to_datetime(x).date().isoformat()
    except Exception:
        return str(x)


def latest_top3_sections(panel: pd.DataFrame, fallback_selections: pd.DataFrame | None = None) -> str:
    """Show current/latest signal-month Top-3 selections, not just evaluated months.

    `monthly_selected_tickers.csv` only contains months with known forward 1M/2M/3M
    returns, so it naturally lags by about three months. For the README's latest
    selection block we instead use `momentum_acceleration_panel.csv`, which has
    the latest available first-trading-day signal month even when future returns
    are not known yet.
    """
    if panel.empty:
        # Backward-compatible fallback for old output folders.
        selections = fallback_selections if fallback_selections is not None else pd.DataFrame()
        if selections.empty or "decision_month" not in selections.columns or "strategy" not in selections.columns:
            return "_No latest selections available. Run `python run_all.py` first._"
        latest_month = selections["decision_month"].max()
        latest = selections[selections["decision_month"] == latest_month].copy()
        if latest.empty:
            return "_No latest selections available. Run `python run_all.py` first._"
        rows = []
        family_map = latest.groupby("strategy")["family"].first().to_dict() if "family" in latest.columns else {}
        date_map = latest.groupby("strategy")["decision_date"].first().to_dict() if "decision_date" in latest.columns else {}
        for strategy, g in latest.sort_values(["strategy", "rank"]).groupby("strategy"):
            g = g.sort_values("rank")
            row = {"strategy": strategy, "decision_date": date_map.get(strategy, ""), "top1": "", "top2": "", "top3": "", "family": family_map.get(strategy, "")}
            for _, r in g.iterrows():
                rank = int(r.get("rank", 0)) if not pd.isna(r.get("rank", None)) else 0
                if rank in {1, 2, 3}:
                    row[f"top{rank}"] = r.get("ticker", "")
            rows.append(row)
        latest_wide = pd.DataFrame(rows)
        latest_label = str(latest_month)
    else:
        p = panel.copy()
        if "month" not in p.columns or "ticker" not in p.columns:
            return "_No latest selections available. Run `python run_all.py` first._"
        p["month_period"] = pd.PeriodIndex(p["month"].astype(str), freq="M")
        p = p[p["month_period"] >= pd.Period(BACKTEST_START, freq="M")].copy()
        specs = acceleration_feature_specs()
        needed_any = ["momentum_6m"] + [s["feature"] for s in specs]
        existing_needed = [c for c in needed_any if c in p.columns]
        if not existing_needed:
            return "_No latest selections available. Run `python run_all.py` first._"
        valid_months = p.groupby("month_period")[existing_needed].apply(lambda x: x.notna().any().any())
        valid_months = valid_months[valid_months]
        if valid_months.empty:
            return "_No latest selections available. Run `python run_all.py` first._"
        latest_period = valid_months.index.max()
        latest = p[p["month_period"] == latest_period].copy()
        latest_label = str(latest_period)
        rows: list[dict] = []

        def add_row(strategy: str, family: str, ranked: pd.DataFrame):
            if ranked.empty:
                return
            row = {
                "strategy": strategy,
                "decision_date": _date_only(ranked["first_trade_date"].min()) if "first_trade_date" in ranked.columns else "",
                "top1": "",
                "top2": "",
                "top3": "",
                "family": family,
            }
            for i, (_, r) in enumerate(ranked.head(TOP_N).iterrows(), start=1):
                row[f"top{i}"] = r.get("ticker", "")
            rows.append(row)

        # Strategy A.
        if "momentum_6m" in latest.columns:
            mdf = latest.dropna(subset=["momentum_6m"]).copy()
            if not mdf.empty:
                add_row(f"A_momentum_6m_top{TOP_N}", "A_original_6m_momentum", select_top_n(mdf, "momentum_6m"))

        for spec in specs:
            feature = spec["feature"]
            label = spec["label"]
            if feature not in latest.columns:
                continue

            # Pure acceleration.
            mdf = latest.dropna(subset=[feature]).copy()
            if not mdf.empty:
                add_row(f"Pure_{label}_top{TOP_N}", "Pure_acceleration", select_top_n(mdf, feature))

            # Strategy B.
            if "momentum_6m" in latest.columns:
                mdf = latest.dropna(subset=["momentum_6m", feature]).copy()
                if not mdf.empty:
                    momentum_pool = select_top_n(mdf, "momentum_6m", top_n=FILTER_TOP_N)
                    add_row(
                        f"B_mom6m_top{FILTER_TOP_N}_then_{label}_top{TOP_N}",
                        "B_6m_momentum_filter_then_acceleration",
                        select_top_n(momentum_pool, feature),
                    )

            # Strategy C.
            if "momentum_6m" in latest.columns:
                mdf = latest.dropna(subset=["momentum_6m", feature]).copy()
                if not mdf.empty:
                    combo_col = f"combined_score_{feature}"
                    mdf["z_momentum_6m"] = _zscore(mdf["momentum_6m"])
                    mdf[f"z_{feature}"] = _zscore(mdf[feature])
                    mdf[combo_col] = mdf["z_momentum_6m"] + COMBINED_ACCEL_WEIGHT * mdf[f"z_{feature}"]
                    add_row(
                        f"C_z_mom6m_plus_{COMBINED_ACCEL_WEIGHT:g}_z_{label}_top{TOP_N}",
                        "C_combined_score",
                        select_top_n(mdf, combo_col),
                    )

        latest_wide = pd.DataFrame(rows)

    if latest_wide.empty:
        return "_No latest selections available. Run `python run_all.py` first._"

    parts: list[str] = [f"Latest available decision month: **{latest_label}**."]
    for family in FAMILY_ORDER:
        fdf = latest_wide[latest_wide["family"] == family].copy() if "family" in latest_wide.columns else pd.DataFrame()
        if fdf.empty:
            continue
        fdf = fdf.sort_values("strategy", key=lambda col: col.map(strategy_sort_key))
        show_cols = ["strategy", "decision_date", "top1", "top2", "top3"]
        parts.append(f"### {FAMILY_TITLES[family]}")
        parts.append(to_md(fdf[[c for c in show_cols if c in fdf.columns]]))

    if len(parts) == 1:
        latest_wide = latest_wide.sort_values("strategy", key=lambda col: col.map(strategy_sort_key))
        show_cols = ["strategy", "decision_date", "top1", "top2", "top3"]
        parts.append(to_md(latest_wide[[c for c in show_cols if c in latest_wide.columns]]))
    return "\n\n".join(parts)


def main() -> None:
    comparison_file = OUTPUT_DIR / "strategy_comparison.csv"
    monthly_file = OUTPUT_DIR / "monthly_strategy_returns.csv"
    selection_file = OUTPUT_DIR / "monthly_selected_tickers.csv"
    panel_file = OUTPUT_DIR / "momentum_acceleration_panel.csv"

    comparison = pd.read_csv(comparison_file) if comparison_file.exists() else pd.DataFrame()
    monthly = pd.read_csv(monthly_file) if monthly_file.exists() else pd.DataFrame()
    selections = pd.read_csv(selection_file) if selection_file.exists() else pd.DataFrame()
    latest_panel = pd.read_csv(panel_file) if panel_file.exists() else pd.DataFrame()

    show_comparison = compact_comparison_table(comparison)
    show_comparison = format_percent_cols(show_comparison) if not show_comparison.empty else show_comparison

    if not monthly.empty:
        period = monthly[
            (monthly["decision_month"] >= REPORT_START_MONTH)
            & (monthly["decision_month"] <= REPORT_END_MONTH)
        ].copy()
    else:
        period = pd.DataFrame()

    specs = acceleration_feature_specs()
    feature_rows = "\n".join(
        f"| `{s['feature']}` | `{s['method']}` | {s['description']} | Pure, Strategy B, Strategy C |"
        for s in specs
    )
    pure_rows = "\n".join(
        f"| `Pure_{s['label']}_top{TOP_N}` | `Pure_acceleration` | Rank all eligible stocks directly by `{s['feature']}`; select Top-{TOP_N}. | Tests whether `{s['feature']}` alone has predictive power. |"
        for s in specs
    )
    b_rows = "\n".join(
        f"| `B_mom6m_top{FILTER_TOP_N}_then_{s['label']}_top{TOP_N}` | `B_6m_momentum_filter_then_acceleration` | Step 1: select Top-{FILTER_TOP_N} by `momentum_6m`. Step 2: inside that pool, rank by `{s['feature']}`; select Top-{TOP_N}. | Tests whether `{s['feature']}` improves a strong 6M momentum shortlist. |"
        for s in specs
    )
    c_rows = "\n".join(
        f"| `C_z_mom6m_plus_{COMBINED_ACCEL_WEIGHT:g}_z_{s['label']}_top{TOP_N}` | `C_combined_score` | Score = `z(momentum_6m) + {COMBINED_ACCEL_WEIGHT:g} * z({s['feature']})` computed cross-sectionally within each decision month; select Top-{TOP_N}. | Tests whether 6M momentum and `{s['feature']}` should be blended. |"
        for s in specs
    )

    text = f"""# Momentum Acceleration Ablation Backtest

## Latest month Top-3 selections by strategy

This section uses the latest available signal month in `outputs/momentum_acceleration_panel.csv`, so it can show the current month even when future 1M/2M/3M returns are not known yet.

{latest_top3_sections(latest_panel, selections)}

## Monthly auto-update

This project is configured to update automatically on the **first NYSE trading day of each month** after the US market close. The GitHub Actions workflow runs on weekdays during calendar days 1-7 and uses `src/should_run_monthly_update.py` to allow the full update only when that day is the first NYSE trading day of the month.

On an eligible monthly update, the workflow runs `python run_all.py`, rebuilds the universe and price panel, recomputes all strategy results, and refreshes this README. Therefore the **Latest month Top-3 selections by strategy** section above is automatically updated to the newest available decision month and becomes the current month's Top-3 buy suggestion for every strategy.

Manual updates are also supported through GitHub Actions `workflow_dispatch`, or locally with:

```bash
python run_all.py
```

This project tests whether **momentum acceleration** adds useful ranking power beyond raw momentum. It is intentionally independent from XGBoost: no model, no labels, no probability ranking.

## Research question

Raw momentum answers:

> Which stocks have gone up the most on average?

Momentum acceleration answers:

> Which stocks have a rising average momentum profile, meaning their momentum itself is improving?

The goal is to compare pure acceleration, original momentum, momentum-plus-acceleration filtering, and combined momentum/acceleration scoring on the same monthly universe.

## Core correction: acceleration is momentum difference, not raw-return difference

For decision month `t`, all signals are known before the first trading day of month `t`.

```text
monthly_return(t) = first_trading_day_price(t) / first_trading_day_price(t-1) - 1

Momentum_N(t) = average of the previous N monthly open-to-open returns
```

The strategy matrix uses exactly **six acceleration features**: `avg_accel_3m`, `avg_accel_4m`, `avg_accel_5m`, `avg_accel_6m`, `accel_3v3_6m`, and `accel_2v2_4m`.

### Method 1: average momentum acceleration

```text
avg_accel_N(t) = mean(diff([Momentum_N(t-N), ..., Momentum_N(t)]))
               = (Momentum_N(t) - Momentum_N(t-N+1)) / (N - 1)
```

This measures whether the **N-month average momentum itself** is rising or falling over its own lookback window.

### Method 2: recent-three acceleration minus previous-three acceleration, used only as `accel_3v3_6m`

First compute monthly momentum acceleration:

```text
monthly_mom_accel_N(t) = Momentum_N(t) - Momentum_N(t-1)
```

Then compare the recent three observations with the previous three observations. In this project this contrast is applied only to the 6M momentum series:

```text
accel_3v3_6m(t)
= mean(monthly_mom_accel_6m(t), monthly_mom_accel_6m(t-1), monthly_mom_accel_6m(t-2))
  - mean(monthly_mom_accel_6m(t-3), monthly_mom_accel_6m(t-4), monthly_mom_accel_6m(t-5))
```

In words:

> `accel_3v3_6m` = 6M momentum series 的后三个月平均动量加速度 − 前三个月平均动量加速度。

This is a stricter acceleration-strengthening signal. It does not merely ask whether 6M momentum is rising; it asks whether the **acceleration of 6M momentum is itself stronger recently than before**.

### Method 3: recent-two acceleration minus previous-two acceleration, used only as `accel_2v2_4m`

This is the shorter, more reactive contrast feature. In this project it is applied only to the 4M momentum series:

```text
accel_2v2_4m(t)
= mean(monthly_mom_accel_4m(t), monthly_mom_accel_4m(t-1))
  - mean(monthly_mom_accel_4m(t-2), monthly_mom_accel_4m(t-3))
```

In words:

> `accel_2v2_4m` = 4M momentum series 的后两个月平均动量加速度 − 前两个月平均动量加速度。

Compared with `accel_3v3_6m`, this feature reacts faster to a recent strengthening or weakening of momentum acceleration, but it may also be noisier.

## Universe and timing

- Universe: ETF/index panel universe based on the same source idea as `James1424/XGB_tail_boom_4000`: QQQ, SPY, growth ETFs, semiconductor/AI ETFs, software/cloud ETFs, biotech, clean energy, innovation/high-beta ETFs, plus a manual high-interest ticker list.
- Decision time: the **first trading day of each month**.
- Backtest period: from **2016-01** to the latest month available in downloaded prices.
- README detailed monthly result window: **{REPORT_START_MONTH} to {REPORT_END_MONTH}**.
- Portfolio construction: equal-weight Top-{TOP_N} stocks for every strategy/month.
- Main forward evaluation targets:
  - `future_return_1m`: return from this month’s first trading day to next month’s first trading day.
  - `future_max_return_1_3m`: max of forward 1M, 2M, and 3M open-to-open returns.

## Feature definitions

| Feature | Definition | Used by |
|---|---|---|
| `momentum_3m` | Average of previous 3 monthly open-to-open returns | acceleration calculation and diagnostics |
| `momentum_4m` | Average of previous 4 monthly open-to-open returns | acceleration calculation and diagnostics |
| `momentum_5m` | Average of previous 5 monthly open-to-open returns | acceleration calculation and diagnostics |
| `momentum_6m` | Average of previous 6 monthly open-to-open returns | Strategy A, B filter, Strategy C base component |
{feature_rows}
| `z_momentum_6m` | Cross-sectional z-score of `momentum_6m` within a decision month | Strategy C |
| `z_<acceleration_feature>` | Cross-sectional z-score of each acceleration feature within a decision month | Strategy C |

## Strategy set

### 1. Pure acceleration strategies

These strategies ignore raw 6M momentum and rank directly by an acceleration feature.

| Strategy name | Family | Exact rule | Purpose |
|---|---|---|---|
{pure_rows}

### 2. Strategy A: original six-month average momentum baseline

| Strategy name | Family | Exact rule | Purpose |
|---|---|---|---|
| `A_momentum_6m_top{TOP_N}` | `A_original_6m_momentum` | Rank all eligible stocks by `momentum_6m`; select Top-{TOP_N}. | Baseline: the original raw 6M average momentum strategy. |

### 3. Strategy B: six-month momentum plus acceleration filter

This strategy asks whether acceleration is useful **after** requiring the stock to already be a strong 6M momentum name.

| Strategy name | Family | Exact rule | Purpose |
|---|---|---|---|
{b_rows}

### 4. Strategy C: combined score

This strategy asks whether momentum and acceleration should be blended into one cross-sectional score.

| Strategy name | Family | Exact rule | Purpose |
|---|---|---|---|
{c_rows}

## Ablation design

| Comparison | What it tests |
|---|---|
| Pure acceleration vs Strategy A | Whether acceleration alone can beat raw 6M momentum. |
| Strategy B vs Strategy A | Whether acceleration improves selection after a 6M momentum shortlist. |
| Strategy C vs Strategy A | Whether blending momentum and acceleration improves ranking. |
| `avg_accel_3m/4m/5m/6m` | Which simple average momentum-acceleration horizon is most useful. |
| `accel_3v3_6m` vs `accel_2v2_4m` | Whether a longer 6M 3-vs-3 contrast or shorter 4M 2-vs-2 contrast is more useful. |
| Simple average acceleration vs contrast acceleration | Whether direct momentum acceleration or acceleration-regime strengthening is more predictive. |
| `future_return_1m` vs `future_max_return_1_3m` | Whether acceleration predicts immediate next-month return or a larger move inside the next three months. |

## Summary metrics

`outputs/strategy_summary.csv` and `outputs/strategy_comparison.csv` include:

| Metric | Meaning |
|---|---|
| `avg_future_return_1m` | Average equal-weight Top-{TOP_N} future 1M return. |
| `median_future_return_1m` | Median equal-weight Top-{TOP_N} future 1M return. |
| `win_rate_1m` | Fraction of months where the equal-weight Top-{TOP_N} 1M return is positive. |
| `cumulative_return_1m_rebalanced` | Cumulative monthly-rebalanced return using `future_return_1m`. |
| `max_drawdown_1m_rebalanced` | Max drawdown of the monthly-rebalanced 1M equity curve. |
| `avg_future_max_return_1_3m` | Average equal-weight Top-{TOP_N} max return across forward 1M/2M/3M horizons. |
| `median_future_max_return_1_3m` | Median of the forward 1M/2M/3M max-return metric. |
| `hit_rate_positive_max_1_3m` | Fraction of months where the forward max 1–3M return is positive. |
| `best_month_1m` | Best monthly equal-weight Top-{TOP_N} 1M return. |
| `worst_month_1m` | Worst monthly equal-weight Top-{TOP_N} 1M return. |

## Run

```bash
pip install -r requirements.txt
python run_all.py
```

Fast smoke test:

```bash
python -m src.get_holdings_universe
python -m src.download_data --max-tickers 40
python -m src.backtest_acceleration
python -m src.update_readme
```

## Outputs

| File | Meaning |
|---|---|
| `data/seed_universe.csv` | Selected universe and source metadata |
| `data/daily_prices.csv.gz` | yfinance adjusted daily prices |
| `outputs/first_trading_day_prices.csv` | First-trading-day price panel |
| `outputs/momentum_acceleration_panel.csv` | Momentum, acceleration features, and forward-return panel |
| `outputs/monthly_strategy_returns.csv` | Monthly Top-{TOP_N} portfolio returns for every strategy |
| `outputs/monthly_selected_tickers.csv` | Selected tickers by strategy/month/rank |
| `outputs/strategy_summary.csv` | Full summary metrics by strategy |
| `outputs/strategy_comparison.csv` | Compact comparison table sorted by average future 1M return |

## Strategy comparison

{to_md(show_comparison)}

## Monthly strategy returns: {REPORT_START_MONTH} to {REPORT_END_MONTH}

This section shows the requested backtest window, grouped by strategy family. For Pure acceleration, Strategy B, and Strategy C, each individual strategy has its own table. Full monthly history remains in `outputs/monthly_strategy_returns.csv`.

{monthly_returns_sections(period)}

## Interpretation

This project is a feature-level hypothesis test. The key question is:

> Does the acceleration of average momentum help distinguish stocks whose trend is still strengthening from stocks whose raw six-month momentum is already high but fading?

The most important comparisons are:

1. `A_momentum_6m_top{TOP_N}` versus all Strategy B variants.
2. `A_momentum_6m_top{TOP_N}` versus all Strategy C variants.
3. Pure acceleration variants versus Strategy A.
4. The six exact acceleration features: `avg_accel_3m`, `avg_accel_4m`, `avg_accel_5m`, `avg_accel_6m`, `accel_3v3_6m`, and `accel_2v2_4m`.
5. Rankings by `avg_future_return_1m` and `avg_future_max_return_1_3m`.
"""
    README_FILE.write_text(text, encoding="utf-8")
    print(f"Updated {README_FILE}")


if __name__ == "__main__":
    main()
