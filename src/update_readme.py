import pandas as pd

from .backtest_acceleration import acceleration_feature_specs
from .config import (
    COMBINED_ACCEL_WEIGHT,
    FILTER_TOP_N,
    OUTPUT_DIR,
    PROJECT_ROOT,
    REPORT_END_MONTH,
    REPORT_START_MONTH,
    TOP_N,
)

README_FILE = PROJECT_ROOT / "README.md"


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


def compact_monthly_table(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    cols = [
        "decision_month", "strategy", "family", "accel_method", "accel_feature", "selected_tickers",
        "portfolio_future_return_1m", "portfolio_future_max_return_1_3m",
        "top1_ticker", "top1_future_return_1m", "top1_future_max_return_1_3m",
        "top2_ticker", "top2_future_return_1m", "top2_future_max_return_1_3m",
        "top3_ticker", "top3_future_return_1m", "top3_future_max_return_1_3m",
    ]
    return monthly[[c for c in cols if c in monthly.columns]].copy()


def main() -> None:
    summary_file = OUTPUT_DIR / "strategy_summary.csv"
    comparison_file = OUTPUT_DIR / "strategy_comparison.csv"
    monthly_file = OUTPUT_DIR / "monthly_strategy_returns.csv"
    selection_file = OUTPUT_DIR / "monthly_selected_tickers.csv"

    summary = pd.read_csv(summary_file) if summary_file.exists() else pd.DataFrame()
    comparison = pd.read_csv(comparison_file) if comparison_file.exists() else pd.DataFrame()
    monthly = pd.read_csv(monthly_file) if monthly_file.exists() else pd.DataFrame()
    selections = pd.read_csv(selection_file) if selection_file.exists() else pd.DataFrame()

    show_summary = format_percent_cols(summary) if not summary.empty else summary
    show_comparison = format_percent_cols(comparison) if not comparison.empty else comparison

    if not monthly.empty:
        period = monthly[
            (monthly["decision_month"] >= REPORT_START_MONTH)
            & (monthly["decision_month"] <= REPORT_END_MONTH)
        ].copy()
        period = compact_monthly_table(period)
        period = format_percent_cols(period)
    else:
        period = pd.DataFrame()

    if not selections.empty:
        period_sel = selections[
            (selections["decision_month"] >= REPORT_START_MONTH)
            & (selections["decision_month"] <= REPORT_END_MONTH)
        ].copy()
        # Keep this compact in README; full expanded selections remain in CSV.
        period_sel = period_sel[[c for c in [
            "decision_month", "strategy", "rank", "ticker", "score_col", "score",
            "future_return_1m", "future_max_return_1_3m",
        ] if c in period_sel.columns]].head(300)
        period_sel = format_percent_cols(period_sel)
    else:
        period_sel = pd.DataFrame()

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
| `outputs/strategy_summary.csv` | Summary metrics by strategy |
| `outputs/strategy_comparison.csv` | Compact comparison table sorted by average future 1M return |

## Strategy comparison

{to_md(show_comparison)}

## Strategy summary

{to_md(show_summary)}

## Monthly strategy returns: {REPORT_START_MONTH} to {REPORT_END_MONTH}

This section replaces the old single-latest-month view. It shows the requested backtest window, not only 2026-03. Full monthly history remains in `outputs/monthly_strategy_returns.csv`.

{to_md(period)}

## Expanded selected tickers sample: {REPORT_START_MONTH} to {REPORT_END_MONTH}

This is capped at the first 300 rows to keep the README readable. Full selections are in `outputs/monthly_selected_tickers.csv`.

{to_md(period_sel)}

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
