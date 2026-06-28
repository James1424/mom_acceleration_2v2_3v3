# Momentum Acceleration Ablation Backtest

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
- README detailed monthly result window: **2025-01 to 2026-03**.
- Portfolio construction: equal-weight Top-3 stocks for every strategy/month.
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
| `avg_accel_3m` | `avg_accel` | Average month-to-month change in the 3M momentum series. | Pure, Strategy B, Strategy C |
| `avg_accel_4m` | `avg_accel` | Average month-to-month change in the 4M momentum series. | Pure, Strategy B, Strategy C |
| `avg_accel_5m` | `avg_accel` | Average month-to-month change in the 5M momentum series. | Pure, Strategy B, Strategy C |
| `avg_accel_6m` | `avg_accel` | Average month-to-month change in the 6M momentum series. | Pure, Strategy B, Strategy C |
| `accel_3v3_6m` | `recent3_minus_previous3_avg_accel` | Recent 3-month average momentum acceleration minus previous 3-month average momentum acceleration, based on the 6M momentum series. | Pure, Strategy B, Strategy C |
| `accel_2v2_4m` | `recent2_minus_previous2_avg_accel` | Recent 2-month average momentum acceleration minus previous 2-month average momentum acceleration, based on the 4M momentum series. | Pure, Strategy B, Strategy C |
| `z_momentum_6m` | Cross-sectional z-score of `momentum_6m` within a decision month | Strategy C |
| `z_<acceleration_feature>` | Cross-sectional z-score of each acceleration feature within a decision month | Strategy C |

## Strategy set

### 1. Pure acceleration strategies

These strategies ignore raw 6M momentum and rank directly by an acceleration feature.

| Strategy name | Family | Exact rule | Purpose |
|---|---|---|---|
| `Pure_avg_accel_3m_top3` | `Pure_acceleration` | Rank all eligible stocks directly by `avg_accel_3m`; select Top-3. | Tests whether `avg_accel_3m` alone has predictive power. |
| `Pure_avg_accel_4m_top3` | `Pure_acceleration` | Rank all eligible stocks directly by `avg_accel_4m`; select Top-3. | Tests whether `avg_accel_4m` alone has predictive power. |
| `Pure_avg_accel_5m_top3` | `Pure_acceleration` | Rank all eligible stocks directly by `avg_accel_5m`; select Top-3. | Tests whether `avg_accel_5m` alone has predictive power. |
| `Pure_avg_accel_6m_top3` | `Pure_acceleration` | Rank all eligible stocks directly by `avg_accel_6m`; select Top-3. | Tests whether `avg_accel_6m` alone has predictive power. |
| `Pure_accel_3v3_6m_top3` | `Pure_acceleration` | Rank all eligible stocks directly by `accel_3v3_6m`; select Top-3. | Tests whether `accel_3v3_6m` alone has predictive power. |
| `Pure_accel_2v2_4m_top3` | `Pure_acceleration` | Rank all eligible stocks directly by `accel_2v2_4m`; select Top-3. | Tests whether `accel_2v2_4m` alone has predictive power. |

### 2. Strategy A: original six-month average momentum baseline

| Strategy name | Family | Exact rule | Purpose |
|---|---|---|---|
| `A_momentum_6m_top3` | `A_original_6m_momentum` | Rank all eligible stocks by `momentum_6m`; select Top-3. | Baseline: the original raw 6M average momentum strategy. |

### 3. Strategy B: six-month momentum plus acceleration filter

This strategy asks whether acceleration is useful **after** requiring the stock to already be a strong 6M momentum name.

| Strategy name | Family | Exact rule | Purpose |
|---|---|---|---|
| `B_mom6m_top10_then_avg_accel_3m_top3` | `B_6m_momentum_filter_then_acceleration` | Step 1: select Top-10 by `momentum_6m`. Step 2: inside that pool, rank by `avg_accel_3m`; select Top-3. | Tests whether `avg_accel_3m` improves a strong 6M momentum shortlist. |
| `B_mom6m_top10_then_avg_accel_4m_top3` | `B_6m_momentum_filter_then_acceleration` | Step 1: select Top-10 by `momentum_6m`. Step 2: inside that pool, rank by `avg_accel_4m`; select Top-3. | Tests whether `avg_accel_4m` improves a strong 6M momentum shortlist. |
| `B_mom6m_top10_then_avg_accel_5m_top3` | `B_6m_momentum_filter_then_acceleration` | Step 1: select Top-10 by `momentum_6m`. Step 2: inside that pool, rank by `avg_accel_5m`; select Top-3. | Tests whether `avg_accel_5m` improves a strong 6M momentum shortlist. |
| `B_mom6m_top10_then_avg_accel_6m_top3` | `B_6m_momentum_filter_then_acceleration` | Step 1: select Top-10 by `momentum_6m`. Step 2: inside that pool, rank by `avg_accel_6m`; select Top-3. | Tests whether `avg_accel_6m` improves a strong 6M momentum shortlist. |
| `B_mom6m_top10_then_accel_3v3_6m_top3` | `B_6m_momentum_filter_then_acceleration` | Step 1: select Top-10 by `momentum_6m`. Step 2: inside that pool, rank by `accel_3v3_6m`; select Top-3. | Tests whether `accel_3v3_6m` improves a strong 6M momentum shortlist. |
| `B_mom6m_top10_then_accel_2v2_4m_top3` | `B_6m_momentum_filter_then_acceleration` | Step 1: select Top-10 by `momentum_6m`. Step 2: inside that pool, rank by `accel_2v2_4m`; select Top-3. | Tests whether `accel_2v2_4m` improves a strong 6M momentum shortlist. |

### 4. Strategy C: combined score

This strategy asks whether momentum and acceleration should be blended into one cross-sectional score.

| Strategy name | Family | Exact rule | Purpose |
|---|---|---|---|
| `C_z_mom6m_plus_0.5_z_avg_accel_3m_top3` | `C_combined_score` | Score = `z(momentum_6m) + 0.5 * z(avg_accel_3m)` computed cross-sectionally within each decision month; select Top-3. | Tests whether 6M momentum and `avg_accel_3m` should be blended. |
| `C_z_mom6m_plus_0.5_z_avg_accel_4m_top3` | `C_combined_score` | Score = `z(momentum_6m) + 0.5 * z(avg_accel_4m)` computed cross-sectionally within each decision month; select Top-3. | Tests whether 6M momentum and `avg_accel_4m` should be blended. |
| `C_z_mom6m_plus_0.5_z_avg_accel_5m_top3` | `C_combined_score` | Score = `z(momentum_6m) + 0.5 * z(avg_accel_5m)` computed cross-sectionally within each decision month; select Top-3. | Tests whether 6M momentum and `avg_accel_5m` should be blended. |
| `C_z_mom6m_plus_0.5_z_avg_accel_6m_top3` | `C_combined_score` | Score = `z(momentum_6m) + 0.5 * z(avg_accel_6m)` computed cross-sectionally within each decision month; select Top-3. | Tests whether 6M momentum and `avg_accel_6m` should be blended. |
| `C_z_mom6m_plus_0.5_z_accel_3v3_6m_top3` | `C_combined_score` | Score = `z(momentum_6m) + 0.5 * z(accel_3v3_6m)` computed cross-sectionally within each decision month; select Top-3. | Tests whether 6M momentum and `accel_3v3_6m` should be blended. |
| `C_z_mom6m_plus_0.5_z_accel_2v2_4m_top3` | `C_combined_score` | Score = `z(momentum_6m) + 0.5 * z(accel_2v2_4m)` computed cross-sectionally within each decision month; select Top-3. | Tests whether 6M momentum and `accel_2v2_4m` should be blended. |

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
| `avg_future_return_1m` | Average equal-weight Top-3 future 1M return. |
| `median_future_return_1m` | Median equal-weight Top-3 future 1M return. |
| `win_rate_1m` | Fraction of months where the equal-weight Top-3 1M return is positive. |
| `cumulative_return_1m_rebalanced` | Cumulative monthly-rebalanced return using `future_return_1m`. |
| `max_drawdown_1m_rebalanced` | Max drawdown of the monthly-rebalanced 1M equity curve. |
| `avg_future_max_return_1_3m` | Average equal-weight Top-3 max return across forward 1M/2M/3M horizons. |
| `median_future_max_return_1_3m` | Median of the forward 1M/2M/3M max-return metric. |
| `hit_rate_positive_max_1_3m` | Fraction of months where the forward max 1–3M return is positive. |
| `best_month_1m` | Best monthly equal-weight Top-3 1M return. |
| `worst_month_1m` | Worst monthly equal-weight Top-3 1M return. |

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
| `outputs/monthly_strategy_returns.csv` | Monthly Top-3 portfolio returns for every strategy |
| `outputs/monthly_selected_tickers.csv` | Selected tickers by strategy/month/rank |
| `outputs/strategy_summary.csv` | Summary metrics by strategy |
| `outputs/strategy_comparison.csv` | Compact comparison table sorted by average future 1M return |

## Strategy comparison

_No data available. Run `python run_all.py` first._

## Strategy summary

_No data available. Run `python run_all.py` first._

## Monthly strategy returns: 2025-01 to 2026-03

This section replaces the old single-latest-month view. It shows the requested backtest window, not only 2026-03. Full monthly history remains in `outputs/monthly_strategy_returns.csv`.

_No data available. Run `python run_all.py` first._

## Expanded selected tickers sample: 2025-01 to 2026-03

This is capped at the first 300 rows to keep the README readable. Full selections are in `outputs/monthly_selected_tickers.csv`.

_No data available. Run `python run_all.py` first._

## Interpretation

This project is a feature-level hypothesis test. The key question is:

> Does the acceleration of average momentum help distinguish stocks whose trend is still strengthening from stocks whose raw six-month momentum is already high but fading?

The most important comparisons are:

1. `A_momentum_6m_top3` versus all Strategy B variants.
2. `A_momentum_6m_top3` versus all Strategy C variants.
3. Pure acceleration variants versus Strategy A.
4. The six exact acceleration features: `avg_accel_3m`, `avg_accel_4m`, `avg_accel_5m`, `avg_accel_6m`, `accel_3v3_6m`, and `accel_2v2_4m`.
5. Rankings by `avg_future_return_1m` and `avg_future_max_return_1_3m`.
