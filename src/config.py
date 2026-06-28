from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SOURCE_ETFS_FILE = DATA_DIR / "source_etfs.csv"
MANUAL_TICKERS_FILE = DATA_DIR / "manual_tickers.csv"
SEED_UNIVERSE_FILE = DATA_DIR / "seed_universe.csv"
DAILY_PRICES_FILE = DATA_DIR / "daily_prices.csv.gz"

START_DATE = "2014-01-01"
BACKTEST_START = "2016-01-01"
BENCHMARK = "QQQ"
TOP_N = 3
FILTER_TOP_N = 10
ACCEL_LOOKBACK_MONTHS = [3, 4, 5, 6]
MOMENTUM_LOOKBACK_MONTHS = [3, 4, 5, 6]
BASE_MOMENTUM_MONTHS = 6

# Strategy C uses:
# score = z(momentum_6m) + COMBINED_ACCEL_WEIGHT * z(acceleration_feature)
COMBINED_ACCEL_WEIGHT = 0.3

DOWNLOAD_CHUNK_SIZE = 80
DOWNLOAD_SLEEP_SECONDS = 1.0

# README monthly strategy return window.
# REPORT_END_MONTH = None means: automatically use the previous calendar month.
REPORT_START_MONTH = "2025-01"
REPORT_END_MONTH = None
