import argparse
import time
from typing import Iterable

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from .config import (
    BENCHMARK,
    DAILY_PRICES_FILE,
    DATA_DIR,
    DOWNLOAD_CHUNK_SIZE,
    DOWNLOAD_SLEEP_SECONDS,
    SEED_UNIVERSE_FILE,
    START_DATE,
)


def chunks(xs: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def normalize_download(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if t not in raw.columns.get_level_values(0):
                continue
            one = raw[t].copy()
            if one.empty:
                continue
            one["ticker"] = t
            frames.append(one.reset_index().rename(columns={"Date": "date"}))
    else:
        one = raw.copy().reset_index().rename(columns={"Date": "date"})
        one["ticker"] = tickers[0] if tickers else ""
        frames.append(one)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    if "close" in out.columns and "adj_close" not in out.columns:
        out["adj_close"] = out["close"]
    keep = [c for c in ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"] if c in out.columns]
    out = out[keep].dropna(subset=["date", "ticker", "adj_close"])
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["ticker"] = out["ticker"].astype(str).str.upper()
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=DOWNLOAD_CHUNK_SIZE)
    parser.add_argument("--sleep", type=float, default=DOWNLOAD_SLEEP_SECONDS)
    parser.add_argument("--max-tickers", type=int, default=0, help="Debug cap. 0 means all tickers.")
    args = parser.parse_args(argv)

    if not SEED_UNIVERSE_FILE.exists():
        raise FileNotFoundError(f"Missing {SEED_UNIVERSE_FILE}; run python -m src.get_holdings_universe first")

    uni = pd.read_csv(SEED_UNIVERSE_FILE)
    tickers = uni["ticker"].dropna().astype(str).str.upper().drop_duplicates().tolist()
    if BENCHMARK not in tickers:
        tickers.append(BENCHMARK)
    if args.max_tickers > 0:
        tickers = tickers[: args.max_tickers]
        if BENCHMARK not in tickers:
            tickers.append(BENCHMARK)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_frames: list[pd.DataFrame] = []
    failures: list[dict] = []
    for batch in tqdm(list(chunks(tickers, args.chunk_size)), desc="Downloading yfinance batches"):
        try:
            raw = yf.download(
                tickers=batch,
                start=START_DATE,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            norm = normalize_download(raw, batch)
            got = set(norm["ticker"].unique()) if not norm.empty else set()
            failures.extend({"ticker": t, "reason": "no_rows_in_batch"} for t in batch if t not in got)
            if not norm.empty:
                all_frames.append(norm)
        except Exception as exc:
            print(f"Download failed for batch {batch[:5]}...: {exc}")
            failures.extend({"ticker": t, "reason": str(exc)[:200]} for t in batch)
        time.sleep(args.sleep)

    if not all_frames:
        raise RuntimeError("No price data downloaded.")
    daily = pd.concat(all_frames, ignore_index=True).sort_values(["ticker", "date"])
    daily.to_csv(DAILY_PRICES_FILE, index=False, compression="gzip")
    pd.DataFrame(failures).to_csv(DATA_DIR / "download_failures.csv", index=False)
    print(f"Saved {DAILY_PRICES_FILE}: {len(daily):,} rows, {daily['ticker'].nunique():,} tickers")


if __name__ == "__main__":
    main()
