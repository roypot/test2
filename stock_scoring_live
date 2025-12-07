#!/usr/bin/env python3
"""
stock_scoring_live.py (SP500-enabled)

Live stock scoring with `yfinance`, now with a convenience flag to pull the full
S&P 500 universe (symbols) from Wikipedia.

Additions vs previous clean version
- `--sp500` flag: auto-load current S&P 500 tickers from Wikipedia
- Robust symbol normalization for Yahoo Finance (e.g., BRK.B -> BRK-B, BF.B -> BF-B)
- `--limit` to cap the number of tickers processed (useful when testing)

Core features (unchanged)
- Indicators: 3m/6m momentum, RSI(14), annualized volatility, max drawdown
- Trend: SMA(50) vs SMA(200)
- Valuation: PE, PB (if available via fast_info)
- Liquidity: avg volume (3m)
- Z-score normalization across universe; composite score (Momentum 40%, Quality 25%, Valuation 25%, Liquidity 10%)
- Input via `--tickers` or `--file`; optional `--out` CSV

Usage examples
--------------
    pip install yfinance pandas numpy

    # Score the whole S&P 500 (can take a while)
    python stock_scoring_live.py --sp500 --out scores_sp500.csv

    # Limit to the first 50 tickers (for a quick run)
    python stock_scoring_live.py --sp500 --limit 50

    # Mix & match with custom tickers
    python stock_scoring_live.py --sp500 --tickers AAPL MSFT NVDA --period 1y --interval 1d

Notes
-----
- Fetching ~500 tickers sequentially can be slow and may hit rate limits; consider
  reducing `--period` or using `--limit` for testing.
- The S&P 500 list is scraped from Wikipedia at runtime; if that fails, the script
  will ask you to supply tickers via `--file` or `--tickers`.
- This tool is for research/education; not investment advice.

"""
from __future__ import annotations

import argparse
import sys
import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("Missing dependency: yfinance. Install with `pip install yfinance`.", file=sys.stderr)
    raise


# ------------------------------
# Utility functions
# ------------------------------

def pct_change(series: pd.Series, periods: int) -> float:
    try:
        if len(series) < periods + 1:
            return np.nan
        return (series.iloc[-1] / series.iloc[-periods - 1]) - 1.0
    except Exception:
        return np.nan


def rsi(series: pd.Series, window: int = 14) -> float:
    """Compute RSI(14) from closing prices."""
    try:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(alpha=1/window, min_periods=window).mean()
        roll_down = down.ewm(alpha=1/window, min_periods=window).mean()
        rs = roll_up / (roll_down + 1e-12)
        rsi_vals = 100 - (100 / (1 + rs))
        return float(rsi_vals.iloc[-1])
    except Exception:
        return np.nan


def max_drawdown(series: pd.Series) -> float:
    try:
        roll_max = series.cummax()
        drawdown = series / roll_max - 1.0
        return float(drawdown.min())
    except Exception:
        return np.nan


def annualized_vol(series: pd.Series, window: int = 90, trading_days: int = 252) -> float:
    try:
        returns = series.pct_change().dropna()
        if len(returns) < max(10, window//2):
            return np.nan
        vol = returns.tail(window).std() * np.sqrt(trading_days)
        return float(vol)
    except Exception:
        return np.nan


def sma(series: pd.Series, window: int) -> float:
    try:
        return float(series.rolling(window).mean().iloc[-1])
    except Exception:
        return np.nan


# ------------------------------
# Universe helpers
# ------------------------------

def normalize_symbol_for_yahoo(sym: str) -> str:
    """Convert symbols like 'BRK.B' to Yahoo style 'BRK-B'."""
    return sym.replace('.', '-')


def load_sp500_tickers() -> List[str]:
    """Scrape the S&P 500 constituents from Wikipedia.

    Returns a list of Yahoo Finance-compatible tickers ('.' replaced with '-').
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        # Find the table with a 'Symbol' column
        candidates = [t for t in tables if any(c.strip().lower() == 'symbol' for c in t.columns)]
        if not candidates:
            raise RuntimeError("No table with 'Symbol' column found on Wikipedia page.")
        df = candidates[0]
        symbols = df['Symbol'].astype(str).str.strip().tolist()
        # Normalize for Yahoo Finance
        symbols = [normalize_symbol_for_yahoo(s) for s in symbols]
        # Deduplicate while preserving order
        seen = set()
        uniq = []
        for s in symbols:
            if s not in seen:
                uniq.append(s)
                seen.add(s)
        return uniq
    except Exception as e:
        print(f"Failed to load S&P 500 from Wikipedia: {e}", file=sys.stderr)
        return []


# ------------------------------
# Data fetch
# ------------------------------

def fetch_prices(ticker: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def fetch_fast_info(ticker: str) -> Dict[str, float]:
    """Pull simple valuation ratios if available."""
    info = {
        "pe": np.nan,
        "pb": np.nan,
        "market_cap": np.nan,
    }
    try:
        tk = yf.Ticker(ticker)
        # fast_info is lightweight; attributes may vary by ticker
        fi = getattr(tk, "fast_info", {})
        if fi:
            pe = fi.get("trailing_pe", np.nan)
            pb = fi.get("price_to_book", np.nan)
            mc = fi.get("market_cap", np.nan)
            info["pe"] = float(pe) if pe not in (None, "None") else np.nan
            info["pb"] = float(pb) if pb not in (None, "None") else np.nan
            info["market_cap"] = float(mc) if mc not in (None, "None") else np.nan
    except Exception:
        pass
    return info


# ------------------------------
# Indicator & scoring
# ------------------------------

def compute_indicators(df: pd.DataFrame) -> Dict[str, float]:
    close = df["Close"].dropna()
    vol_series = df["Volume"].dropna()

    ind = {
        "mom_3m": pct_change(close, 63),           # ~63 trading days ~3m
        "mom_6m": pct_change(close, 126),          # ~126 trading days ~6m
        "rsi_14": rsi(close, 14),
        "vol_ann": annualized_vol(close, 90),
        "mdd": max_drawdown(close),
        "sma50": sma(close, 50),
        "sma200": sma(close, 200),
        "sma_trend": np.nan,                       # will fill below
        "avg_volume_3m": float(vol_series.tail(63).mean()) if len(vol_series) >= 20 else np.nan,
    }
    if not np.isnan(ind["sma50"]) and not np.isnan(ind["sma200"]):
        ind["sma_trend"] = (ind["sma50"] / ind["sma200"]) - 1.0
    return ind


def zscore(series: pd.Series) -> pd.Series:
    series = series.copy()
    mu = series.mean()
    sigma = series.std(ddof=0)
    if np.isfinite(sigma) and sigma > 0:
        return (series - mu) / sigma
    else:
        # If constant or all NaN, return zeros
        return pd.Series(np.zeros(len(series)), index=series.index)


def build_scores(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).set_index("ticker")

    # Z-score transformations (higher is better)
    # Momentum: 3m, 6m, SMA trend, RSI centered around 50
    df["rsi_centered"] = df["rsi_14"] - 50
    momentum_cols = ["mom_3m", "mom_6m", "sma_trend", "rsi_centered"]

    # Quality: lower vol and smaller drawdown are better => invert signs before z-score
    df["vol_inv"] = -df["vol_ann"]
    df["mdd_inv"] = -df["mdd"]  # mdd is negative; -mdd makes a positive when smaller drawdown
    quality_cols = ["vol_inv", "mdd_inv"]

    # Valuation: lower PE/PB are better => invert
    df["pe_inv"] = -df["pe"]
    df["pb_inv"] = -df["pb"]
    valuation_cols = ["pe_inv", "pb_inv"]

    # Liquidity: higher avg volume better
    liquidity_cols = ["avg_volume_3m"]

    # Compute z-scores per group
    def group_score(cols: List[str]) -> pd.Series:
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.Series(np.zeros(len(df)), index=df.index)
        Z = pd.DataFrame({c: zscore(df[c].astype(float)) for c in present})
        # average across available metrics (skip NaNs)
        return Z.mean(axis=1)

    df["score_momentum"] = group_score(momentum_cols)
    df["score_quality"] = group_score(quality_cols)
    df["score_valuation"] = group_score(valuation_cols)
    df["score_liquidity"] = group_score(liquidity_cols)

    # Aggregate score with weights (tune as desired)
    w = {
        "score_momentum": 0.4,
        "score_quality": 0.25,
        "score_valuation": 0.25,
        "score_liquidity": 0.10,
    }
    df["total_score"] = (
        w["score_momentum"] * df["score_momentum"]
        + w["score_quality"] * df["score_quality"]
        + w["score_valuation"] * df["score_valuation"]
        + w["score_liquidity"] * df["score_liquidity"]
    )

    # Tidy output columns
    cols_out = [
        "total_score",
        "score_momentum", "score_quality", "score_valuation", "score_liquidity",
        "mom_3m", "mom_6m", "sma_trend", "rsi_14",
        "vol_ann", "mdd",
        "pe", "pb", "market_cap",
        "avg_volume_3m",
        "last_price",
    ]

    for c in cols_out:
        if c not in df.columns:
            df[c] = np.nan

    df = df[cols_out].sort_values("total_score", ascending=False)
    return df


# ------------------------------
# Main
# ------------------------------

def load_tickers_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Live stock scoring using yfinance (with S&P 500 support)")
    parser.add_argument("--tickers", nargs="*", help="Tickers (space-separated)")
    parser.add_argument("--file", type=str, default=None, help="Path to text file with tickers (one per line)")
    parser.add_argument("--sp500", action="store_true", help="Use the current S&P 500 constituents from Wikipedia")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of tickers to process")
    parser.add_argument("--period", type=str, default="1y", help="yfinance period (e.g., 6mo, 1y, 2y)")
    parser.add_argument("--interval", type=str, default="1d", help="yfinance interval (e.g., 1d, 1h)")
    parser.add_argument("--out", type=str, default=None, help="Optional CSV output path")

    args = parser.parse_args()

    tickers: List[str] = []
    if args.sp500:
        sp = load_sp500_tickers()
        if not sp:
            print("Warning: failed to fetch S&P 500 tickers online. Provide --file or --tickers.", file=sys.stderr)
        else:
            tickers.extend(sp)
    if args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        tickers.extend(load_tickers_from_file(args.file))
    if args.tickers:
        tickers.extend(args.tickers)

    # Normalize, uppercase and dedupe
    norm = []
    seen = set()
    for t in tickers:
        t2 = normalize_symbol_for_yahoo(t.strip().upper())
        if t2 and t2 not in seen:
            norm.append(t2)
            seen.add(t2)
    tickers = norm

    if args.limit is not None:
        tickers = tickers[:max(0, args.limit)]

    if not tickers:
        print("No tickers provided. Use --sp500, --tickers or --file.")
        sys.exit(1)

    rows: List[Dict] = []

    for i, t in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Fetching {t}...", file=sys.stderr)
        df = fetch_prices(t, period=args.period, interval=args.interval)
        if df is None:
            print(f"  Warning: no price data for {t}", file=sys.stderr)
            continue

        ind = compute_indicators(df)
        fi = fetch_fast_info(t)

        row = {
            "ticker": t,
            "last_price": float(df["Close"].iloc[-1]) if not df.empty else np.nan,
            **ind,
            **fi,
        }
        rows.append(row)

    if not rows:
        print("No data gathered. Exiting.", file=sys.stderr)
        sys.exit(1)

    scores = build_scores(rows)

    # Print to stdout
    pd.options.display.float_format = lambda x: f"{x:,.4f}"
    print("\n===== Stock Scores (higher is better) =====")
    print(scores.head(len(scores)))

    # Save CSV if requested
    if args.out:
        scores.to_csv(args.out)
        print(f"\nSaved CSV to: {args.out}")


if __name__ == "__main__":
    main()
