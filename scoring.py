# scoring.py
import numpy as np
import pandas as pd
import yfinance as yf

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
# Data fetch
# ------------------------------

def fetch_prices(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def fetch_fast_info(ticker: str) -> dict:
    info = {"pe": np.nan, "pb": np.nan, "market_cap": np.nan}
    try:
        tk = yf.Ticker(ticker)
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

def compute_indicators(df: pd.DataFrame) -> dict:
    close = df["Close"].dropna()
    vol_series = df["Volume"].dropna()

    ind = {
        "mom_3m": pct_change(close, 63),
        "mom_6m": pct_change(close, 126),
        "rsi_14": rsi(close, 14),
        "vol_ann": annualized_vol(close, 90),
        "mdd": max_drawdown(close),
        "sma50": sma(close, 50),
        "sma200": sma(close, 200),
        "sma_trend": np.nan,
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
        return pd.Series(np.zeros(len(series)), index=series.index)


def build_scores(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).set_index("ticker")

    # Momentum: 3m, 6m, SMA trend, RSI centered around 50
    df["rsi_centered"] = df["rsi_14"] - 50

    # Quality: invert vol and drawdown
    df["vol_inv"] = -df["vol_ann"]
    df["mdd_inv"] = -df["mdd"]

    # Valuation: invert PE / PB
    df["pe_inv"] = -df["pe"]
    df["pb_inv"] = -df["pb"]

    def group_score(cols: list[str]) -> pd.Series:
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.Series(np.zeros(len(df)), index=df.index)
        Z = pd.DataFrame({c: zscore(df[c].astype(float)) for c in present})
        return Z.mean(axis=1)

    df["score_momentum"]  = group_score(["mom_3m", "mom_6m", "sma_trend", "rsi_centered"])
    df["score_quality"]   = group_score(["vol_inv", "mdd_inv"])
    df["score_valuation"] = group_score(["pe_inv", "pb_inv"])
    df["score_liquidity"] = group_score(["avg_volume_3m"])

    w = {"score_momentum": 0.4, "score_quality": 0.25, "score_valuation": 0.25, "score_liquidity": 0.10}
    df["total_score"] = (
        w["score_momentum"] * df["score_momentum"]
        + w["score_quality"] * df["score_quality"]
        + w["score_valuation"] * df["score_valuation"]
        + w["score_liquidity"] * df["score_liquidity"]
    )

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
    return df[cols_out].sort_values("total_score", ascending=False)


# ------------------------------
# Public API
# ------------------------------

def score_universe(tickers: list[str], period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    rows: list[dict] = []
    for t in tickers:
        df = fetch_prices(t, period=period, interval=interval)
        if df is None or df.empty:
            continue
        ind = compute_indicators(df)
        fi  = fetch_fast_info(t)
        rows.append({
            "ticker": t,
            "last_price": float(df["Close"].iloc[-1]),
            **ind, **fi
        })
    if not rows:
        return pd.DataFrame()
    return build_scores(rows)
