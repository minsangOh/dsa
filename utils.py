"""Utility functions that support technical indicator calculations."""
from typing import Optional
import pandas as pd

def calculate_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Compute the Average True Range (ATR) from an OHLC dataframe."""
    if df is None or len(df) == 0:
        return None
    if not isinstance(period, int) or period < 1:
        period = 14

    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        return None

    d = df[["high", "low", "close"]].copy()
    d = d.apply(pd.to_numeric, errors="coerce")

    high_low = d["high"] - d["low"]
    prev_close = d["close"].shift(1)
    high_close = (d["high"] - prev_close).abs()
    low_close = (d["low"] - prev_close).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atr = tr.rolling(window=period, min_periods=period).mean()

    last = atr.iloc[-1]
    return float(last) if pd.notna(last) else None


def format_symbol_display(symbol: str) -> str:
    """Return a user-friendly symbol by stripping the trailing quote currency."""

    if not isinstance(symbol, str):
        return str(symbol)

    return symbol.removesuffix("_KRW")
