"""Trading strategy & indicator calculation module.

This module separates the core logic for calculating indicators and evaluating
buy conditions from the main bot orchestration. It runs in a separate process
to avoid blocking the main UI thread.
"""
import multiprocessing
import time
from collections import deque
from decimal import Decimal, InvalidOperation

import numpy as np
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal, pyqtSlot, QTimer

from bithumb_api import BithumbWrapper
import config
from config import (
    ATR_PERIOD,
    BBANDS_PERIOD,
    BBANDS_STDDEV,
    BLOCK_LIST,
    C1_ATR_BREAKOUT_MULTIPLIER,
    C1_HIGH_PERIOD,
    C1_OBI_THRESHOLD,
    C1_VOL_ZSCORE_THRESHOLD,
    C2_RSI_OVERSOLD_THRESHOLD,
    C2_VOL_ZSCORE_THRESHOLD,
    C2_VWAP_STDDEV_MULTIPLIER,
    C3_LOW_PERIOD,
    C3_VOL_ZSCORE_THRESHOLD,
    C3_WICK_BODY_RATIO,
    C4_RANGE_ATR_MULTIPLIER,
    C4_SQUEEZE_PERIOD,
    C4_SQUEEZE_THRESHOLD,
    C4_VOL_ZSCORE_THRESHOLD,
    C5_1MIN_TICK_RISE,
    C5_2MIN_TICK_RISE,
    C5_ORDERBOOK_RATIO,
    C5_ORDERBOOK_TICKS,
    C5_SELL_LIQUIDITY_FACTOR,
    C5_SELL_LIQUIDITY_RANGE_PCT,
    C5_VOLUME_LONG_PERIOD,
    C5_VOLUME_SHORT_PERIOD,
    C5_VOLUME_SPIKE_FACTOR,
    COOLDOWN_PERIOD_SECONDS,
    INDICATOR_CACHE_TTL_SECONDS,
    KELTNER_ATR_MULTIPLIER,
    KELTNER_PERIOD,
    LIQUIDITY_FILTER_KRW,
    MANUAL_EXCLUDE_LIST,
    MAX_POSITIONS,
    MIN_PRICE_KRW,
    OBI_LEVELS,
    RSI_PERIOD,
    STABLE_COINS,
    VOL_ZSCORE_PERIOD,
)
from utils import format_symbol_display


# --- Process-based Worker Function ---

def _calculate_indicators_for_process(df, df_5m):
    """Isolated indicator calculation logic for multiprocessing."""
    max_period = max(
        VOL_ZSCORE_PERIOD,
        BBANDS_PERIOD,
        KELTNER_PERIOD,
        C1_HIGH_PERIOD,
        C3_LOW_PERIOD,
    )
    if df is None or df.empty or len(df) < max_period + 5:
        return {}

    df_float = df.astype(float)
    df_float.ta.atr(length=ATR_PERIOD, append=True)
    df_float.ta.rsi(length=RSI_PERIOD, append=True)
    df_float.ta.bbands(length=BBANDS_PERIOD, std=BBANDS_STDDEV, append=True)
    df_float.ta.kc(length=KELTNER_PERIOD, scalar=KELTNER_ATR_MULTIPLIER, append=True)
    df_float.ta.vwap(append=True)

    if "VWAP_D" in df_float.columns:
        rolling_std = df_float["close"].rolling(window=BBANDS_PERIOD).std().fillna(0)
        df_float["VWAP_UPPER"] = (
            df_float["VWAP_D"] + (rolling_std * C2_VWAP_STDDEV_MULTIPLIER)
        )
        df_float["VWAP_LOWER"] = (
            df_float["VWAP_D"] - (rolling_std * C2_VWAP_STDDEV_MULTIPLIER)
        )

    vol_mean = df_float["volume"].rolling(window=VOL_ZSCORE_PERIOD).mean()
    vol_std = df_float["volume"].rolling(window=VOL_ZSCORE_PERIOD).std().fillna(0)
    df_float["VOL_ZSCORE"] = ((df_float["volume"] - vol_mean) / vol_std).where(
        vol_std > 0, 0
    )
    df_float["HIGH_20"] = df_float["high"].rolling(window=C1_HIGH_PERIOD).max()
    df_float["LOW_30"] = df_float["low"].rolling(window=C3_LOW_PERIOD).min()
    df_float["HIGH_30MIN"] = df_float["high"].rolling(window=30).max()

    latest_indicators = df_float.iloc[-1].to_dict()

    if df_5m is not None and not df_5m.empty and len(df_5m) >= C5_VOLUME_LONG_PERIOD:
        df_5m_float = df_5m.astype(float)
        latest_indicators["c5_short_vol_avg"] = (
            df_5m_float["volume"]
            .rolling(window=C5_VOLUME_SHORT_PERIOD)
            .mean()
            .iloc[-1]
        )
        latest_indicators["c5_long_vol_avg"] = (
            df_5m_float["volume"]
            .rolling(window=C5_VOLUME_LONG_PERIOD)
            .mean()
            .iloc[-1]
        )

    bb_bandwidth_col = f"BBB_{BBANDS_PERIOD}_{float(BBANDS_STDDEV)}"
    kc_upper_col = f"KCUe_{KELTNER_PERIOD}_{float(KELTNER_ATR_MULTIPLIER)}"
    kc_lower_col = f"KCLe_{KELTNER_PERIOD}_{float(KELTNER_ATR_MULTIPLIER)}"

    if all(
        c in df_float.columns
        for c in [bb_bandwidth_col, kc_upper_col, kc_lower_col]
    ):
        kc_width = df_float[kc_upper_col] - df_float[kc_lower_col]
        kc_width[kc_width == 0] = np.nan
        squeeze_ratio = df_float[bb_bandwidth_col] / kc_width
        is_in_squeeze = (
            squeeze_ratio.iloc[-C4_SQUEEZE_PERIOD:] <= C4_SQUEEZE_THRESHOLD
        ).all()
        latest_indicators["is_in_squeeze"] = is_in_squeeze
    else:
        latest_indicators["is_in_squeeze"] = False

    for key, val in df.iloc[-1].items():
        latest_indicators[key] = val
    if len(df) > 1:
        for key, val in df.iloc[-2].items():
            latest_indicators[f"PREV_{key.upper()}"] = val
    if f"RSI_{RSI_PERIOD}" in df_float.columns and len(df_float) >= 2:
        latest_indicators["PREV_RSI"] = float(
            df_float.iloc[-2][f"RSI_{RSI_PERIOD}"]
        )
    return latest_indicators


def run_indicator_process_task(api_key, secret_key, symbol, tick_size, result_queue):
    """Function to be run in a separate process for indicator calculation."""
    import pandas as pd
    import pandas_ta as ta

    try:
        api = BithumbWrapper(api_key, secret_key)
        df_1m = api.get_candlestick(symbol, "1m")
        df_5m = api.get_candlestick(symbol, "5m")
        if df_1m is None or df_1m.empty:
            raise ValueError("1 minute data is empty.")

        indicators = _calculate_indicators_for_process(df_1m, df_5m)
        if not indicators:
            raise ValueError("The index calculation result is empty.")

        df_1h = api.get_candlestick(symbol, "1h")
        if df_1h is not None and not df_1h.empty:
            df_1h_float = df_1h.astype(float)
            df_1h_float.ta.vwap(append=True)
            if "VWAP_D" in df_1h_float.columns:
                indicators["vwap_1h"] = df_1h_float["VWAP_D"].iloc[-1]

        if tick_size and tick_size > 0:
            if df_1m is not None and len(df_1m) >= 2:
                prev = df_1m.iloc[-2]
                indicators["min1_tick_rise"] = (
                    float(prev["close"]) - float(prev["open"])
                ) / float(tick_size)
            if df_1m is not None and len(df_1m) >= 3:
                prev_two = df_1m.iloc[-3]
                latest_prev = df_1m.iloc[-2]
                indicators["min2_tick_rise"] = (
                    float(latest_prev["close"]) - float(prev_two["open"])
                ) / float(tick_size)

        result_queue.put((symbol, indicators))
    except Exception as e:
        print(f"[IndicatorProcess:{symbol}] Error: {e}")
        result_queue.put((symbol, {"__failed__": True}))


# --- Qt-based Workers and Signals (for non-blocking tasks) ---


class WorkerSignals(QObject):
    """Container for signals emitted by buy-check workers toward the UI."""

    buy_signal_found = pyqtSignal(dict)
    completed = pyqtSignal(str)


class UniverseScanSignals(QObject):
    """Signals for the universe scan worker."""

    scan_completed = pyqtSignal(list)
    error = pyqtSignal(str)


class UniverseScanWorker(QRunnable):
    """Worker that fetches all ticker data and builds the tradable universe."""

    def __init__(self, api_key, secret_key):
        super().__init__()
        self.api = BithumbWrapper(api_key, secret_key)
        self.signals = UniverseScanSignals()

    @pyqtSlot()
    def run(self):
        try:
            tickers = self.api.get_all_ticker_data()
            if not tickers:
                raise ValueError(
                    "Unable to retrieve ticker information for universe scan."
                )

            tradable_universe = []
            for ticker, data in tickers.items():
                symbol = f"{ticker}_KRW"
                price = data.get("closing_price")
                quote_volume = data.get("quoteVolume", Decimal("0"))

                if not price or price < MIN_PRICE_KRW:
                    continue
                if quote_volume < LIQUIDITY_FILTER_KRW:
                    continue
                if (
                    any(s in ticker for s in STABLE_COINS)
                    or symbol in BLOCK_LIST
                    or symbol in MANUAL_EXCLUDE_LIST
                ):
                    continue
                tradable_universe.append(symbol)
            self.signals.scan_completed.emit(tradable_universe)
        except Exception as e:
            self.signals.error.emit(str(e))


class BuyCheckWorker(QRunnable):
    """Evaluate buy conditions for a symbol using cached indicators and data."""

    def __init__(self, bot_instance, api_key, secret_key, coin_data, indicators):
        """Capture bot reference, symbol snapshot, and indicator bundle."""
        super().__init__()
        self.bot = bot_instance  # Keep for logging and state access
        self.api = BithumbWrapper(api_key, secret_key)
        self.coin = coin_data
        self.indicators = indicators
        self.signals = WorkerSignals()

    def _check_c1(self, price_float, vwap_1h):
        """Check for strong new high breakout (C1)."""
        if vwap_1h is not None and price_float < vwap_1h:
            return False

        i = self.indicators
        atr_val = i.get(f"ATRr_{ATR_PERIOD}", 0)
        if atr_val is None:
            return False

        breakout_price = i.get("HIGH_20", np.inf) + max(
            price_float * 0.0007, C1_ATR_BREAKOUT_MULTIPLIER * atr_val
        )
        vol_zscore = i.get("VOL_ZSCORE", 0)

        if price_float >= breakout_price and vol_zscore >= C1_VOL_ZSCORE_THRESHOLD:
            return True
        return False

    def _check_c2(self, price_float, vwap_1h):
        """Check for mean reversion after oversold (C2)."""
        if vwap_1h is not None and price_float >= vwap_1h:
            return False

        i = self.indicators
        return (
            float(i.get("PREV_CLOSE", np.inf)) <= i.get("VWAP_LOWER", np.inf)
            and price_float > i.get("VWAP_LOWER", np.inf)
            and i.get("PREV_RSI", 100) <= C2_RSI_OVERSOLD_THRESHOLD
            and i.get(f"RSI_{RSI_PERIOD}", 0) > i.get("PREV_RSI", 100)
            and i.get("VOL_ZSCORE", 0) >= C2_VOL_ZSCORE_THRESHOLD
        )

    def _check_c3(self, price, price_float, vwap_1h):
        """Check for stop hunt reversal (C3)."""
        if vwap_1h is not None and price_float >= vwap_1h:
            return False

        i = self.indicators
        body = abs(i.get("open", Decimal("0")) - price)
        wick_body_ratio = (
            (min(i.get("open", price), price) - i.get("low", price)) / body
            if body > Decimal("1e-9")
            else 0.0
        )
        return (
            float(i.get("PREV_LOW", np.inf)) < i.get("LOW_30", np.inf)
            and price_float > i.get("LOW_30", np.inf)
            and wick_body_ratio >= C3_WICK_BODY_RATIO
            and i.get("VOL_ZSCORE", 0) >= C3_VOL_ZSCORE_THRESHOLD
        )

    def _check_c4(self, price, price_float, vwap_1h):
        """Check for squeeze breakout (C4)."""
        if vwap_1h is not None and price_float < vwap_1h:
            return False

        i = self.indicators
        if not i.get("is_in_squeeze", False):
            return False

        atr_val = i.get(f"ATRr_{ATR_PERIOD}", 0)
        if atr_val is None or not isinstance(atr_val, (int, float)):
            return False

        breakout_range = Decimal(str(C4_RANGE_ATR_MULTIPLIER)) * Decimal(
            str(atr_val)
        )
        current_range = i.get("high", Decimal("0")) - i.get("low", Decimal("0"))
        bbu_val = i.get(f"BBU_{BBANDS_PERIOD}_{float(BBANDS_STDDEV)}")

        return (
            bbu_val is not None
            and current_range >= breakout_range
            and i.get("VOL_ZSCORE", 0) >= C4_VOL_ZSCORE_THRESHOLD
            and price > Decimal(str(bbu_val))
        )

    def _check_c5(self, symbol, price_float):
        """Check for minute candle tick rise breakout (C5)."""
        i = self.indicators
        high_30min = i.get("HIGH_30MIN")
        if high_30min is not None and high_30min > 0:
            if (high_30min - price_float) / high_30min >= 0.05:
                return False

        original_c5_passed = (
            i.get("min1_tick_rise", 0) >= C5_1MIN_TICK_RISE
            and i.get("min2_tick_rise", 0) < C5_2MIN_TICK_RISE
        )
        if not original_c5_passed:
            return False

        short_vol_avg = i.get("c5_short_vol_avg")
        long_vol_avg = i.get("c5_long_vol_avg")
        if short_vol_avg is None or long_vol_avg is None or long_vol_avg == 0:
            return False
        if short_vol_avg < (long_vol_avg * C5_VOLUME_SPIKE_FACTOR):
            return False

        orderbook = self.api.get_orderbook(symbol)
        if not orderbook or not orderbook.get("bids") or not orderbook.get("asks"):
            return False

        bids = orderbook["bids"][:C5_ORDERBOOK_TICKS]
        asks = orderbook["asks"][:C5_ORDERBOOK_TICKS]
        bid_size = sum(entry["quantity"] for entry in bids)
        ask_size = sum(entry["quantity"] for entry in asks)
        if ask_size == 0:
            return False
        if (bid_size / ask_size) < C5_ORDERBOOK_RATIO:
            return False

        upper_bound = price_float * (1 + C5_SELL_LIQUIDITY_RANGE_PCT / 100)
        current_sell_wall = sum(
            ask["quantity"]
            for ask in orderbook["asks"]
            if price_float <= float(ask["price"]) <= upper_bound
        )

        now = time.time()
        with self.bot.state_lock:
            if symbol not in self.bot.strategy_manager.c5_sell_liquidity_history:
                self.bot.strategy_manager.c5_sell_liquidity_history[symbol] = deque(
                    maxlen=100
                )

            history = self.bot.strategy_manager.c5_sell_liquidity_history[symbol]
            history.append((now, current_sell_wall))

            if len(history) < 20:
                return True

            five_mins_ago = now - 300
            recent_values = [val for ts, val in history if ts >= five_mins_ago]

            if not recent_values:
                return True

            historical_avg_sell_wall = sum(recent_values) / len(recent_values)

            if (
                historical_avg_sell_wall > 0
                and current_sell_wall
                > (historical_avg_sell_wall * C5_SELL_LIQUIDITY_FACTOR)
            ):
                return False

        return True

    @pyqtSlot()
    def run(self):
        """Screen all buy strategies and notify the bot when one fires."""
        symbol = self.coin["symbol"]
        display_symbol = format_symbol_display(symbol)
        try:
            i = self.indicators
            price = i.get("close", Decimal("0"))
            if price == Decimal("0"):
                return

            price_float = float(price)
            vwap_1h = i.get("vwap_1h")

            c1_candidate = self._check_c1(price_float, vwap_1h)
            c2 = self._check_c2(price_float, vwap_1h)
            c3 = self._check_c3(price, price_float, vwap_1h)
            c4 = self._check_c4(price, price_float, vwap_1h)
            c5 = self._check_c5(symbol, price_float)

            c1 = False
            if c1_candidate or c2 or c3 or c4 or c5:
                orderbook_data = self.api.get_orderbook(symbol)
                if (
                    not orderbook_data
                    or not orderbook_data.get("bids")
                    or not orderbook_data.get("asks")
                ):
                    self.bot.log(
                        "Trade Fail",
                        f"[{display_symbol}] Buy signal processing failed due to order book query failure (pre-check)",
                    )
                    return

                if c1_candidate:
                    bid_vol = sum(
                        entry["quantity"]
                        for entry in orderbook_data["bids"][:OBI_LEVELS]
                    )
                    ask_vol = sum(
                        entry["quantity"]
                        for entry in orderbook_data["asks"][:OBI_LEVELS]
                    )
                    total_vol = bid_vol + ask_vol
                    obi_value = (
                        float((bid_vol - ask_vol) / total_vol) if total_vol > 0 else 0.0
                    )
                    if obi_value >= C1_OBI_THRESHOLD:
                        c1 = True
            else:
                return

            if c1 or c2 or c3 or c4 or c5:
                if not self.bot.check_buy_constraints(symbol, price, orderbook_data):
                    return

                with self.bot.state_lock:
                    available = self.bot.krw_balance - self.bot.krw_reserved
                    budget = (
                        (
                            self.bot.get_total_assets_internal()
                            - self.bot.get_btc_value_internal()
                        )
                        * Decimal("0.25")
                        if self.bot.fixed_buy_amount == 0
                        else self.bot.fixed_buy_amount
                    )

                budget = min(budget, available)
                min_order_val = self.api.get_min_order_value(symbol)
                if budget < min_order_val:
                    return

                ask_price = orderbook_data["asks"][0]["price"]
                amount = self.api.amount_to_precision(symbol, budget / ask_price)
                if not amount or (amount * ask_price) < min_order_val:
                    return

                triggered = []
                if c1:
                    triggered.append("C1")
                if c2:
                    triggered.append("C2")
                if c3:
                    triggered.append("C3")
                if c4:
                    triggered.append("C4")
                if c5:
                    triggered.append("C5")

                self.signals.buy_signal_found.emit(
                    {
                        "symbol": symbol,
                        "ask_price": ask_price,
                        "amount": amount,
                        "strategies": triggered,
                    }
                )
        except (KeyError, TypeError, InvalidOperation) as e:
            self.bot.log("Errors", f"BuyCheckWorker <{display_symbol}>: {e}")
        finally:
            self.signals.completed.emit(symbol)


class StrategyManager(QObject):
    """Manages the trading strategy evaluation lifecycle."""

    log_signal = pyqtSignal(str, str)
    status_update_signal = pyqtSignal(str)
    buy_signal_found = pyqtSignal(dict)

    def __init__(self, bot_instance):
        super().__init__()
        self.bot = bot_instance
        self.is_running = True

        self.indicator_cache = {}
        self.c5_sell_liquidity_history = {}
        self.processing_queue = deque()
        self.master_cycle_start_time = 0
        self.is_busy = False  # Flag to indicate a task is running

        self.buy_thread_pool = QThreadPool()
        self.buy_thread_pool.setMaxThreadCount(1)
        self.scan_thread_pool = QThreadPool()
        self.scan_thread_pool.setMaxThreadCount(1)

        # --- Multiprocessing Setup ---
        self.mp_manager = multiprocessing.Manager()
        self.indicator_results_queue = self.mp_manager.Queue()
        self.indicator_process = None

        # Main processing loop timer
        self.main_loop_timer = QTimer()
        self.main_loop_timer.timeout.connect(self._main_loop)
        self.main_loop_timer.start(200)  # Tick every 200ms

        # Timer to trigger a new universe scan
        self.scan_timer = QTimer()
        self.scan_timer.timeout.connect(self.start_universe_scan)

    def stop(self):
        self.is_running = False
        self.main_loop_timer.stop()
        self.scan_timer.stop()
        self.buy_thread_pool.clear()
        self.buy_thread_pool.waitForDone()
        self.scan_thread_pool.clear()
        self.scan_thread_pool.waitForDone()
        if self.indicator_process and self.indicator_process.is_alive():
            self.indicator_process.terminate()
            self.indicator_process.join()
        try:
            self.mp_manager.shutdown()
        except Exception:
            pass # Already shutdown

    def start_scan_timer(self):
        self.scan_timer.start(config.MASTER_SCAN_INTERVAL_SECONDS * 1000)
        self.start_universe_scan() # Trigger first scan immediately

    @pyqtSlot()
    def _main_loop(self):
        if not self.is_running or self.is_busy:
            return

        # 1. Check for indicator results
        if not self.indicator_results_queue.empty():
            try:
                symbol, indicators = self.indicator_results_queue.get_nowait()
                self.handle_indicator_result(symbol, indicators)
                # Don't return, let buy check start if needed, but loop will be busy
            except Exception:
                pass # Queue was empty, race condition

        # 2. If not busy with a buy check, process next symbol
        if self.is_busy:
            return

        if self.processing_queue:
            symbol_to_process = self.processing_queue.popleft()
            self.start_indicator_process(symbol_to_process)
        else:
            # Queue is empty, do nothing and wait for next universe scan
            self.status_update_signal.emit(
                f"<Cycle {self.bot.cycle_count}> Waiting for next scan..."
            )

    @pyqtSlot()
    def start_universe_scan(self):
        """Initiates the universe scan by dispatching a background worker."""
        if self.is_busy:
            return

        self.is_busy = True
        self.bot.cycle_count += 1
        self.status_update_signal.emit(f"<Cycle {self.bot.cycle_count}> Scanning universe...")
        worker = UniverseScanWorker(self.bot.api.api_key, self.bot.api.secret_key)
        worker.signals.scan_completed.connect(self.handle_scan_completion)
        worker.signals.error.connect(self.handle_scan_error)
        self.scan_thread_pool.start(worker)

    @pyqtSlot(str)
    def handle_scan_error(self, error_msg):
        self.bot.log("Errors", error_msg)
        self.is_busy = False

    @pyqtSlot(list)
    def handle_scan_completion(self, tradable_universe):
        self.bot.log("System", f"<Cycle {self.bot.cycle_count}> Found {len(tradable_universe)} symbols.")
        with self.bot.state_lock:
            self.bot.tradable_universe = tradable_universe

        targets, reason = self._build_targets_for_cycle()
        if not targets:
            self.bot.log("System", f"No targets to process. Reason: {reason}")
        else:
            self.bot.log("System", f"Queueing {len(targets)} symbols for analysis.")
            self.processing_queue = deque(targets)

        self.is_busy = False

    def start_indicator_process(self, symbol):
        self.is_busy = True
        display_symbol = format_symbol_display(symbol)
        total = len(self.bot.tradable_universe)
        processed = total - len(self.processing_queue)
        progress_pct = (processed / total) * 100 if total > 0 else 0

        status_msg = f"<Cycle {self.bot.cycle_count}> [{display_symbol}] Evaluating {processed}/{total} ({progress_pct:.0f}%)"
        self.status_update_signal.emit(status_msg)
        self.bot.log("System", status_msg)

        args = (
            self.bot.api.api_key,
            self.bot.api.secret_key,
            symbol,
            self.bot.tick_sizes.get(symbol),
            self.indicator_results_queue,
        )
        self.indicator_process = multiprocessing.Process(
            target=run_indicator_process_task, args=args
        )
        self.indicator_process.start()

    def handle_indicator_result(self, symbol, indicators):
        if indicators and not indicators.get("__failed__", False):
            with self.bot.state_lock:
                self.indicator_cache[symbol] = {
                    "data": indicators,
                    "timestamp": time.time(),
                }
            self.start_buy_evaluation(symbol, indicators)
        else:
            self.bot.log("System", f"Indicator calculation failed for {symbol}.")
            self.is_busy = False # Free up the loop if calc fails

    def _build_targets_for_cycle(self):
        with self.bot.state_lock:
            portfolio_keys = set(self.bot.portfolio.keys())
            trade_history_copy = list(self.bot.trade_history)
            portfolio_size = len(self.bot.portfolio)

        if portfolio_size >= MAX_POSITIONS:
            return [], "max_positions"

        now = time.time()
        final_targets = []
        for symbol in self.bot.tradable_universe:
            if symbol in portfolio_keys:
                continue
            last_trade_time = 0
            for trade_symbol, trade_time, trade_side in reversed(trade_history_copy):
                if trade_symbol == symbol and trade_side == "sell":
                    last_trade_time = trade_time
                    break
            if now - last_trade_time < COOLDOWN_PERIOD_SECONDS:
                continue
            final_targets.append(symbol)

        if not final_targets:
            return [], "no_targets"
        return final_targets, None

    def start_buy_evaluation(self, symbol, indicators):
        self.bot.log("System", f"[{format_symbol_display(symbol)}] Starting buy evaluation.")
        with self.bot.state_lock:
            if len(self.bot.portfolio) >= MAX_POSITIONS:
                self.bot.log("System", "Max positions reached, stopping cycle.")
                self.processing_queue.clear()
                self.is_busy = False
                return

        coin_data = {"symbol": symbol}
        worker = BuyCheckWorker(
            self.bot,
            self.bot.api.api_key,
            self.bot.api.secret_key,
            coin_data,
            indicators,
        )
        worker.signals.buy_signal_found.connect(self.buy_signal_found)
        worker.signals.completed.connect(self.on_buy_worker_completed)
        self.buy_thread_pool.start(worker)

    @pyqtSlot(str)
    def on_buy_worker_completed(self, symbol):
        self.bot.log("System", f"[{format_symbol_display(symbol)}] Buy evaluation completed.")
        self.is_busy = False

    @pyqtSlot()
    def cleanup_indicator_cache(self):
        try:
            if not self.is_running:
                return

            now = time.time()
            with self.bot.state_lock:
                stale_keys = [
                    symbol
                    for symbol, cache_entry in self.indicator_cache.items()
                    if now - cache_entry.get("timestamp", 0)
                    > INDICATOR_CACHE_TTL_SECONDS
                ]

                if stale_keys:
                    for key in stale_keys:
                        del self.indicator_cache[key]
                    self.bot.log(
                        "System",
                        f"Cleaned up {len(stale_keys)} expired indicator data.",
                    )
        except Exception as e:
            self.bot.log("Errors", f"cleanup_indicator_cache Error: {e}")
