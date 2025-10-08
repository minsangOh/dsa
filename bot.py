"""Trading workers and the core automated trading logic.

The module defines background workers that calculate indicators, evaluate entry
conditions, and coordinate order management. The ``TradingBot`` thread
orchestrates portfolio state, execution, and recurring scan cycles while
exposing Qt signals for the UI.
"""

import logging
import math
import multiprocessing
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, InvalidOperation

from PyQt6.QtCore import (
    QObject,
    QRunnable,
    QThread,
    QThreadPool,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)

import config
from bithumb_api import BithumbWrapper
from config import (
    BLOCK_LIST,
    INITIAL_STOP_LOSS_PCT,
    MANUAL_EXCLUDE_LIST,
    MIN_ORDER_VALUE_KRW,
    ORDER_STALE_SECONDS,
    POSITION_TIME_EXIT_SECONDS,
    SELL_CYCLE_INTERVAL_SECONDS,
    STOP_LOSS_ATR_MULTIPLIER,
)
from datastore import OrderStore
from strategy import StrategyManager
from utils import calculate_atr, format_symbol_display

loggers = {
    "system": logging.getLogger("system"),
    "buy": logging.getLogger("buy"),
    "sell": logging.getLogger("sell"),
    "trade_fail": logging.getLogger("trade_fail"),
    "errors": logging.getLogger("errors"),
}


@dataclass
class TrailingStopResult:
    ts_pct: Decimal
    min_ticks: int
    lim_cushion: int
    trail_dist: Decimal
    stop: Decimal
    limit: Decimal


def floor_to_tick(value: Decimal, tick: Decimal) -> Decimal:
    """Round ``value`` down to the nearest multiple of ``tick``."""
    if tick <= 0:
        raise ValueError("Tick size must be positive for trailing stop calculations.")
    steps = (value / tick).to_integral_value(rounding=ROUND_FLOOR)
    return steps * tick


def compute_trailing_stop(anchor: Decimal, tick: Decimal) -> TrailingStopResult:
    """Compute trailing-stop and limit levels using the configured band rules."""
    anchor_dec = Decimal(str(anchor))
    tick_dec = Decimal(str(tick))
    if anchor_dec <= 0 or tick_dec <= 0:
        raise ValueError("Anchor price and tick size must be positive.")

    bounds = [Decimal(str(b)) for b in config.TRAILING_BOUNDS]
    idx = len(bounds) - 1
    for i, bound in enumerate(bounds):
        if anchor_dec <= bound:
            idx = i
            break

    ts_pct = Decimal(str(config.TRAILING_TS_PCT[idx]))
    min_ticks = config.TRAILING_MIN_TICKS[idx]
    lim_cushion = config.TRAILING_LIM_CUSHION[idx]

    pct_component = Decimal(math.ceil(float(anchor_dec * ts_pct)))
    min_tick_component = tick_dec * Decimal(min_ticks)
    trail_dist = (
        pct_component if pct_component >= min_tick_component else min_tick_component
    )

    raw_stop = anchor_dec - trail_dist
    stop = floor_to_tick(raw_stop, tick_dec)
    limit = stop - (tick_dec * Decimal(lim_cushion))
    if limit < Decimal("0"):
        limit = Decimal("0")

    return TrailingStopResult(
        ts_pct=ts_pct,
        min_ticks=min_ticks,
        lim_cushion=lim_cushion,
        trail_dist=trail_dist,
        stop=stop,
        limit=limit,
    )


def safe_decimal(value, default=Decimal("0")) -> Decimal:
    """Convert arbitrary numeric-like input to ``Decimal`` with graceful fallback."""

    if isinstance(value, Decimal):
        return value
    if value is None:
        return default

    try:
        candidate = Decimal(str(value))
    except (InvalidOperation, ArithmeticError, TypeError, ValueError):
        return default

    if candidate.is_nan() or candidate.is_infinite():
        return default
    return candidate


class CancelOrderWorker(QRunnable):
    """Background task that cancels stale orders and verifies the outcome."""

    def __init__(self, bot_instance, api_key, secret_key, order_id, order_data):
        """Keep references needed to attempt cancellation for a given order."""
        super().__init__()
        self.bot = bot_instance
        self.api = BithumbWrapper(api_key, secret_key)
        self.order_id = order_id
        self.data = order_data

    @pyqtSlot()
    def run(self):
        """Issue the cancel request and poll for confirmation with retries."""
        bot = self.bot
        order_id = self.order_id
        data = self.data

        bot.log(
            "System",
            f"[{data['symbol']}] Order ({order_id}) exceeds 15 seconds, automatically canceled.",
        )

        cancel_res = self.api.cancel_order(order_id, data["symbol"], data.get("side"))
        if not cancel_res:
            bot.log(
                "Trade Fail",
                f"[{data['symbol']}] Failed to call order({order_id}) cancellation API.",
            )
            with bot.state_lock:
                if order_id in bot.pending_orders:
                    bot.pending_orders[order_id].pop("cancellation_attempted", None)
            return

        for _ in range(5):
            time.sleep(0.5)
            details = self.api.get_order_details(order_id, data["symbol"])

            if not details:
                continue

            if details.get("status") == "canceled":
                bot.log(
                    "System",
                    f"[{data['symbol']}] Order ({order_id}) cancellation confirmed.",
                )
                with bot.state_lock:
                    if order_id in bot.pending_orders:
                        if bot.pending_orders[order_id]["side"] == "buy":
                            cost = bot.pending_orders[order_id].get(
                                "cost", Decimal("0")
                            )
                            bot.krw_reserved -= cost
                        del bot.pending_orders[order_id]
                        bot.db.update_order_status(order_id, "canceled")
                return

            if details.get("status") == "closed":
                bot.log(
                    "System",
                    f"[{data['symbol']}] Confirmation of order cancellation attempt ({order_id}).",
                )
                return

        bot.log(
            "System",
            f"[{data['symbol']}] After canceling an order ({order_id}), the status check will be delayed. It will be checked again during the next regular cycle.",
        )
        with bot.state_lock:
            if order_id in bot.pending_orders:
                bot.pending_orders[order_id].pop("cancellation_attempted", None)


class PortfolioUpdateWorker(QRunnable):
    """Run portfolio maintenance in the background to keep the UI thread responsive."""

    def __init__(self, bot_instance: "TradingBot") -> None:
        super().__init__()
        self.bot = bot_instance

    @pyqtSlot()
    def run(self):
        try:
            if self.bot.is_running:
                self.bot.perform_order_check_cycle()
                self.bot.refresh_wallet_state()
                self.bot.update_ui_portfolio()
        except Exception as e:
            self.bot.log("Errors", f"PortfolioUpdateWorker Error: {e}")
        finally:
            self.bot.mark_portfolio_update_finished()


class TradingBot(QThread):
    """Primary thread that coordinates scanning, orders, and portfolio state."""

    log_signal = pyqtSignal(str, str)
    portfolio_signal = pyqtSignal(list)
    status_signal = pyqtSignal(str)
    status_update_signal = pyqtSignal(str)

    def __init__(self, api_key, secret_key, fixed_buy_amount_str):
        """Initialise API clients, runtime state, and the supporting thread pools."""
        super().__init__()
        self.api = BithumbWrapper(api_key, secret_key)
        self.db = OrderStore()
        self.is_running = True
        self.kill_switch_activated = False
        self.state_lock = threading.Lock()

        self.portfolio = {}
        self.pending_orders = {}
        self.krw_balance = Decimal("0")
        self.krw_reserved = Decimal("0")
        self.tick_sizes = {}
        self.trade_history = deque(maxlen=100)
        self.cycle_count = 0
        self.tradable_universe = []
        self.starting_equity = Decimal("0")
        self.start_timestamp = None
        self._last_portfolio_payload = None
        self.strategy_manager = None

        try:
            fixed_amount = Decimal(fixed_buy_amount_str)
            self.fixed_buy_amount = (
                fixed_amount if fixed_amount >= MIN_ORDER_VALUE_KRW else Decimal("0")
            )
        except (InvalidOperation, ValueError):
            self.fixed_buy_amount = Decimal("0")

        self.worker_thread_pool = QThreadPool()
        self.worker_thread_pool.setMaxThreadCount(2)  # For portfolio and cancels
        self._portfolio_update_lock = threading.Lock()
        self._portfolio_update_active = False

    def log(self, tab, message):
        """Write a message to the appropriate logger and forward it to the UI."""
        logger = loggers.get(tab.lower().replace(" ", "_"), loggers["system"])
        logger.info(message)
        self.log_signal.emit(tab, message)

    def stop(self):
        """Terminate timers, drain workers, and end the Qt thread cleanly."""
        self.log("System", "Stop signal received. Signals thread termination.")
        self.is_running = False
        if self.strategy_manager:
            self.strategy_manager.stop()
        self.worker_thread_pool.clear()
        self.worker_thread_pool.waitForDone()
        self.quit()

    def activate_kill_switch(self):
        """Cancel open orders, liquidate holdings, and stop the trading loop."""
        self.kill_switch_activated = True
        self.is_running = False
        self.log(
            "System", "Kill switch activated! Cancel all orders and liquidate positions."
        )
        self.execute_kill_switch()
        self.stop()

    def run(self):
        """Configure recurring timers and start the bot event loop."""
        self.log("System", f"TradingBot thread ID: {threading.get_ident()}")

        # --- Strategy Manager Setup ---
        self.strategy_manager = StrategyManager(self)
        self.strategy_manager.buy_signal_found.connect(self.handle_buy_check_result)
        self.strategy_manager.status_update_signal.connect(self.status_update_signal)

        if not all(
            len(lst) == len(config.TRAILING_BOUNDS)
            for lst in [
                config.TRAILING_TS_PCT,
                config.TRAILING_MIN_TICKS,
                config.TRAILING_LIM_CUSHION,
            ]
        ):
            self.log(
                "Errors", "Trailing stop configuration lists have inconsistent lengths."
            )
            self.stop()
            return

        self.log("System", "Start the trading bot.")
        self.load_initial_data()

        if not self.is_running:
            return

        # --- Timers Setup ---
        self.sell_timer = QTimer()
        self.sell_timer.timeout.connect(self.perform_sell_cycle)
        self.sell_timer.start(SELL_CYCLE_INTERVAL_SECONDS * 1000)

        # Start the strategy manager's own internal timers
        self.strategy_manager.start_timers()

        self.exec()

        if not self.kill_switch_activated:
            self.log("System", "The trading bot has been safely stopped.")
            self.status_signal.emit("Stopped")

    def load_initial_data(self):
        """Fetch balances and market metadata required before the first cycle."""
        self.krw_balance = self.api.get_balance("KRW")
        self.log("System", f"Initial KRW balance : {self.krw_balance:,.0f} KRW")
        initial_assets = self.krw_balance
        markets = self.api.get_all_market_details()
        if not markets:
            self.log("Errors", "Failed to retrieve market information.")
            self.starting_equity = initial_assets
            self.start_timestamp = time.time()
            return self.stop()
        for market in markets.values():
            if market["quote"] == "KRW":
                symbol = market["symbol"].replace("/", "_")
                self.tick_sizes[symbol] = self.api.get_tick_size(symbol)
        self.log(
            "System", f"Caching the entire coin tick size : {len(self.tick_sizes)}ea"
        )

        balances = self.api.get_balance_all()
        assets_to_load = []
        for coin, data in balances.items():
            symbol = f"{coin}_KRW"
            if coin == "KRW" or symbol not in self.tick_sizes:
                continue

            amount = data.get("available", Decimal("0"))
            if amount <= 0:
                continue

            price = self.api.get_current_price(symbol)
            if not price:
                continue

            min_order_value = self.api.get_min_order_value(symbol)
            if (amount * price) <= min_order_value:
                continue

            extras = self.get_new_position_extras(symbol, price)
            assets_to_load.append(
                {
                    "symbol": symbol,
                    "amount": amount,
                    "price": price,
                    "extras": extras,
                }
            )

        with self.state_lock:
            for asset in assets_to_load:
                symbol = asset["symbol"]
                self.portfolio[symbol] = {
                    "amount": asset["amount"],
                    "entry_price": asset["price"],
                    **asset["extras"],
                }
                self.log(
                    "System",
                    f"Added existing asset '{format_symbol_display(symbol)}' to your portfolio.",
                )
                initial_assets += asset["amount"] * asset["price"]

        self.starting_equity = initial_assets
        self.start_timestamp = time.time()

    def refresh_wallet_state(self):
        """Synchronise cash balances and holdings with the exchange wallet."""
        balances = self.api.get_balance_all()
        if not balances:
            return

        all_tickers = self.api.get_all_ticker_data()
        all_markets = self.api.get_all_market_details()

        if not all_tickers or not all_markets:
            self.log(
                "Errors", "Failed to fetch ticker or market data for wallet refresh."
            )
            return

        with self.state_lock:
            current_symbols = set(self.portfolio.keys())

        krw_info = balances.get("KRW")
        if krw_info and krw_info.get("available") is not None:
            available = krw_info.get("available")
            krw_available = (
                available
                if isinstance(available, Decimal)
                else Decimal(str(available))
            )
        else:
            krw_available = self.krw_balance

        updated_assets: list[tuple[str, Decimal, Decimal, dict | None]] = []
        for coin, data in balances.items():
            if coin == "KRW":
                continue

            symbol = f"{coin}_KRW"
            amount_val = data.get("available", Decimal("0"))
            amount = (
                amount_val
                if isinstance(amount_val, Decimal)
                else Decimal(str(amount_val))
            )

            if amount <= 0:
                updated_assets.append((symbol, Decimal("0"), Decimal("0"), None))
                continue

            ticker_info = all_tickers.get(coin)
            price_raw = ticker_info.get("closing_price") if ticker_info else None
            price = (
                price_raw
                if isinstance(price_raw, Decimal)
                else Decimal(str(price_raw)) if price_raw else Decimal("0")
            )

            market_info = all_markets.get(symbol.replace("_", "/"))
            min_order_val = Decimal("5000")
            if (
                market_info
                and market_info.get("limits", {}).get("cost", {}).get("min") is not None
            ):
                min_order_val = Decimal(str(market_info["limits"]["cost"]["min"]))

            if price <= 0 or (amount * price) < min_order_val:
                updated_assets.append((symbol, Decimal("0"), Decimal("0"), None))
                continue

            extras = None
            if symbol not in current_symbols:
                extras = self.get_new_position_extras(symbol, price)
                extras["amount"] = amount
                extras["entry_price"] = price
                extras["last_price"] = price

            updated_assets.append((symbol, amount, price, extras))

        with self.state_lock:
            self.krw_balance = krw_available

            updated_symbols = set()
            for symbol, amount, price, extras in updated_assets:
                if amount <= 0:
                    if symbol in self.portfolio:
                        del self.portfolio[symbol]
                    continue

                existing = self.portfolio.get(symbol)
                if existing:
                    existing["amount"] = amount
                    existing["last_price"] = price
                    if existing.get("entry_price", Decimal("0")) <= 0:
                        existing["entry_price"] = price
                else:
                    new_entry = extras or self.get_new_position_extras(symbol, price)
                    new_entry["amount"] = amount
                    new_entry["entry_price"] = price
                    new_entry["last_price"] = price
                    self.portfolio[symbol] = new_entry

                updated_symbols.add(symbol)

            stale_symbols = [
                sym for sym in list(self.portfolio.keys()) if sym not in updated_symbols
            ]
            for sym in stale_symbols:
                if sym in self.portfolio:
                    del self.portfolio[sym]

    def check_buy_constraints(self, symbol, current_price, orderbook=None):
        """Apply additional market constraints such as spread thresholds."""
        orderbook_data = orderbook or self.api.get_orderbook(symbol)
        if (
            not orderbook_data
            or not orderbook_data.get("asks")
            or not orderbook_data.get("bids")
        ):
            return False
        spread = (
            orderbook_data["asks"][0]["price"] - orderbook_data["bids"][0]["price"]
        )
        tick = self.tick_sizes.get(symbol, Decimal("1"))
        if tick <= 0:
            tick = Decimal("1")
        if (spread / tick) >= 4:
            return False
        return True

    @pyqtSlot(dict)
    def handle_buy_check_result(self, buy_data):
        """Validate buy worker output and create pending buy orders when allowed."""
        try:
            if not self.is_running:
                return

            symbol = buy_data["symbol"]
            price = buy_data["ask_price"]
            amount = buy_data["amount"]
            strategies = buy_data.get("strategies", [])
            display_symbol = format_symbol_display(symbol)

            with self.state_lock:
                if len(self.portfolio) >= config.MAX_POSITIONS:
                    return

                cost = price * amount
                if self.krw_balance - self.krw_reserved < cost:
                    return
                if any(
                    o["symbol"] == symbol and o["side"] == "buy"
                    for o in self.pending_orders.values()
                ):
                    return

            self.log(
                "Buy",
                f"[{display_symbol}]" + (f" - {', '.join(strategies)}" if strategies else ""),
            )

            for attempt in range(1, 4):
                precise_price = self.api.price_to_precision(symbol, price)
                if not precise_price:
                    self.log(
                        "Trade Fail",
                        f"[{display_symbol}] Failure to refine order pricing ({attempt}/3).",
                    )
                    if attempt < 3:
                        time.sleep(0.5)
                        continue
                    return

                res = self.api.place_limit_buy_order(symbol, precise_price, amount)
                if res and res.get("id"):
                    with self.state_lock:
                        self.krw_reserved += cost
                        self.pending_orders[res["id"]] = {
                            "symbol": symbol,
                            "side": "buy",
                            "amount": amount,
                            "price": precise_price,
                            "filled": Decimal("0"),
                            "cost": cost,
                            "created_at": time.time(),
                            "strategies": list(strategies),
                        }
                    self.db.add_order(
                        res["id"],
                        symbol,
                        "buy",
                        float(precise_price),
                        float(amount),
                        "pending",
                    )
                    self.log(
                        "Buy",
                        f"Buy - [{display_symbol}] : Order ID [{res['id']}]",
                    )
                    return

                self.log(
                    "Trade Fail",
                    f"[{display_symbol}] Buy order failed ({attempt}/3) : {res}",
                )
                if attempt < 3:
                    time.sleep(0.5)

            self.log(
                "Trade Fail",
                f"[{display_symbol}] Buy order failed: Maximum retries exceeded",
            )
            return
        except Exception as e:
            self.log("Errors", f"handle_buy_check_result Error: {e}")

    @pyqtSlot()
    def perform_sell_cycle(self):
        """Iterate positions and evaluate whether any sell rules are met."""
        try:
            if not self.is_running:
                return
            with self.state_lock:
                portfolio_copy = list(self.portfolio.items())
            if portfolio_copy:
                self.process_sell_logic(portfolio_copy)
        except Exception as e:
            self.log("Errors", f"perform_sell_cycle Error: {e}")

    def process_sell_logic(self, portfolio_copy):
        """Compare prices against exit criteria and decide which symbols to sell."""
        to_sell = []
        for symbol, _ in portfolio_copy:
            if symbol in BLOCK_LIST:
                continue

            with self.state_lock:
                current_position = self.portfolio.get(symbol)
                has_pending_sell = any(
                    order["symbol"] == symbol and order["side"] == "sell"
                    for order in self.pending_orders.values()
                )
                amount = current_position["amount"] if current_position else None

            if amount is None or has_pending_sell:
                continue

            price = self.api.get_current_price(symbol)
            if (
                not price
                or amount is None
                or (price * amount) < self.api.get_min_order_value(symbol)
            ):
                continue

            self.update_trailing_stop(symbol, price)

            with self.state_lock:
                current_data = self.portfolio.get(symbol)
                if not current_data:
                    continue
                stop_loss = current_data["stop_loss"]
                trailing_stop = current_data["trailing_stop"]
                entry_timestamp = current_data.get("entry_timestamp")
                entry_price = current_data.get("entry_price")

            if (
                entry_timestamp
                and time.time() - entry_timestamp >= POSITION_TIME_EXIT_SECONDS
                and stop_loss > 0
                and entry_price is not None
            ):
                midpoint = entry_price + ((entry_price - stop_loss) / Decimal("2"))
                if price < midpoint:
                    to_sell.append({"symbol": symbol, "reason": "Time over"})
                    continue

            if stop_loss > 0 and price <= stop_loss:
                to_sell.append({"symbol": symbol, "reason": "Stop loss"})
            elif trailing_stop > 0 and price <= trailing_stop:
                to_sell.append({"symbol": symbol, "reason": "Tailing stop"})

        for item in to_sell:
            self.execute_sell(item["symbol"], item["reason"])

    def update_trailing_stop(self, symbol, current_price):
        """Maintain trailing-stop values so profits are protected as price rises."""
        tick_size = self.tick_sizes.get(symbol)
        if not tick_size:
            return
        with self.state_lock:
            if symbol not in self.portfolio:
                return
            data = self.portfolio[symbol]
            if "trailing_limit" not in data:
                data["trailing_limit"] = Decimal("0")
            if "trailing_meta" not in data:
                data["trailing_meta"] = None

            if not data["trailing_active"]:
                price_gain_ticks = (current_price - data["entry_price"]) / tick_size
                if price_gain_ticks >= 7:
                    data["trailing_active"] = True
                    data["trailing_high"] = current_price
                    try:
                        result = compute_trailing_stop(current_price, tick_size)
                    except ValueError:
                        return
                    data["trailing_stop"] = result.stop
                    data["trailing_limit"] = result.limit
                    data["trailing_meta"] = result
                return

            if current_price > data.get("trailing_high", current_price):
                data["trailing_high"] = current_price
                try:
                    result = compute_trailing_stop(data["trailing_high"], tick_size)
                except ValueError:
                    return
                if result.stop > data.get("trailing_stop", Decimal("0")):
                    data["trailing_stop"] = result.stop
                data["trailing_limit"] = result.limit
                data["trailing_meta"] = result

    def execute_sell(self, symbol, reason):
        """Submit a sell order for the given symbol using current order book data."""
        display_symbol = format_symbol_display(symbol)
        with self.state_lock:
            if symbol not in self.portfolio:
                return
            if any(
                order["symbol"] == symbol and order["side"] == "sell"
                for order in self.pending_orders.values()
            ):
                self.log(
                    "Sell",
                    f"[{display_symbol}] Existing sell order is being processed. Duplicate sell request ({reason}) will be skipped.",
                )
                return
            amount = self.portfolio[symbol]["amount"]

        for attempt in range(1, 4):
            orderbook = self.api.get_orderbook(symbol)
            if not orderbook or not orderbook.get("bids"):
                self.log(
                    "Trade Fail",
                    f"[{display_symbol}] Failed to check sell order book ({attempt}/3)",
                )
                if attempt < 3:
                    time.sleep(0.5)
                    continue
                return
            price = orderbook["bids"][0]["price"]

            precise_price = self.api.price_to_precision(symbol, price)
            if not precise_price:
                self.log(
                    "Trade Fail",
                    f"[{display_symbol}] Failure to refine the selling price ({attempt}/3)",
                )
                if attempt < 3:
                    time.sleep(0.5)
                    continue
                return

            precise_amount = self.api.amount_to_precision(symbol, amount)
            if not precise_amount:
                self.log(
                    "Trade Fail",
                    f"[{display_symbol}] Failure to refine the selling amount ({attempt}/3)",
                )
                if attempt < 3:
                    time.sleep(0.5)
                    continue
                return

            min_order_val = self.api.get_min_order_value(symbol)
            if (precise_price * precise_amount) < min_order_val:
                self.log(
                    "Trade Fail",
                    f"[{display_symbol}] Order value below minimum ({min_order_val} KRW) after precision adjustment. Skipping sell.",
                )
                return

            res = self.api.place_limit_sell_order(
                symbol, precise_price, precise_amount
            )
            if res and res.get("id"):
                with self.state_lock:
                    cost = precise_price * precise_amount
                    self.pending_orders[res["id"]] = {
                        "symbol": symbol,
                        "side": "sell",
                        "amount": precise_amount,
                        "price": precise_price,
                        "filled": Decimal("0"),
                        "cost": cost,
                        "created_at": time.time(),
                    }
                self.db.add_order(
                    res["id"],
                    symbol,
                    "sell",
                    float(precise_price),
                    float(precise_amount),
                    "pending",
                )
                self.log(
                    "Sell",
                    f"Sell({reason}) - [{display_symbol}] ID: {res['id']}",
                )
                return

            self.log(
                "Trade Fail",
                f"[{display_symbol}] Sell ​​order failed ({attempt}/3) : {res}",
            )
            if attempt < 3:
                time.sleep(0.5)

        self.log(
            "Trade Fail",
            f"[{display_symbol}] Sell ​​order failed: maximum retries exceeded",
        )
        return

    @pyqtSlot()
    def perform_order_check_cycle(self):
        """Poll outstanding orders and schedule follow-up actions when needed."""
        try:
            if not self.is_running:
                return
            with self.state_lock:
                pending_copy = list(self.pending_orders.items())

            if not pending_copy:
                return

            for order_id, data in pending_copy:
                if "cancellation_attempted" not in data and time.time() - data.get(
                    "created_at", 0
                ) > ORDER_STALE_SECONDS:
                    with self.state_lock:
                        if order_id in self.pending_orders:
                            self.pending_orders[order_id][
                                "cancellation_attempted"
                            ] = True

                    worker = CancelOrderWorker(
                        self, self.api.api_key, self.api.secret_key, order_id, data
                    )
                    self.worker_thread_pool.start(worker)
                else:
                    self.check_order_filled(order_id, data)
        except Exception as e:
            self.log("Errors", f"perform_order_check_cycle Error: {e}")

    def check_order_filled(self, order_id, order_data):
        """Inspect order details to update fills, cancellations, or failures."""
        details = None
        for attempt in range(1, 4):
            details = self.api.get_order_details(order_id, order_data["symbol"])
            if details:
                break
            self.log(
                "Trade Fail",
                f"Order({order_id}) Failed to retrieve detailed information ({attempt}/3)",
            )
            if attempt < 3:
                time.sleep(0.5)
        if not details:
            self.log(
                "Trade Fail",
                f"Order({order_id}) Failed to retrieve detailed information : Maximum retries exceeded.",
            )
            return

        if details.get("status") == "not_found":
            self._handle_missing_order_details(order_id, order_data)
            return

        filled = details.get("filled", Decimal("0"))
        is_new_buy_fill = False
        symbol = order_data["symbol"]
        avg_price_for_extras = None
        with self.state_lock:
            if order_id in self.pending_orders:
                order = self.pending_orders[order_id]
                newly_filled = filled - order["filled"]
                if (
                    newly_filled > 0
                    and order["side"] == "buy"
                    and symbol not in self.portfolio
                ):
                    is_new_buy_fill = True
                    avg_price_for_extras = details.get("average", order["price"])

        extras = None
        if is_new_buy_fill:
            extras = self.get_new_position_extras(symbol, avg_price_for_extras)

        with self.state_lock:
            if order_id not in self.pending_orders:
                return

            order = self.pending_orders[order_id]
            prev_filled = order["filled"]
            newly_filled = filled - prev_filled

            if newly_filled > 0:
                avg_price = details.get("average", order["price"])
                if order["side"] == "buy":
                    fill_cost = avg_price * newly_filled
                    self.krw_reserved -= fill_cost
                    if order_id in self.pending_orders:
                        self.pending_orders[order_id]["cost"] = max(
                            Decimal("0"),
                            self.pending_orders[order_id]["cost"] - fill_cost,
                        )

                    self.handle_buy_fill(
                        symbol,
                        newly_filled,
                        avg_price,
                        order.get("strategies"),
                        extras=extras,
                    )
                else:
                    self.handle_sell_fill(symbol, newly_filled, avg_price)
                self.pending_orders[order_id]["filled"] = filled

            if details.get("status") in ["closed", "canceled"]:
                order = self.pending_orders[order_id]
                if (
                    details.get("status") == "closed"
                    and order["filled"] < order["amount"]
                ):
                    unaccounted_amount = order["amount"] - order["filled"]
                    avg_price = details.get("average", order["price"])
                    self.log(
                        "System",
                        f"[{format_symbol_display(symbol)}] Order {order_id} is 'closed' with unaccounted fills. Reconciling {unaccounted_amount} of {order['side']} order.",
                    )
                    if order["side"] == "buy":
                        fill_cost = avg_price * unaccounted_amount
                        self.krw_reserved -= fill_cost
                        if order_id in self.pending_orders:
                            self.pending_orders[order_id]["cost"] = max(
                                Decimal("0"),
                                self.pending_orders[order_id]["cost"] - fill_cost,
                            )
                        self.handle_buy_fill(
                            symbol,
                            unaccounted_amount,
                            avg_price,
                            order.get("strategies"),
                            extras=extras,
                        )
                    else:
                        self.handle_sell_fill(symbol, unaccounted_amount, avg_price)
                    self.pending_orders[order_id]["filled"] = order["amount"]

                if order_data["side"] == "buy":
                    self.krw_reserved -= self.pending_orders[order_id]["cost"]
                del self.pending_orders[order_id]
                self.db.update_order_status(order_id, details.get("status"))

    def _handle_missing_order_details(self, order_id, order_data):
        """Handle orders that disappear from the exchange without a status.

        Args:
            order_id: Exchange order identifier that could not be queried.
            order_data: Locally cached order metadata.
        """
        symbol = order_data["symbol"]
        with self.state_lock:
            cached_order = self.pending_orders.get(order_id)
            if not cached_order:
                return
            cached_copy = dict(cached_order)

        side = cached_copy.get("side")
        if side != "sell":
            self.log(
                "System",
                f"[{format_symbol_display(symbol)}] Order {order_id} missing from exchange. Awaiting wallet sync for reconciliation.",
            )
            with self.state_lock:
                cached = self.pending_orders.pop(order_id, None)
                if cached and cached.get("side") == "buy":
                    self.krw_reserved -= cached.get("cost", Decimal("0"))
                    if self.krw_reserved < 0:
                        self.krw_reserved = Decimal("0")
            self.db.update_order_status(order_id, "closed")
            return

        filled_amount = cached_copy.get("filled", Decimal("0"))
        total_amount = cached_copy.get("amount", Decimal("0"))
        remaining_amount = max(Decimal("0"), total_amount - filled_amount)
        if remaining_amount <= 0:
            with self.state_lock:
                self.pending_orders.pop(order_id, None)
            self.db.update_order_status(order_id, "closed")
            return

        price = cached_copy.get("price", Decimal("0"))
        if price <= 0:
            fallback_price = self.api.get_current_price(symbol)
            price = fallback_price if fallback_price else Decimal("0")

        if price <= 0:
            self.log(
                "Errors",
                f"[{format_symbol_display(symbol)}] Unable to infer fill price for missing order {order_id}.",
            )
            return

        display_symbol = format_symbol_display(symbol)
        self.log(
            "Sell",
            f"[{display_symbol}] Order {order_id} not found on exchange. Assuming remaining quantity filled at limit price.",
        )
        self.handle_sell_fill(symbol, remaining_amount, price)

        with self.state_lock:
            self.pending_orders.pop(order_id, None)
        self.db.update_order_status(order_id, "closed")

    def handle_buy_fill(
        self, symbol, filled_amount, price, strategies=None, extras=None
    ):
        """Update balances, portfolio state, and history for filled buy orders."""
        self.krw_balance -= price * filled_amount
        display_symbol = format_symbol_display(symbol)
        if symbol not in self.portfolio:
            if extras is None:
                self.log(
                    "Errors",
                    f"handle_buy_fill for new symbol {symbol} called without pre-fetched extras.",
                )
                extras = self.get_new_position_extras(symbol, price)
            extras["last_price"] = price
            self.portfolio[symbol] = {
                "amount": filled_amount,
                "entry_price": price,
                **extras,
            }
        else:
            old_val = (
                self.portfolio[symbol]["entry_price"]
                * self.portfolio[symbol]["amount"]
            )
            new_val = price * filled_amount
            total_amount = self.portfolio[symbol]["amount"] + filled_amount
            self.portfolio[symbol]["amount"] = total_amount
            self.portfolio[symbol]["entry_price"] = (
                (old_val + new_val) / total_amount if total_amount > 0 else Decimal("0")
            )
            self.portfolio[symbol]["entry_timestamp"] = time.time()
            self.portfolio[symbol]["last_price"] = price
        self.log(
            "Buy",
            f"[{display_symbol}] Quantity: {filled_amount:.8f}, Average price: {price:.0f}",
        )

    def handle_sell_fill(self, symbol, filled_amount, price):
        """Adjust portfolio quantities and cash after a sell fill."""
        self.krw_balance += price * filled_amount
        display_symbol = format_symbol_display(symbol)
        if symbol in self.portfolio:
            self.portfolio[symbol]["amount"] -= filled_amount
            self.portfolio[symbol]["last_price"] = price

            market_info = self.api.get_market_info(symbol)
            min_amount = Decimal("0.00000001")
            if (
                market_info
                and market_info.get("limits", {}).get("amount", {}).get("min")
                is not None
            ):
                min_amount = Decimal(str(market_info["limits"]["amount"]["min"]))
            if self.portfolio[symbol]["amount"] <= min_amount:
                del self.portfolio[symbol]
                self.trade_history.append((symbol, time.time(), "sell"))
        self.log(
            "Sell",
            f"[{display_symbol}] Sell order executed! Quantity: {filled_amount:.8f}, Average price: {price:.0f}",
        )

    def get_new_position_extras(self, symbol, entry_price):
        """Calculate initial stop-loss and trailing metadata for new positions."""
        ohlcv = self.api.get_candlestick(symbol, "1h")
        stop_loss = entry_price * Decimal(INITIAL_STOP_LOSS_PCT)
        if ohlcv is not None and len(ohlcv) > 14:
            atr = calculate_atr(ohlcv, period=14)
            if atr:
                atr_stop_loss = entry_price - (
                    Decimal(str(atr)) * Decimal(str(STOP_LOSS_ATR_MULTIPLIER))
                )
                stop_loss = min(stop_loss, atr_stop_loss)
        return {
            "stop_loss": stop_loss,
            "trailing_high": entry_price,
            "trailing_stop": Decimal("0"),
            "trailing_active": False,
            "trailing_limit": Decimal("0"),
            "trailing_meta": None,
            "entry_timestamp": time.time(),
            "last_price": entry_price,
        }

    def get_total_assets_internal(self):
        """Aggregate cash, reserved funds, and marked holdings into total equity."""
        value = self.krw_balance + self.krw_reserved
        for symbol, data in self.portfolio.items():
            cache_entry = self.strategy_manager.indicator_cache.get(symbol)
            raw_price = cache_entry["data"].get("close") if cache_entry else None
            price = safe_decimal(raw_price)
            if price > 0:
                value += data["amount"] * price
        return value

    def get_btc_value_internal(self):
        """Return the evaluated value of the BTC position if one exists."""
        if "BTC_KRW" in self.portfolio:
            cache_entry = self.strategy_manager.indicator_cache.get("BTC_KRW")
            raw_price = cache_entry["data"].get("close") if cache_entry else None
            price = safe_decimal(raw_price)
            if price > 0:
                return self.portfolio["BTC_KRW"]["amount"] * price
        return Decimal("0")

    @pyqtSlot()
    def request_portfolio_update(self):
        """Schedule a portfolio refresh so the UI can display up-to-date data."""
        try:
            if not self.is_running:
                return
            if not self._try_begin_portfolio_update():
                return
            worker = PortfolioUpdateWorker(self)
            self.worker_thread_pool.start(worker)
        except Exception as e:
            self.mark_portfolio_update_finished()
            self.log("Errors", f"request_portfolio_update Error: {e}")

    def update_ui_portfolio(self):
        """Prepare UI-friendly portfolio rows and emit them to the table model."""
        with self.state_lock:
            portfolio_copy = list(self.portfolio.items())
            cache_copy = self.strategy_manager.indicator_cache.copy()
            cash_balance = self.krw_balance
            reserved_balance = self.krw_reserved
            starting_equity = (
                self.starting_equity if self.starting_equity else Decimal("0")
            )

        ui_data = []
        FEE = Decimal("0.0004")
        total_buy_val = Decimal("0")
        total_eval_val = Decimal("0")
        for symbol, data in portfolio_copy:
            cache_entry = cache_copy.get(symbol)
            price = safe_decimal(data.get("last_price"), data["entry_price"])
            if price <= 0:
                price = data["entry_price"]

            if cache_entry:
                cache_price = safe_decimal(cache_entry["data"].get("close"))
                if price <= 0 and cache_price > 0:
                    price = cache_price

            buy_val = data["amount"] * data["entry_price"]
            eval_val = data["amount"] * price
            total_buy_val += buy_val
            total_eval_val += eval_val

            total_buy_cost = buy_val * (Decimal("1") + FEE)
            net_sell_value = eval_val * (Decimal("1") - FEE)
            pnl = net_sell_value - total_buy_cost
            roi = (
                (pnl / total_buy_cost) * Decimal("100")
                if total_buy_cost > 0
                else Decimal("0")
            )
            stop_loss_value = safe_decimal(data.get("stop_loss"), Decimal("0"))

            ui_data.append(
                [
                    format_symbol_display(symbol).replace("_", "/"),
                    f"{data['amount']:.8f}",
                    f"{data['entry_price']:,.2f}",
                    f"{buy_val:,.0f}",
                    f"{pnl:,.0f}",
                    f"{float(roi):.2f}%",
                    f"{stop_loss_value:,.2f}",
                    f"{price:,.2f}",
                    f"{eval_val:,.0f}",
                ]
            )

        total_assets = total_eval_val + cash_balance + reserved_balance
        base_equity = starting_equity if starting_equity > 0 else total_assets
        overall_pnl = total_assets - base_equity
        roi_since_start = (
            (overall_pnl / base_equity) * Decimal("100")
            if base_equity > 0
            else Decimal("0")
        )

        summary_row = [
            "TOTAL",
            "",
            "",
            f"{base_equity:,.0f}",
            f"{overall_pnl:,.0f}",
            f"{float(roi_since_start):.2f}% (Since Start)",
            "-",
            f"Cash {cash_balance:,.0f}",
            f"{total_assets:,.0f}",
        ]
        ui_data.append(summary_row)
        payload_signature = tuple(tuple(row) for row in ui_data)
        if payload_signature == self._last_portfolio_payload:
            return

        self._last_portfolio_payload = payload_signature
        self.portfolio_signal.emit(ui_data)

    def _try_begin_portfolio_update(self) -> bool:
        """Attempt to reserve the right to run a portfolio refresh."""
        with self._portfolio_update_lock:
            if self._portfolio_update_active:
                return False
            self._portfolio_update_active = True
            return True

    def mark_portfolio_update_finished(self):
        """Release the reservation so future refresh cycles can run."""
        with self._portfolio_update_lock:
            self._portfolio_update_active = False

    def execute_kill_switch(self):
        """Cancel existing orders, liquidate positions, and reset bot state."""
        self.log("System", "Kill Switch: Attempts to cancel all orders.")
        with self.state_lock:
            pending_copy = list(self.pending_orders.items())
        for order_id, data in pending_copy:
            self.api.cancel_order(order_id, data["symbol"], data.get("side"))

        time.sleep(2)

        self.log("System", "Kill Switch: Sell all positions at market price.")
        with self.state_lock:
            portfolio_copy = list(self.portfolio.items())

        for symbol, data in portfolio_copy:
            if symbol != "BTC_KRW" and data["amount"] > 0:
                display_symbol = format_symbol_display(symbol)
                market_info = self.api.get_market_info(symbol)
                min_amount = None
                if market_info:
                    min_amount = (
                        market_info.get("limits", {}).get("amount", {}).get("min")
                    )
                if min_amount is not None and data["amount"] < Decimal(
                    str(min_amount)
                ):
                    self.log(
                        "System",
                        f"Kill Switch: Skip sell if [{display_symbol}] quantity ({data['amount']}) is less than the minimum order quantity.",
                    )
                    continue

                self.api.place_market_sell_order(symbol, data["amount"])
                self.log(
                    "Sell",
                    f"Kill Switch: Sell at market price [{display_symbol}] {data['amount']}",
                )

        self.log("System", "Kill Switch: Clears internal state.")
        with self.state_lock:
            self.pending_orders.clear()
            self.portfolio.clear()
            self.krw_reserved = Decimal("0")
            self._last_portfolio_payload = None

        log_id = f"kill_switch_{int(time.time())}"
        self.db.add_order(
            order_id=log_id,
            symbol="SYSTEM",
            side="AUDIT",
            price=0,
            amount=0,
            status="force_sell",
        )

        self.log("System", "The kill switch procedure has been completed.")