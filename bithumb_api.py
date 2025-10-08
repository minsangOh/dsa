"""CCXT-based wrapper and helpers for interacting with the Bithumb exchange."""

import math
import time
import threading
from decimal import Decimal
from functools import wraps

import ccxt
import pandas as pd
from ccxt.base.decimal_to_precision import DECIMAL_PLACES, SIGNIFICANT_DIGITS


def create_bithumb_client(api_key, secret_key):
    """Create an independent ``ccxt.bithumb`` client using the given keys."""
    try:
        client = ccxt.bithumb({
            'apiKey': api_key,
            'secret': secret_key,
            'enableRateLimit': True,
            'timeout': 10000,
        })
        client.load_markets()
        return client
    except ccxt.BaseError as e:
        print(f"Failed to initialize CCXT API key: {e}")
        return None


def api_call_handler(func):
    """Decorator that retries transient CCXT errors with exponential backoff."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = 3
        base_delay = 1
        for attempt in range(max_retries):
            try:
                return func(self, *args, **kwargs)
            except (ccxt.DDoSProtection, ccxt.RateLimitExceeded,
                    ccxt.RequestTimeout, ccxt.NetworkError) as e:
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2**attempt)
                    print(
                        f"API call failed: {type(e).__name__}. Retrying in {sleep_time}s..."
                    )
                    time.sleep(sleep_time)
                else:
                    print(f"API call failed after {max_retries} attempts: {e}")
                    return None
            except (ccxt.InvalidOrder, ccxt.InsufficientFunds) as e:
                print(f"API call failed with non-retryable error: {e}")
                return None
            except ccxt.BaseError as e:
                print(f"An unexpected CCXT error occurred: {e}")
                return None

    return wrapper


class BithumbWrapper:
    """Thread-safe wrapper around CCXT's Bithumb client with convenience APIs."""

    def __init__(self, api_key, secret_key):
        """Persist API credentials and set up thread-local storage."""
        self.api_key = api_key
        self.secret_key = secret_key
        self._thread_local = threading.local()

    def get_client(self):
        """Return the per-thread CCXT client, creating one if necessary."""
        client = getattr(self._thread_local, 'client', None)
        if client is None:
            new_client = create_bithumb_client(self.api_key, self.secret_key)
            if new_client:
                self._thread_local.client = new_client
            return new_client
        return client

    @api_call_handler
    def get_market_info(self, symbol):
        """Fetch market metadata for the provided symbol."""
        client = self.get_client()
        if not client: return None
        ccxt_symbol = symbol.replace('_', '/')
        return client.markets.get(ccxt_symbol)

    def get_tick_size(self, symbol):
        """Return the price increment for the symbol as a ``Decimal``."""
        market = self.get_market_info(symbol)
        if not market:
            return Decimal("1")

        info = market.get('info', {})
        increment = info.get('priceIncrement')
        if increment:
            try:
                value = Decimal(str(increment))
                if value > 0:
                    return value
            except (ArithmeticError, TypeError):
                pass

        precision_val = market.get('precision', {}).get('price')
        client = self.get_client()
        precision_mode = getattr(client, 'precisionMode', None) if client else None

        if precision_val is not None and client:
            if precision_mode == DECIMAL_PLACES:
                try:
                    return Decimal('1') / (Decimal('10') ** int(precision_val))
                except (ArithmeticError, TypeError, ValueError):
                    pass

            if precision_mode == SIGNIFICANT_DIGITS:
                price_ref = info.get('closing_price') or info.get('prev_closing_price')
                if not price_ref:
                    current = self.get_current_price(symbol)
                    price_ref = current if current else Decimal('1')

                try:
                    price_decimal = Decimal(str(price_ref))
                except (ArithmeticError, TypeError):
                    price_decimal = Decimal('1')

                if price_decimal <= 0:
                    price_decimal = Decimal('1')

                price_float = float(price_decimal)
                if price_float <= 0:
                    price_float = 1.0

                exponent = math.floor(math.log10(price_float)) if price_float != 0 else 0
                power = exponent - int(precision_val) + 1
                step = Decimal('10') ** power
                if step > 0:
                    return step

        orderbook = self.get_orderbook(symbol)
        if orderbook:
            prices = []
            for side in ('bids', 'asks'):
                for entry in orderbook.get(side, [])[:10]:
                    price = entry.get('price')
                    if isinstance(price, Decimal):
                        prices.append(price)

            prices = sorted(set(prices))
            diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
            for diff in diffs:
                if diff > 0:
                    return diff

        return Decimal("1")

    @api_call_handler
    def amount_to_precision(self, symbol, quantity):
        """Adjust an order quantity to the exchange precision rules."""
        client = self.get_client()
        if not client: return None
        ccxt_symbol = symbol.replace('_', '/')
        amount_str = client.amount_to_precision(ccxt_symbol, str(quantity))
        return Decimal(amount_str)

    @api_call_handler
    def price_to_precision(self, symbol, price):
        """Adjust an order price to the exchange precision rules."""
        client = self.get_client()
        if not client: return None
        ccxt_symbol = symbol.replace('_', '/')
        price_str = client.price_to_precision(ccxt_symbol, str(price))
        return Decimal(price_str)

    def get_min_order_value(self, symbol):
        """Return the minimum order cost in KRW for the given symbol."""
        market_info = self.get_market_info(symbol)
        if market_info and market_info.get('limits', {}).get('cost',
                                                             {}).get('min') is not None:
            return Decimal(str(market_info['limits']['cost']['min']))
        return Decimal('5000')

    @api_call_handler
    def get_all_ticker_data(self):
        """Fetch price and quote-volume data for every KRW market."""
        client = self.get_client()
        if not client: return None
        all_tickers = client.fetch_tickers()
        processed_tickers = {}
        for symbol, data in all_tickers.items():
            if symbol.endswith('/KRW'):
                ticker = symbol.split('/')[0]
                if data.get('last') is not None:
                    quote_volume = data.get('quoteVolume')
                    processed_tickers[ticker] = {
                        'closing_price':
                        Decimal(str(data.get('last'))),
                        'quoteVolume':
                        Decimal(str(quote_volume)) if quote_volume is not None else Decimal('0')
                    }
        return processed_tickers

    @api_call_handler
    def get_current_price(self, symbol):
        """Return the latest traded price for the specified symbol."""
        client = self.get_client()
        if not client: return None
        ccxt_symbol = symbol.replace('_', '/')
        if ccxt_symbol in client.markets:
            ticker_data = client.fetch_ticker(ccxt_symbol)
            price = ticker_data.get('last')
            return Decimal(str(price)) if price is not None else None
        return None

    @api_call_handler
    def get_balance(self, currency="KRW"):
        """Retrieve the free balance for the requested currency."""
        client = self.get_client()
        if not client: return Decimal('0')
        balance_info = client.fetch_balance()
        balance = balance_info.get(currency, {}).get('free')
        return Decimal(str(balance)) if balance is not None else Decimal('0')

    @api_call_handler
    def get_balance_all(self):
        """Return available and total balances for each currency."""
        client = self.get_client()
        if not client: return {}
        balance_info = client.fetch_balance()
        processed_balance = {}
        if 'total' in balance_info:
            for currency, data in balance_info['total'].items():
                if data > 0:
                    free_balance = balance_info[currency].get('free')
                    total_balance = data
                    processed_balance[currency] = {
                        'available':
                        Decimal(str(free_balance)) if free_balance is not None else Decimal('0'),
                        'total':
                        Decimal(str(total_balance)) if total_balance is not None else Decimal('0')
                    }
        return processed_balance

    @api_call_handler
    def get_orderbook(self, symbol):
        """Fetch the order book and normalise entries to ``Decimal`` values."""
        client = self.get_client()
        if not client: return None
        ccxt_symbol = symbol.replace('_', '/')
        if ccxt_symbol in client.markets:
            orderbook = client.fetch_order_book(ccxt_symbol)
            return {
                'bids': [{
                    'price': Decimal(str(b[0])),
                    'quantity': Decimal(str(b[1]))
                } for b in orderbook['bids']],
                'asks': [{
                    'price': Decimal(str(a[0])),
                    'quantity': Decimal(str(a[1]))
                } for a in orderbook['asks']]
            }
        return None

    @api_call_handler
    def get_candlestick(self, symbol, chart_intervals="1m"):
        """Return OHLCV data as a pandas DataFrame for the requested timeframe."""
        client = self.get_client()
        if not client: return None
        ccxt_symbol = symbol.replace('_', '/')
        if ccxt_symbol in client.markets:
            ohlcv_list = client.fetch_ohlcv(
                ccxt_symbol, timeframe=chart_intervals, limit=200)
            if not ohlcv_list: return pd.DataFrame()
            df = pd.DataFrame(
                ohlcv_list,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
            return df
        return pd.DataFrame()

    @api_call_handler
    def place_limit_buy_order(self, symbol, price, quantity):
        """Submit a limit buy order via the CCXT client."""
        client = self.get_client()
        if not client: return None
        ccxt_symbol = symbol.replace('_', '/')
        return client.create_limit_buy_order(ccxt_symbol, str(quantity),
                                             str(price))

    @api_call_handler
    def place_limit_sell_order(self, symbol, price, quantity):
        """Submit a limit sell order via the CCXT client."""
        client = self.get_client()
        if not client: return None
        ccxt_symbol = symbol.replace('_', '/')
        return client.create_limit_sell_order(ccxt_symbol, str(quantity),
                                              str(price))

    @api_call_handler
    def place_market_sell_order(self, symbol, quantity):
        """Submit a market sell order via the CCXT client."""
        client = self.get_client()
        if not client: return None
        ccxt_symbol = symbol.replace('_', '/')
        return client.create_market_sell_order(ccxt_symbol, str(quantity))

    @api_call_handler
    def get_order_details(self, order_id, symbol):
        """Fetch order status information for the provided identifier."""
        client = self.get_client()
        if not client: return None
        try:
            ccxt_symbol = symbol.replace('_', '/')
            order = client.fetch_order(order_id, ccxt_symbol)
            for key in ['filled', 'amount', 'price', 'average', 'cost']:
                if key in order and order[key] is not None:
                    order[key] = Decimal(str(order[key]))
            return order
        except ccxt.OrderNotFound:
            return {
                'status': 'not_found',
                'id': order_id,
                'filled': Decimal('0'),
            }

    @api_call_handler
    def cancel_order(self, order_id, symbol, side=None):
        """Request cancellation of the specified open order."""
        client = self.get_client()
        if not client:
            return None
        ccxt_symbol = symbol.replace('_', '/')
        params = {}
        if side:
            params['side'] = str(side).lower()
        return client.cancel_order(order_id, ccxt_symbol, params)

    @api_call_handler
    def get_all_market_details(self):
        """Return the exchange's market metadata dictionary."""
        client = self.get_client()
        if not client: return None
        return client.markets

    @api_call_handler
    def get_recent_trades(self, symbol, count=100):
        """Fetch recent public trades for the symbol up to the requested count."""
        client = self.get_client()
        if not client: return []
        ccxt_symbol = symbol.replace('_', '/')
        trades = client.fetch_trades(ccxt_symbol, limit=count)
        processed_trades = []
        for trade in trades:
            if trade.get('price') is not None and trade.get('amount') is not None:
                processed_trades.append({
                    'transaction_date':
                    trade['datetime'],
                    'price':
                    Decimal(str(trade['price'])),
                    'units_traded':
                    Decimal(str(trade['amount']))
                })
        return processed_trades