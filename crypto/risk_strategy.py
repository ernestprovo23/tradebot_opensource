import alpaca_trade_api as tradeapi
import requests
import json
from trade_stats import download_trades
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from datetime import datetime, timedelta
from port_op import optimize_portfolio
import os
from dotenv import load_dotenv
import logging
import pandas as pd
from pymongo import MongoClient
import time
from scipy.stats import norm, pearsonr
from collections import deque
import numpy as np
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Alpaca credentials
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'  # or https://api.alpaca.markets for live trading
ALPHA_VANTAGE_API = os.getenv('ALPHA_VANTAGE_API')


# Connect to MongoDB
client = MongoClient(MONGO_CONN_STRING)

alpha_vantage_ts = TimeSeries(key=ALPHA_VANTAGE_API, output_format='pandas')
alpha_vantage_crypto = CryptoCurrencies(key=ALPHA_VANTAGE_API, output_format='pandas')

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)


account = api.get_account()
equity = float(account.equity)


def load_risk_params():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    risk_params_file_path = os.path.join(current_script_dir, 'risk_params.json')

    if not os.path.exists(risk_params_file_path):
        risk_params_file_path = os.path.join(current_script_dir, '..', 'risk_params.json')
        risk_params_file_path = os.path.normpath(risk_params_file_path)

    if os.path.exists(risk_params_file_path):
        with open(risk_params_file_path, 'r') as f:
            risk_params = json.load(f)
            # print("Risk parameters loaded:", risk_params)
            # logging.info("Risk parameters loaded successfully.")
            return risk_params
    else:
        logging.error(f"Error: 'risk_params.json' not found at {risk_params_file_path}")
        return None


risk_params = load_risk_params()
if risk_params:
    print("Risk parameters loaded successfully.")
    # Continue with your script using the loaded risk_params
else:
    print("Failed to load risk parameters.")


print(f'Here is max position size currently: {risk_params['max_position_size']}')


class CryptoAsset:
    def __init__(self, symbol, quantity, value_usd):
        self.symbol = symbol
        self.quantity = quantity
        self.value_usd = value_usd
        self.value_24h_ago = None  # To store the value 24 hours ago

        # Connect to MongoDB and retrieve unique crypto pairs
        self.crypto_symbols = self.get_unique_crypto_pairs()

    def get_unique_crypto_pairs(self):
        # Connect to the MongoDB database
        db = client.stock_data
        collection = db.crypto_data

        # Retrieve unique pairs
        pipeline = [
            {"$group": {"_id": {"Crypto": "$Crypto", "Quote": "$Quote"}}},
            {"$project": {"_id": 0, "pair": {"$concat": ["$_id.Crypto", "/", "$_id.Quote"]}}}
        ]
        results = collection.aggregate(pipeline)
        unique_pairs = [doc['pair'] for doc in results]

        return unique_pairs

    def profit_loss_24h(self):
        if self.value_24h_ago is not None:
            return (self.value_usd - self.value_24h_ago) / self.value_24h_ago * 100
        else:
            return None


# Add after existing imports in risk_strategy.py
class MarketRegime:
    LOW_VOL = 'low_volatility'
    NORMAL = 'normal'
    HIGH_VOL = 'high_volatility'
    CRISIS = 'crisis'


class PriceHistory:
    def __init__(self, window_size=252):
        self.prices = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

    def add(self, price, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
        self.prices.append(price)
        self.timestamps.append(timestamp)


# Define a class to manage the portfolio
class PortfolioManager:
    def __init__(self, api):
        self.api = api
        self.assets = {}
        self.operations = 0  # track the number of operations

    def increment_operations(self):
        self.operations += 1

    def add_asset(self, symbol, quantity, value_usd):
        self.assets[symbol] = CryptoAsset(symbol, quantity, value_usd)

    def update_asset_value(self, symbol, value_usd):
        if symbol in self.assets:
            self.assets[symbol].value_usd = value_usd

    def portfolio_value(self):
        return sum(asset.value_usd for asset in self.assets.values())

    def portfolio_balance(self):
        return {symbol: (asset.value_usd / self.portfolio_value()) * 100 for symbol, asset in self.assets.items()}

    def sell_decision(self, symbol):
        balance = self.portfolio_balance()

        if balance[symbol] > 25 or balance[symbol] > 0.4 * sum(balance.values()):
            return True
        else:
            return False

    def scale_out(self, symbol):
        quantity_to_sell = int(self.assets[symbol].quantity * 0.1)  # Sell 10% of holdings
        return quantity_to_sell

    def update_asset_values_24h(self):
        for asset in self.assets.values():
            asset.value_24h_ago = asset.value_usd



### Usable functions for the RiskManagement class below
def get_exchange_rate(base_currency, quote_currency):
    # Your Alpha Vantage API key
    api_key = ALPHA_VANTAGE_API

    # Prepare the URL
    url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={base_currency}&to_currency={quote_currency}&apikey={api_key}"

    # Send GET request
    response = requests.get(url)

    # Parse JSON response
    data = json.loads(response.text)

    # Extract exchange rate
    exchange_rate = data["Realtime Currency Exchange Rate"]["5. Exchange Rate"]

    return float(exchange_rate)


def fetch_account_details():
    url = "https://paper-api.alpaca.markets/v2/account"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Failed to fetch account details: {response.text}")
        return None


def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_greeks(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return {
        'delta': norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1),
        'gamma': norm.pdf(d1) / (S * sigma * np.sqrt(T)),
        'theta': -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2) if option_type == "call" else -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2),
        'vega': S * norm.pdf(d1) * np.sqrt(T)
    }

## RiskManagement class developed for usage on the account
class RiskManagement:
    crypto_symbols = ['AAVE/USD', 'ALGO/USD', 'AVAX/USD', 'BCH/USD', 'BTC/USD', 'ETH/USD',
                  'LINK/USD', 'LTC/USD', 'TRX/USD', 'UNI/USD', 'USDT/USD', 'SHIB/USD']
    def __init__(self, api, risk_params):
        self.api = api
        self.risk_params = risk_params
        self.alpha_vantage_crypto = CryptoCurrencies(key=ALPHA_VANTAGE_API, output_format='pandas')
        self.manager = PortfolioManager(api)
        self.crypto_value = 0
        self.crypto_value = 0
        self.commodity_value = 0
        self.options_crypto_notional_value = 0
        self.options_commodity_notional_value = 0
        self.max_options_allocation = 0.2  # Maximum 20% of portfolio allocated to options
        self.max_options_loss_threshold = 0.1
        self.total_trades_today = 0
        self.MIN_TRANSACTION_VALUE = 1000  # Minimum USD value for any trade
        self.REBALANCE_THRESHOLD = 0.015
        self.BATCH_COOLDOWN = 300  # 5-minute cooldown between batch operations
        self.last_rebalance_time = {}  # Track last rebalance time for each asset class
        self.market_metrics = {}

        self.crypto_symbols = ['AAVE/USD', 'ALGO/USD', 'AVAX/USD', 'BCH/USD', 'BTC/USD', 'ETH/USD',
                               'LINK/USD', 'LTC/USD', 'TRX/USD', 'UNI/USD', 'USDT/USD', 'SHIB/USD']
        self.TARGET_ALLOCATION = {
            'options': 0.20,  # 20%
            'crypto': 0.30,  # 30%
            'equities': 0.50  # 50%
        }
        self.initialize_account_info()
        self.price_histories = {}
        self.correlation_threshold = 0.7
        self.vol_window = 252  # 1 year of daily data
        self.correlation_window = 60  # 60 days for correlation
        self.ALPHA_VANTAGE_API = os.getenv('ALPHA_VANTAGE_API')
        # Add new order management parameters
        self.order_cooldown = {}  # Track order timing
        self.MIN_ORDER_SPACING = 300  # 5 minutes between orders for same symbol
        self.STALE_ORDER_THRESHOLD = 3600  # 1 hour for stale orders
        self.price_deviation_threshold = 0.02  # 2% price deviation threshold
        self.min_transaction_value = 1000  # Minimum transaction value
        # Initialize order tracking
        self.active_orders = {}  # Track active orders by symbol
        self.last_order_prices = {}  # Track last order prices
        self.order_history = {}  # Track order history for analysis

        self.initialize_account_info()

    class MarketMetrics:
        def __init__(self, window_size=100):
            self.price_history = deque(maxlen=window_size)
            self.volume_history = deque(maxlen=window_size)
            self.volatility = None
            self.last_update = None

        def update(self, price, volume):
            self.price_history.append(price)
            self.volume_history.append(volume)
            if len(self.price_history) > 1:
                returns = np.diff(np.log(list(self.price_history)))
                self.volatility = np.std(returns) * np.sqrt(252)
            self.last_update = datetime.now()


    class CryptoRiskManager:
        def __init__(self):
            self.metrics = {}
            self.correlation_matrix = {}
            self.VAR_CONFIDENCE = 0.99
            self.MAX_DRAWDOWN = 0.02
            self.MIN_PROFIT_RATIO = 1.5
            self.vol_window = 24

    def track_order(self, symbol, order_type, quantity, price):
        """Track order details for analysis"""
        current_time = time.time()

        if symbol not in self.order_history:
            self.order_history[symbol] = []

        self.order_history[symbol].append({
            'timestamp': current_time,
            'type': order_type,
            'quantity': quantity,
            'price': price
        })

        self.last_order_prices[symbol] = price
        self.order_cooldown[symbol] = current_time

    def check_order_timing(self, symbol):
        """Check if enough time has passed since last order"""
        current_time = time.time()
        last_order_time = self.order_cooldown.get(symbol, 0)

        time_since_last_order = current_time - last_order_time
        if time_since_last_order < self.MIN_ORDER_SPACING:
            logging.info(f"Order cooldown in effect for {symbol}. Time since last order: {time_since_last_order:.2f}s")
            return False
        return True

    def check_price_deviation(self, symbol, proposed_price):
        """Check if price has deviated significantly from last order"""
        if symbol in self.last_order_prices:
            last_price = self.last_order_prices[symbol]
            deviation = abs(proposed_price - last_price) / last_price

            if deviation > self.price_deviation_threshold:
                logging.info(f"Price deviation ({deviation:.2%}) exceeds threshold for {symbol}")
                return False
        return True

    def cleanup_old_orders(self):
        """Remove old order tracking data"""
        current_time = time.time()
        cutoff_time = current_time - self.STALE_ORDER_THRESHOLD

        # Clean order cooldowns
        self.order_cooldown = {
            symbol: timestamp
            for symbol, timestamp in self.order_cooldown.items()
            if timestamp > cutoff_time
        }

        # Clean order history
        for symbol in self.order_history:
            self.order_history[symbol] = [
                order for order in self.order_history[symbol]
                if order['timestamp'] > cutoff_time
            ]

    def validate_and_manage_orders(self, symbol, proposed_order_price, side='buy'):
        """
        Sophisticated order management system that validates and manages open orders.
        Removes position size restrictions for sells.
        """
        try:
            logging.info(f"\nValidating orders for {symbol} - {side}")

            # Get all open orders for the symbol
            open_orders = self.api.list_orders(status='open', symbols=[symbol])

            # Get current position if any
            try:
                position = self.api.get_position(symbol)
                has_position = True
                current_position_size = float(position.qty)
                avg_entry = float(position.avg_entry_price)
                logging.info(f"Current position: {current_position_size} units at ${avg_entry:,.2f}")
            except Exception:
                has_position = False
                current_position_size = 0
                avg_entry = 0
                logging.info("No current position found")

            # Get current market price
            current_price = self.get_current_price(symbol)
            if not current_price:
                return False, "Could not fetch current market price"

            logging.info(f"Current market price: ${current_price:,.2f}")

            if side == 'sell':
                # For sell orders, only check if we have the position
                if not has_position:
                    return False, "No position to sell"

                # Calculate total pending sells
                pending_sell_quantity = sum(
                    float(order.qty) for order in open_orders if order.side == 'sell'
                )

                # Calculate proposed sell quantity based on provided value or position
                if hasattr(self, 'proposed_sell_qty'):
                    proposed_sell_qty = self.proposed_sell_qty
                else:
                    # Use the actual provided sell quantity or a safe default
                    proposed_sell_qty = current_position_size * 0.1  # Default to 10% of position if not specified

                total_sell_quantity = pending_sell_quantity + proposed_sell_qty

                logging.info(f"Sell validation:")
                logging.info(f"- Current position: {current_position_size}")
                logging.info(f"- Pending sells: {pending_sell_quantity}")
                logging.info(f"- Proposed sell quantity: {proposed_sell_qty}")
                logging.info(f"- Total sell quantity: {total_sell_quantity}")

                # Only verify we're not selling more than we own
                if total_sell_quantity > current_position_size:
                    logging.info(f"Cannot sell {total_sell_quantity} units when position is {current_position_size}")
                    return False, "Total sell quantity would exceed position size"

                return True, "Sell order validation successful"

            else:  # Buy order validation
                # Calculate total pending buy orders value
                pending_buy_value = sum(
                    float(order.qty) * float(order.limit_price)
                    for order in open_orders
                    if order.side == 'buy'
                )

                # Get account buying power
                account = self.api.get_account()
                buying_power = float(account.buying_power)

                logging.info(f"Buy validation:")
                logging.info(f"- Buying power: ${buying_power:,.2f}")
                logging.info(f"- Pending buy value: ${pending_buy_value:,.2f}")

                # Price deviation threshold
                price_threshold = 0.02  # 2% deviation

                # Check existing orders at similar price points
                for order in open_orders:
                    if order.side == 'buy':
                        order_price = float(order.limit_price)
                        price_diff = abs(order_price - proposed_order_price) / proposed_order_price

                        if price_diff < price_threshold:
                            return False, f"Existing order at similar price point: {order_price}"

                # Check if current market price is more favorable
                if current_price < proposed_order_price:
                    return False, f"Current market price ({current_price}) is better than limit price ({proposed_order_price})"

                # Calculate total exposure
                total_exposure = pending_buy_value + (current_position_size * current_price)
                max_allocation = self.calculate_equity_allocation('crypto')

                logging.info(f"Exposure check:")
                logging.info(f"- Total exposure: ${total_exposure:,.2f}")
                logging.info(f"- Max allocation: ${max_allocation:,.2f}")

                # Calculate proposed buy quantity with proper order type
                proposed_buy_qty = self.calculate_quantity(symbol, order_type='buy')
                proposed_buy_value = proposed_buy_qty * proposed_order_price

                if total_exposure + proposed_buy_value > max_allocation:
                    return False, "Total exposure would exceed maximum allocation"

                return True, "Buy order validation successful"

        except Exception as e:
            logging.error(f"Error in order validation for {symbol}: {e}")
            return False, f"Validation error: {str(e)}"

    def clean_stale_orders(self):
        """
        Periodically clean up stale orders and rebalance order book.
        """
        try:
            all_orders = self.api.list_orders(status='open')
            current_time = datetime.now()

            for order in all_orders:
                try:
                    symbol = order.symbol
                    current_price = self.get_current_price(symbol)

                    if not current_price:
                        continue

                    order_age = (current_time - parser.parse(order.submitted_at)).total_seconds()
                    order_price = float(order.limit_price)

                    should_cancel = False
                    cancel_reason = ""

                    # Cancel criteria
                    if order_age > 3600:  # 1 hour old
                        should_cancel = True
                        cancel_reason = "Order too old"
                    elif order.side == 'buy' and order_price > current_price * 1.02:
                        should_cancel = True
                        cancel_reason = "Buy order above market price"
                    elif order.side == 'sell' and order_price < current_price * 0.98:
                        should_cancel = True
                        cancel_reason = "Sell order below market price"

                    if should_cancel:
                        self.api.cancel_order(order.id)
                        logging.info(f"Cancelled order {order.id} for {symbol}: {cancel_reason}")

                except Exception as e:
                    logging.error(f"Error processing order {order.id}: {e}")
                    continue

        except Exception as e:
            logging.error(f"Error in clean_stale_orders: {e}")

    def detect_market_regime(self, symbol: str) -> str:
        """
        Detect current market regime based on volatility
        """
        try:
            vol = self.calculate_volatility(symbol)
            if vol is None or vol == 0:
                # Calculate using alternative method if primary fails
                prices = self.get_recent_prices(symbol, 20)
                if prices and len(prices) > 1:
                    returns = np.diff(np.log(prices))
                    vol = np.std(returns) * np.sqrt(252)
                else:
                    return MarketRegime.NORMAL
        except Exception as e:
            logging.error(f"Error detecting market regime for {symbol}: {e}")
            return MarketRegime.NORMAL

    def adjust_position_size_for_regime(self, symbol: str, base_size: float) -> float:
        """
        Adjust position size based on market regime
        """
        try:
            regime = self.detect_market_regime(symbol)
            regime_adjustments = {
                MarketRegime.LOW_VOL: 1.2,
                MarketRegime.NORMAL: 1.0,
                MarketRegime.HIGH_VOL: 0.7,
                MarketRegime.CRISIS: 0.4
            }
            return base_size * regime_adjustments.get(regime, 1.0)
        except Exception as e:
            logging.error(f"Error adjusting position size for regime: {e}")
            return base_size

    def calculate_asset_correlations(self, symbols: list = None) -> pd.DataFrame:
        try:
            if not symbols or len(symbols) < 2:
                return pd.DataFrame()

            price_data = {}
            for symbol in symbols:
                prices = self.get_recent_prices(symbol, 60)
                if prices and len(prices) > 0:
                    price_data[symbol] = prices

            if len(price_data) < 2:  # Need at least 2 assets for correlation
                return pd.DataFrame()

            df = pd.DataFrame(price_data)
            # Fill any missing values before correlation
            df = df.fillna(method='ffill').fillna(method='bfill')
            return df.corr()
        except Exception as e:
            logging.error(f"Error calculating correlations: {e}")
            return pd.DataFrame()

    def adjust_for_correlation(self, symbol: str, position_size: float) -> float:
        """
        Adjust position size based on portfolio correlation
        """
        try:
            corr_matrix = self.calculate_asset_correlations()
            if not corr_matrix.empty and symbol in corr_matrix.index:
                # Get average correlation with other assets
                avg_corr = corr_matrix[symbol].mean()
                # Reduce position size for highly correlated assets
                if avg_corr > 0.7:  # High correlation threshold
                    position_size *= (1 - avg_corr)
            return position_size
        except Exception as e:
            logging.error(f"Error adjusting for correlation: {e}")
            return position_size

    def initialize_account_info(self):
        account = self.api.get_account()
        self.peak_portfolio_value = float(account.cash)

    def calculate_current_allocation(self):
        positions = self.api.list_positions()
        total_value = sum(float(position.market_value) for position in positions)
        allocation = {
            'options': 0,
            'crypto': 0,
            'equities': 0
        }

    def calculate_options_allocation(self):
        positions = self.api.list_positions()
        options_value = sum(float(position.market_value) for position in positions if 'OPT' in position.symbol)
        account_value = float(self.api.get_account().portfolio_value)
        return options_value / account_value

    def calculate_portfolio_value(self):
        positions = self.api.list_positions()
        portfolio_value = sum(float(position.market_value) for position in positions)
        return portfolio_value

    def calculate_current_allocation(self):
        """
        Calculate current allocation across different asset classes with proper error handling
        Returns a dictionary with allocation percentages
        """
        try:
            positions = self.api.list_positions()
            total_value = 0
            allocation = {
                'options': 0,
                'crypto': 0,
                'equities': 0
            }

            # First pass to calculate total value with error handling
            for position in positions:
                try:
                    if hasattr(position, 'market_value') and position.market_value is not None:
                        market_value = float(position.market_value)
                    else:
                        # Fallback calculation if market_value is not available
                        qty = float(position.qty) if position.qty is not None else 0
                        current_price = float(position.current_price) if position.current_price is not None else 0
                        market_value = qty * current_price

                    if market_value > 0:  # Only add valid market values
                        total_value += market_value
                except (ValueError, AttributeError, TypeError) as e:
                    logging.error(f"Error processing position {position.symbol}: {str(e)}")
                    continue

            # Second pass to calculate allocations
            if total_value > 0:  # Only proceed if we have valid total value
                for position in positions:
                    try:
                        if hasattr(position, 'market_value') and position.market_value is not None:
                            market_value = float(position.market_value)
                        else:
                            qty = float(position.qty) if position.qty is not None else 0
                            current_price = float(position.current_price) if position.current_price is not None else 0
                            market_value = qty * current_price

                        if market_value > 0:  # Only process valid market values
                            if 'OPT' in position.symbol:
                                allocation['options'] += market_value
                            elif position.symbol.endswith('USD'):
                                allocation['crypto'] += market_value
                            else:
                                allocation['equities'] += market_value
                    except (ValueError, AttributeError, TypeError) as e:
                        logging.error(f"Error calculating allocation for {position.symbol}: {str(e)}")
                        continue

                # Convert to percentages
                for asset_class in allocation:
                    allocation[asset_class] = (allocation[asset_class] / total_value) if total_value > 0 else 0

            logging.info(f"Current allocation: {allocation}")
            return allocation

        except Exception as e:
            logging.error(f"Error in calculate_current_allocation: {str(e)}")
            return {
                'options': 0,
                'crypto': 0,
                'equities': 0
            }

    def calculate_optimal_sell_quantity(self, position, target_value):
        """Calculate optimal sell quantity with enhanced validation"""
        try:
            if not hasattr(position, 'market_value') or position.market_value is None:
                current_price = self.get_current_price(position.symbol)
                if current_price is None:
                    return 0
                current_value = float(position.qty) * current_price
            else:
                current_value = float(position.market_value)
                current_price = float(position.current_price)

            if current_value <= self.MIN_TRANSACTION_VALUE:
                return float(position.qty)

            excess_value = current_value - target_value
            if excess_value < self.MIN_TRANSACTION_VALUE:
                return 0

            optimal_qty = int(excess_value / current_price)
            lot_size = self.get_lot_size(position.symbol)
            optimal_qty = (optimal_qty // lot_size) * lot_size

            return optimal_qty if optimal_qty * current_price >= self.MIN_TRANSACTION_VALUE else 0

        except Exception as e:
            logging.error(f"Error calculating sell quantity for {position.symbol}: {e}")
            return 0

    def get_lot_size(self, symbol):
        """Get appropriate lot size based on asset type"""
        if symbol.endswith('USD'):  # Crypto
            if 'BTC' in symbol:
                return 0.01  # Bitcoin lot size
            elif 'ETH' in symbol:
                return 0.1  # Ethereum lot size
            return 1.0  # Default crypto lot size
        return 100  # Standard equity lot size

    def can_rebalance_asset(self, asset_class):
        """Check if enough time has passed since last rebalance"""
        current_time = time.time()
        last_time = self.last_rebalance_time.get(asset_class, 0)
        return (current_time - last_time) >= self.BATCH_COOLDOWN

    def rebalance_portfolio(self):
        """Enhanced portfolio rebalancing with improved controls"""
        try:
            current_allocation = self.calculate_current_allocation()
            account = self.api.get_account()
            total_value = float(account.portfolio_value)

            # Calculate maximum imbalance
            max_imbalance = max(
                abs(current_allocation[asset] - target)
                for asset, target in self.TARGET_ALLOCATION.items()
                for asset, target in self.TARGET_ALLOCATION.items()
            )

            # Only proceed if significant imbalance exists
            if max_imbalance < self.REBALANCE_THRESHOLD:
                logging.info(
                    f"Maximum allocation imbalance {max_imbalance:.2%} below threshold {self.REBALANCE_THRESHOLD:.2%}")
                return

            # Calculate and sort all needed trades
            trades_needed = []
            for asset_class, target_pct in self.TARGET_ALLOCATION.items():
                current_pct = current_allocation[asset_class]
                diff_pct = target_pct - current_pct

                if abs(diff_pct) >= self.REBALANCE_THRESHOLD:
                    amount_to_trade = diff_pct * total_value
                    if abs(amount_to_trade) >= self.MIN_TRANSACTION_VALUE:
                        trades_needed.append({
                            'asset_class': asset_class,
                            'amount': amount_to_trade,
                            'is_buy': diff_pct > 0,
                            'imbalance': abs(diff_pct)
                        })

            # Sort by imbalance size
            trades_needed.sort(key=lambda x: x['imbalance'], reverse=True)

            # Execute trades with enhanced error handling
            for trade in trades_needed:
                asset_class = trade.get('asset_class', 'Unknown')
                try:
                    if self.can_rebalance_asset(asset_class):
                        if trade['is_buy']:
                            self.buy_asset_class(asset_class, trade['amount'])
                        else:
                            self.sell_asset_class(asset_class, abs(trade['amount']))
                    else:
                        logging.info(f"Skipping {asset_class} due to cooldown period")
                except Exception as e:
                    logging.error(f"Error processing trade for {asset_class}: {e}")

        except Exception as e:
            logging.error(f"Error in enhanced rebalance_portfolio: {e}")

    def buy_asset_class(self, asset_class, amount_to_trade):
        account_details = fetch_account_details()
        if not account_details:
            return

        available_cash = float(account_details['cash'])

        if asset_class == 'crypto':
            for symbol in self.crypto_symbols:
                alpaca_symbol = self.convert_symbol(symbol, to_alpaca=True, asset_type='crypto')
                current_price = self.get_current_price(symbol)
                if current_price and current_price > 0:
                    qty = self.calculate_quantity(symbol, order_type='buy', asset_type='crypto')
                    if qty > 0:
                        logging.info(f'Amount to Trade: {amount_to_trade}, Suggested Qty: {qty}')
                        if self.validate_trade(alpaca_symbol, qty, 'buy'):
                            try:
                                self.api.submit_order(
                                    symbol=alpaca_symbol,
                                    qty=qty,
                                    side='buy',
                                    type='market',
                                    time_in_force='gtc'
                                )
                                logging.info(f"Bought {qty} of {alpaca_symbol} to rebalance crypto allocation.")
                            except tradeapi.rest.APIError as e:
                                if 'insufficient balance' in str(e):
                                    available_qty = float(str(e).split('available: ')[1].split(')')[0])
                                    if available_qty > 0:
                                        try:
                                            self.api.submit_order(
                                                symbol=symbol,
                                                qty=available_qty,
                                                side='buy',
                                                type='market',
                                                time_in_force='gtc'
                                            )
                                            logging.info(
                                                f"Bought {available_qty} of {symbol} (adjusted for available balance).")
                                        except Exception as e2:
                                            logging.error(f"Failed to submit adjusted order for {symbol}: {e2}")
                                elif 'asset is not active' in str(e):
                                    logging.warning(f"Skipping {symbol} because it is not active.")
                                else:
                                    logging.error(f"Failed to submit order for {symbol}: {e}")
                else:
                    logging.warning(f"Skipping {symbol} due to invalid price: {current_price}")
        elif asset_class == 'options' or asset_class == 'equities':
            logging.info(f"Trading for {asset_class} not yet implemented.")

    def sell_asset_class(self, asset_class, amount_to_trade):
        """Enhanced version with stronger batching and timing controls"""
        try:
            if not self.can_rebalance_asset(asset_class):
                logging.info(f"Skipping {asset_class} rebalance due to cooldown period")
                return

            if asset_class == 'crypto':
                positions = self.api.list_positions()
                crypto_positions = [p for p in positions if p.symbol.endswith('USD')]

                if not crypto_positions:
                    return

                total_portfolio_value = sum(float(p.market_value) for p in positions if p.market_value)
                total_crypto_value = sum(float(p.market_value) for p in crypto_positions if p.market_value)

                if total_portfolio_value == 0:
                    logging.error("Total portfolio value is 0, cannot proceed with rebalancing")
                    return

                target_crypto_value = total_portfolio_value * self.TARGET_ALLOCATION['crypto']
                current_allocation = total_crypto_value / total_portfolio_value if total_portfolio_value > 0 else 0

                if abs(current_allocation - self.TARGET_ALLOCATION['crypto']) < self.REBALANCE_THRESHOLD:
                    logging.info(
                        f"Crypto allocation difference ({abs(current_allocation - self.TARGET_ALLOCATION['crypto']):.2%}) "
                        f"below threshold ({self.REBALANCE_THRESHOLD:.2%})")
                    return

                target_per_position = target_crypto_value / len(crypto_positions) if crypto_positions else 0
                sell_orders = []
                for position in crypto_positions:
                    try:
                        qty_to_sell = self.calculate_optimal_sell_quantity(position, target_per_position)
                        if qty_to_sell > 0:
                            current_price = self.get_current_price(position.symbol)
                            if current_price:
                                sell_orders.append({
                                    'symbol': position.symbol,
                                    'qty': qty_to_sell,
                                    'price': current_price,
                                    'value': qty_to_sell * current_price
                                })
                    except Exception as e:
                        logging.error(f"Error processing sell order for {position.symbol}: {e}")
                        continue

                if sell_orders:
                    total_sell_value = sum(order['value'] for order in sell_orders)
                    if total_sell_value >= self.MIN_TRANSACTION_VALUE:
                        for order in sell_orders:
                            try:
                                if self.validate_trade(order['symbol'], order['qty'], 'sell'):
                                    self.api.submit_order(
                                        symbol=order['symbol'],
                                        qty=order['qty'],
                                        side='sell',
                                        type='market',
                                        time_in_force='gtc'
                                    )
                                    logging.info(f"Executed sell: {order['qty']} of {order['symbol']}")
                            except Exception as e:
                                logging.error(f"Error executing sell order for {order['symbol']}: {e}")
                                continue

                        self.last_rebalance_time[asset_class] = time.time()

        except Exception as e:
            logging.error(f"Error in enhanced sell_asset_class for {asset_class}: {e}")

    def can_rebalance_asset(self, asset_class):
        """Check if enough time has passed since last rebalance"""
        current_time = time.time()
        last_time = self.last_rebalance_time.get(asset_class, 0)
        return (current_time - last_time) >= self.BATCH_COOLDOWN

    def update_max_crypto_equity(self):
        # Get the current buying power of the account
        account = self.api.get_account()
        buying_power = float(account.buying_power)

        # Compute max_crypto_equity
        max_crypto_equity = buying_power

        # Update the JSON file with the new value
        self.risk_params['max_crypto_equity'] = max_crypto_equity

        return max_crypto_equity

    def check_options_risk(self, symbol, quantity, price):
        total_portfolio_value = float(self.api.get_account().portfolio_value)
        options_position_value = quantity * price
        options_allocation = options_position_value / total_portfolio_value

        if options_allocation > self.max_options_allocation:
            return False

        # Check individual options position loss
        if symbol in self.options_positions:
            entry_price = self.options_positions[symbol]['entry_price']
            if (price - entry_price) / entry_price < -self.max_options_loss_threshold:
                return False

        return True

    def monitor_options_expiration(self):
        positions = self.api.list_positions()
        current_date = datetime.now().date()

        for position in positions:
            if 'OPT' in position.symbol:
                expiration_date = position.expiration_date.date()
                days_to_expiration = (expiration_date - current_date).days

                if days_to_expiration <= 7:
                    # Close options position if it's within 7 days of expiration
                    self.close_position(position.symbol)
                    print(f"Closed options position {position.symbol} due to approaching expiration.")

    def monitor_volatility(self):
        # Retrieve market volatility data (e.g., VIX index)
        volatility = self.get_market_volatility()

        if volatility > self.risk_params['max_volatility_threshold']:
            # Adjust options positions based on high volatility
            self.adjust_options_positions(volatility)
            print(f"Adjusted options positions due to high market volatility: {volatility}")



    def calculate_position_greeks(self, position):
        # Extract the necessary parameters from the position object
        S = float(position.current_price)
        K = float(position.strike_price)
        T = (position.expiration_date - datetime.now()).days / 365  # Time to expiration in years
        r = 0.01  # Risk-free rate (adjust as needed)
        sigma = 0.20  # Volatility (adjust as needed)
        option_type = position.option_type.lower()

        # Calculate the Greeks
        greeks = calculate_greeks(S, K, T, r, sigma, option_type)
        return greeks

    def calculate_equity_allocation(self, asset_type='crypto'):
        """Calculate equity allocation with robust error handling and validation"""
        try:
            # Load and validate risk parameters
            risk_params = load_risk_params()
            if not risk_params:
                logging.error("Failed to load risk parameters")
                return float(self.risk_params.get('max_crypto_equity', 100000))  # Default fallback

            # Get account equity
            account = self.api.get_account()
            if not account:
                logging.error("Failed to get account information")
                return float(self.risk_params.get('max_crypto_equity', 100000))

            equity = float(account.equity)
            logging.info(f"Current account equity: ${equity:,.2f}")

            # Get positions
            positions = self.api.list_positions()

            # Calculate total values with proper error handling
            try:
                total_crypto_value = sum(
                    float(p.market_value) if p.market_value is not None else float(p.qty) * float(p.current_price)
                    for p in positions
                    if p.symbol.endswith('USD')
                )

                total_equities_value = sum(
                    float(p.market_value) if p.market_value is not None else float(p.qty) * float(p.current_price)
                    for p in positions
                    if not p.symbol.endswith('USD') and 'OPT' not in p.symbol
                )

                total_options_value = sum(
                    float(p.market_value) if p.market_value is not None else float(p.qty) * float(p.current_price)
                    for p in positions
                    if 'OPT' in p.symbol
                )
            except Exception as e:
                logging.error(f"Error calculating position values: {e}")
                return float(self.risk_params.get('max_crypto_equity', 100000))

            # Get maximum allocations from risk parameters
            max_crypto_equity = float(self.risk_params.get('max_crypto_equity', 100000))
            max_equities_equity = float(self.risk_params.get('max_equity_equity', 100000))

            logging.info(f"Current allocations:")
            logging.info(f"- Crypto: ${total_crypto_value:,.2f}")
            logging.info(f"- Equities: ${total_equities_value:,.2f}")
            logging.info(f"- Options: ${total_options_value:,.2f}")
            logging.info(f"Maximum allocations:")
            logging.info(f"- Max Crypto: ${max_crypto_equity:,.2f}")
            logging.info(f"- Max Equities: ${max_equities_equity:,.2f}")

            if asset_type == 'crypto':
                return max_crypto_equity
            elif asset_type == 'equity':
                return max_equities_equity
            elif asset_type == 'options':
                max_total_allocation = max_crypto_equity + max_equities_equity
                remaining_allocation = max_total_allocation - (
                            total_crypto_value + total_equities_value + total_options_value)
                return max(0, remaining_allocation)
            else:
                logging.error(f"Invalid asset type specified: {asset_type}")
                return 0

        except Exception as e:
            logging.error(f"Error in calculate_equity_allocation: {e}")
            # Return a safe default value instead of 0
            return float(self.risk_params.get('max_crypto_equity', 100000))

    def optimize_portfolio(self, risk_aversion):
        # Get historical data for each symbol
        historical_data = {}
        for symbol in self.crypto_symbols:
            data, _ = alpha_vantage_crypto.get_digital_currency_daily(symbol=symbol, market='USD')
            historical_data[symbol] = data['4b. close (USD)']

        # Calculate expected returns and covariance matrix
        returns_data = pd.DataFrame(historical_data).pct_change()
        expected_returns = returns_data.mean()
        covariance_matrix = returns_data.cov()

        # Total investment amount
        total_investment = float(self.api.get_account().equity)

        # Run optimization in separate script
        quantities_to_purchase = optimize_portfolio(expected_returns, covariance_matrix, risk_aversion,
                                                    total_investment)

        return quantities_to_purchase

    def get_daily_returns(self, symbol: str, days: int = 3) -> float:
        url = "https://paper-api.alpaca.markets/v2/positions"
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
        }
        response = requests.get(url, headers=headers)
        data = response.json()

        # Find the position for the given symbol
        position_data = None
        for position in data:
            if position['symbol'] == symbol:
                position_data = position
                break

        if position_data is None:
            raise ValueError(f"No position found for symbol {symbol}")

        # Get the closing prices for the past `days` days
        closing_prices = []
        for _ in range(days):
            lastday_price = position_data.get('lastday_price')
            if lastday_price is not None:
                closing_prices.append(float(lastday_price))
            else:
                print(f"Missing lastday_price for symbol {symbol}. Skipping calculation of daily returns.")
                return []

        # Calculate the daily returns
        returns = [np.log(closing_prices[i] / closing_prices[i - 1]) for i in range(1, len(closing_prices))]

        # Return the daily returns
        return returns

    def get_position(self, symbol):
        """
        Get position details for a specific symbol
        """
        positions = self.api.list_positions()
        alpaca_symbol = self.convert_symbol(symbol, to_alpaca=True)
        # Filter positions to find matches for the symbol
        symbol_positions = [p for p in positions if p.symbol == alpaca_symbol]

        if not symbol_positions:
            print(f"No positions found for {symbol}")
            return None

        # Assuming there's only one position per symbol
        p = symbol_positions[0]

        # Get actual qty and unsettled qty
        actual_qty = float(p.qty)
        unsettled_qty = float(p.unsettled_qty) if hasattr(p,
                                                          'unsettled_qty') else 0  # Assuming 'unsettled_qty' is the correct attribute name

        pos = {
            "symbol": p.symbol,
            "qty": actual_qty,
            "unsettled_qty": unsettled_qty,
            "avg_entry_price": float(p.avg_entry_price) if p.avg_entry_price is not None else None
        }

        return pos

    def calculate_position_values(self):
        """
        Calculate total values of crypto and commodity positions with robust error handling
        """
        try:
            positions = self.api.list_positions()
            self.crypto_value = 0.0
            self.commodity_value = 0.0

            for position in positions:
                try:
                    # Get position quantity with validation
                    qty = float(position.qty) if position.qty is not None else 0.0

                    # Get current price with fallback methods
                    current_price = None

                    # Try getting price from position first
                    if hasattr(position, 'current_price') and position.current_price is not None:
                        try:
                            current_price = float(position.current_price)
                        except (ValueError, TypeError):
                            pass

                    # If price still None, try getting from market
                    if current_price is None or current_price == 0:
                        current_price = self.get_current_price(position.symbol)

                    # If still no valid price, skip this position
                    if current_price is None or current_price == 0:
                        logging.warning(
                            f"Could not get valid price for {position.symbol}, skipping position value calculation")
                        continue

                    position_value = qty * current_price

                    # Categorize the position
                    if position.symbol.endswith('USD'):  # Crypto position
                        self.crypto_value += position_value
                        logging.debug(f"Added crypto position value for {position.symbol}: ${position_value:,.2f}")
                    else:  # Commodity position
                        self.commodity_value += position_value
                        logging.debug(f"Added commodity position value for {position.symbol}: ${position_value:,.2f}")

                except Exception as e:
                    logging.error(f"Error processing position {position.symbol}: {str(e)}")
                    continue

            logging.info(f"Total crypto value: ${self.crypto_value:,.2f}")
            logging.info(f"Total commodity value: ${self.commodity_value:,.2f}")

            return self.crypto_value, self.commodity_value

        except Exception as e:
            logging.error(f"Error in calculate_position_values: {str(e)}")
            return 0.0, 0.0

    def validate_trade(self, symbol, qty_or_order_type, order_type=None):
        """
        Enhanced trade validation that properly handles sell orders without position size restrictions.
        Includes detailed debugging output for allocation and exposure tracking.
        """
        logging.info(f"\n=== Starting Trade Validation for {symbol} ===")

        # Handle different parameter combinations
        if isinstance(qty_or_order_type, str) and order_type is None:
            order_type = qty_or_order_type.lower()
            qty = None
        else:
            try:
                qty = float(qty_or_order_type)
                order_type = order_type.lower() if order_type else 'buy'
            except (ValueError, AttributeError):
                logging.error(f"Invalid quantity provided: {qty_or_order_type}")
                return False

        logging.info(f"Order type: {order_type}, Initial Quantity: {qty}")

        try:
            # Check order timing cooldown
            if not self.check_order_timing(symbol):
                logging.info(f"Order cooldown in effect for {symbol}")
                return False

            # Get account information with detailed logging
            account = self.api.get_account()
            portfolio_value = float(account.portfolio_value)
            account_cash = float(account.cash)
            logging.info(f"\nAccount Status:")
            logging.info(f"- Portfolio Value: ${portfolio_value:,.2f}")
            logging.info(f"- Available Cash: ${account_cash:,.2f}")

            # Get current price with safety check
            current_price = self.get_current_price(symbol)
            if not current_price or current_price <= 0:
                logging.info(f"Invalid price for {symbol}: {current_price}")
                return False
            current_price = float(current_price)

            # Check for existing orders at similar price points
            can_proceed, reason = self.validate_and_manage_orders(symbol, current_price, order_type)
            if not can_proceed:
                logging.info(f"Order validation failed: {reason}")
                return False

            print(f'Pre calcd Expected quantity: {qty}')
            # If qty wasn't provided, calculate it based on order type
            if qty is None:
                qty = self.calculate_quantity(symbol, order_type=order_type)
                logging.info(f"Validate Trade Calculated quantity for {order_type}: {qty}")

            if qty:
                qty = self.calculate_quantity(symbol, order_type=order_type)
                logging.info(f"Validate Trade Calculated quantity for {order_type}: {qty}")


            # Calculate proposed trade value
            proposed_trade_value = float(qty) * current_price
            logging.info(f"\nTrade Details:")
            logging.info(f"- Current Price: ${current_price:,.2f}")
            logging.info(f"- Proposed Value: ${proposed_trade_value:,.2f}")

            # Get current position
            current_position = self.get_position(symbol)
            current_qty = float(current_position['qty']) if current_position else 0
            current_position_value = current_qty * current_price

            logging.info(f"\nCurrent Position:")
            logging.info(f"- Quantity: {current_qty}")
            logging.info(f"- Value: ${current_position_value:,.2f}")

            # Get total crypto exposure
            positions = self.api.list_positions()
            total_crypto_value = sum(
                float(p.market_value)
                for p in positions
                if p.symbol.endswith('USD')
            )
            max_crypto_allocation = float(self.risk_params['max_crypto_equity'])

            logging.info(f"\nExposure Analysis:")
            logging.info(f"- Total Crypto Value: ${total_crypto_value:,.2f}")
            logging.info(f"- Max Crypto Allocation: ${max_crypto_allocation:,.2f}")

            # SELL ORDER VALIDATION
            if order_type == 'sell':
                if current_position is None:
                    logging.info(f"No position found for {symbol}; cannot sell")
                    return False

                if qty > current_qty:
                    logging.info(f"Attempting to sell {qty} but only own {current_qty} of {symbol}")
                    return False

                # Add minimum transaction value check for sells
                if proposed_trade_value < self.min_transaction_value:
                    logging.info(f"\nSell value too small: ${proposed_trade_value:.2f}")
                    return False

                logging.info(f"\nSell Order Validation Passed:")
                logging.info(f"- Selling {qty} units of {symbol}")
                logging.info(f"- Expected Value: ${proposed_trade_value:,.2f}")

                # Track the sell order
                self.track_order(symbol, order_type, qty, current_price)
                return True

            # BUY ORDER VALIDATION
            else:
                if proposed_trade_value < self.min_transaction_value:
                    logging.info(f"\nTrade value too small: ${proposed_trade_value:.2f}")
                    return False

                if proposed_trade_value > account_cash:
                    logging.info(f"\nInsufficient cash: ${account_cash:.2f} for trade of ${proposed_trade_value:.2f}")
                    return False

                # Calculate new position details
                new_qty = current_qty + float(qty)
                new_position_value = new_qty * current_price
                new_total_exposure = total_crypto_value + proposed_trade_value

                logging.info(f"\nProjected Position:")
                logging.info(f"- New Quantity: {new_qty}")
                logging.info(f"- New Position Value: ${new_position_value:,.2f}")
                logging.info(f"- New Total Exposure: ${new_total_exposure:,.2f}")

                # Position size checks
                max_position_value = float(self.risk_params['max_position_size'])
                if new_position_value > max_position_value:
                    adjusted_qty = int(max_position_value / current_price) - current_qty
                    if adjusted_qty > 0:
                        logging.info(f"\nQuantity Adjustment (Position Size):")
                        logging.info(f"- Original Quantity: {qty}")
                        logging.info(f"- Adjusted Quantity: {adjusted_qty}")
                        qty = adjusted_qty
                    else:
                        logging.info("\nPosition size would exceed maximum allowed; rejecting buy")
                        return False

                # Risk per trade check
                max_risk_per_trade = float(self.risk_params['max_risk_per_trade'])
                max_trade_value = portfolio_value * max_risk_per_trade
                if proposed_trade_value > max_trade_value:
                    adjusted_qty = int(max_trade_value / current_price)
                    if adjusted_qty > 0:
                        logging.info(f"\nQuantity Adjustment (Risk Limit):")
                        logging.info(f"- Original Quantity: {qty}")
                        logging.info(f"- Adjusted Quantity: {adjusted_qty}")
                        qty = adjusted_qty
                    else:
                        logging.info("\nTrade value would exceed risk limit; rejecting buy")
                        return False

                # Crypto allocation check
                if symbol.endswith('USD'):
                    if new_total_exposure > max_crypto_allocation:
                        excess = new_total_exposure - max_crypto_allocation
                        logging.info(f"\nCrypto Allocation Exceeded:")
                        logging.info(f"- Excess Amount: ${excess:,.2f}")
                        logging.info(f"- Rejection Reason: Would exceed maximum crypto allocation")
                        return False

                # Track the order if all checks pass
                self.track_order(symbol, order_type, qty, current_price)
                logging.info("\nTrade validation successful")
                return True

        except Exception as e:
            logging.error(f"Error in trade validation: {e}")
            return False

        finally:
            logging.info("=== Trade Validation Complete ===\n")

    def _determine_asset_class(self, symbol):
        """Helper method to determine asset class from symbol"""
        if len(symbol) > 5 and symbol[0].isalpha() and symbol[-1].isdigit():
            return 'options'
        elif symbol.endswith('USD'):
            return 'crypto'
        elif 'commodity' in symbol.lower():
            return 'commodity'
        else:
            return 'equity'

    def _get_current_allocation(self, asset_class):
        """Helper method to get current allocation for an asset class"""
        positions = self.api.list_positions()
        total_value = sum(float(p.current_price) * float(p.qty) for p in positions
                          if self._determine_asset_class(p.symbol) == asset_class)
        return total_value

    def get_option_price_from_alpaca(self, symbol):
        """Enhanced option price fetching"""
        try:
            # First try getting the latest trade
            latest_trade = self.api.get_latest_trade(symbol)
            if latest_trade and hasattr(latest_trade, 'price'):
                return float(latest_trade.price)

            # Fallback to latest quote
            latest_quote = self.api.get_latest_quote(symbol)
            if latest_quote:
                # Use midpoint of bid/ask
                if hasattr(latest_quote, 'bid_price') and hasattr(latest_quote, 'ask_price'):
                    if latest_quote.bid_price > 0 and latest_quote.ask_price > 0:
                        return (float(latest_quote.bid_price) + float(latest_quote.ask_price)) / 2

            logging.warning(f"Could not get price for option {symbol}")
            return None

        except Exception as e:
            logging.error(f"Error getting option price for {symbol}: {e}")
            return None

    def monitor_account_status(self):
        # Monitor and report on account status
        try:
            account = self.api.get_account()
            print(f"Equity: {account.equity}")
            print(f"Cash: {account.cash}")
            print(f"Buying Power: {account.buying_power}")
            return account
        except Exception as e:
            print(f"An exception occurred while monitoring account status: {str(e)}")
            return None

    def monitor_positions(self):
        # Monitor and report on open positions
        try:
            positions = self.api.list_positions()
            for position in positions:
                pos_details = self.get_position(position.symbol)
                if pos_details:
                    print(
                        f"Symbol: {pos_details['symbol']}, Quantity: {pos_details['qty']}, Avg Entry Price: {pos_details['avg_entry_price']}")
            return positions
        except Exception as e:
            print(f"An exception occurred while monitoring positions: {str(e)}")
            return None

    def get_crypto_fee(self, volume):
        if volume < 100_000:
            return 0.0025
        elif volume < 500_000:
            return 0.0022
        elif volume < 1_000_000:
            return 0.002
        elif volume < 10_000_000:
            return 0.0018
        elif volume < 25_000_000:
            return 0.0015
        elif volume < 50_000_000:
            return 0.0013
        elif volume < 100_000_000:
            return 0.0012
        else:
            return 0.001

    def report_profit_and_loss(self):
        url = "https://paper-api.alpaca.markets/v2/account"
        url_portfolio_history = "https://paper-api.alpaca.markets/v2/account/portfolio/history"
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
        }

        try:
            # Get account data
            account_response = requests.get(url, headers=headers)
            account_data = account_response.json()
            cash_not_invested = float(account_data['cash'])

            # Get portfolio history data
            portfolio_history_response = requests.get(url_portfolio_history, headers=headers)
            portfolio_history_data = portfolio_history_response.json()

            # Filter out 'None' values
            equity_values = [v for v in portfolio_history_data['equity'] if v is not None]

            # Calculate PnL based on portfolio history
            first_equity = float(equity_values[0])  # First equity value
            last_equity = float(equity_values[-1])  # Last equity value
            commissions = first_equity * 0.01

            print(f'First equity is: {first_equity}.')
            print(f'Last equity is: {last_equity}.')
            print(f'Total commisions were: {commissions}.')

            # find pnl for account
            pnl_total = last_equity - first_equity - commissions

            # find total equity for reporting
            total_equity = pnl_total + cash_not_invested

            print(
                f"Total Profit/Loss: ${round(pnl_total,2)}. Total equity (cash invested plus cash not invested): ${round(total_equity,2)}")
            return pnl_total

        except Exception as e:
            print(f"An exception occurred while reporting profit and loss: {str(e)}")
            return 0

    def get_equity(self):
        return float(self.api.get_account().equity)

    def update_risk_parameters(self):
        # Get total PnL
        pnl_total = self.report_profit_and_loss()
        account = self.api.get_account()
        current_equity = float(account.equity)

        # Calculate PnL as a percentage of current equity
        pnl_percentage = pnl_total / current_equity if current_equity != 0 else 0

        # Define adjustment factor (e.g., adjust by 50% of PnL percentage)
        adjustment_factor = 1 + (pnl_percentage * 0.005)

        # Update risk parameters with boundaries
        min_position_size = self.risk_params.get('min_position_size', 50)
        max_position_size_limit = self.risk_params.get('max_position_size_limit', 10000)

        new_max_position_size = self.risk_params['max_position_size'] * adjustment_factor
        new_max_portfolio_size = self.risk_params['max_portfolio_size'] * adjustment_factor

        # Apply boundaries
        self.risk_params['max_position_size'] = max(min_position_size,
                                                    min(new_max_position_size, max_position_size_limit))
        self.risk_params['max_portfolio_size'] = max(current_equity * 0.05,
                                                     min(new_max_portfolio_size, current_equity * 2))

        # Save updated risk parameters
        with open('risk_params.json', 'w') as f:
            json.dump(self.risk_params, f)

        print("Risk parameters updated.")
        return self.risk_params

    def calculate_drawdown(self):
        try:
            portfolio = self.api.list_positions()
            portfolio_value = sum([float(position.current_price) * float(position.qty) for position in portfolio])

            # Update peak portfolio value if current portfolio value is higher
            if portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = portfolio_value

            # Calculate drawdown if portfolio is not empty
            if portfolio_value > 0 and self.peak_portfolio_value > 0:
                drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
            else:
                drawdown = 0

            return drawdown
        except Exception as e:
            print(f"An exception occurred while calculating drawdown: {str(e)}")
            return None

    def check_risk_before_order(self, symbol, new_shares):
        """
        Check the risk parameters before placing an order.

        The function will prevent an order if the new shares would result in a position size
        that violates the risk parameters.
        """
        # Get the current position
        try:
            current_position = self.api.get_position(symbol)
            current_shares = float(current_position.qty)
        except:
            current_shares = 0

        # Calculate the new quantity of shares after the purchase
        total_shares = current_shares + float(new_shares)

        # Check if the new quantity violates the risk parameters
        if total_shares > self.risk_params['max_position_size']:
            return 'Order exceeded permissible balance. Total order share exceed max position size allowable.'
            # If the new quantity violates the max position size, prevent the order
            return False
        else:
            # If the new quantity doesn't violate the risk parameters, adjust the quantity and place the order
            delta_shares = self.risk_params['max_position_size'] - current_shares

            if delta_shares > 0:
                # Get the average entry price
                avg_entry_price = self.get_avg_entry_price(symbol)

                if avg_entry_price is not None and avg_entry_price != 0:
                    # Calculate the adjusted quantity based on the average entry price
                    adjusted_quantity = int(delta_shares / avg_entry_price)

                    # Place the order with the adjusted quantity
                    self.api.submit_order(
                        symbol=symbol,
                        qty=adjusted_quantity,
                        side='buy',
                        type='limit',
                        time_in_force='gtc',
                        limit_price=avg_entry_price
                    )

            return True

    def check_momentum(self, symbol, momentum_signal):
        """
        Checks the momentum signal and decides whether to sell the entire position.
        Returns True if a sell order was placed, False otherwise.
        """
        alpaca_symbol = self.convert_symbol(symbol, to_alpaca=True)
        position_list = [position for position in self.api.list_positions() if position.symbol == alpaca_symbol]

        if len(position_list) == 0:
            print(f"No position exists for {symbol}.")
            return False

        position = position_list[0]

        if momentum_signal == "Sell" and float(position.unrealized_plpc) < 0:
            qty = position.qty
            if self.validate_trade(symbol, qty, "sell"):
                self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"Selling the entire position of {symbol} due to negative momentum.")
                return True

        return False

    def convert_symbol(self, symbol, to_alpaca=True, asset_type=None):
        """
        Convert between different symbol formats based on asset type.

        :param symbol: The symbol to convert
        :param to_alpaca: If True, convert to Alpaca format; if False, convert from Alpaca format
        :param asset_type: 'crypto', 'equity', 'option', or None (will try to infer)
        :return: Converted symbol
        """
        if asset_type is None:
            # Try to infer asset type
            if '/' in symbol:
                asset_type = 'crypto'
            elif len(symbol) > 15 and any(char.isdigit() for char in symbol):
                asset_type = 'option'
            else:
                asset_type = 'equity'

        if asset_type == 'crypto':
            if to_alpaca:
                return symbol.replace('/USD', 'USD').replace('/', '')
            else:
                if 'USD' in symbol:
                    return f"{symbol[:-3]}/USD"
                else:
                    return symbol  # Assume it's already in the correct format
        elif asset_type == 'option':
            # Options symbols are the same in Alpaca and elsewhere
            return symbol
        elif asset_type == 'equity':
            # Equity symbols are the same in Alpaca and elsewhere
            return symbol
        else:
            raise ValueError(f"Unknown asset type: {asset_type}")

    def base_quantity_calculation(self, symbol, order_type, asset_type):
        # Retrieve account details to get portfolio and cash values
        account = self.api.get_account()
        portfolio_value = float(account.portfolio_value)
        account_cash = float(account.cash)

        # Get current price for the symbol
        current_price = self.get_current_price(symbol)
        if not current_price or current_price <= 0:
            logging.error(f"Invalid price for {symbol}: {current_price}")
            return 0

        # Adjust base size for market regime
        base_size = self.MIN_TRANSACTION_VALUE / current_price
        adjusted_size = self.adjust_position_size_for_regime(symbol, base_size)

        # Calculate potential quantity based on available cash and adjusted size
        potential_qty = account_cash / current_price if order_type == 'buy' else adjusted_size

        # Set minimum quantity based on MIN_TRANSACTION_VALUE
        min_qty = max(self.MIN_TRANSACTION_VALUE / current_price, 1)

        # Respect portfolio allocations for asset type
        if symbol.endswith('USD'):
            max_crypto_allocation = float(self.risk_params['max_crypto_equity'])
            print(f'Here is max crypto allocation: {max_crypto_allocation}')
            current_crypto_value = sum(
                float(p.market_value) for p in self.api.list_positions() if p.symbol.endswith('USD')
            )
            print(f'Here is current crypto value in port: {current_crypto_value} and order type {order_type}')
            if order_type == 'buy' and (current_crypto_value + (potential_qty * current_price) > max_crypto_allocation):
                logging.info("Trade would exceed maximum crypto allocation; adjusting quantity")
                potential_qty = (max_crypto_allocation - current_crypto_value) / current_price

        # Ensure the final quantity meets minimum and adjusted transaction constraints
        return max(min_qty, int(potential_qty))

    def calculate_quantity(self, symbol, order_type='buy', asset_type=None):
        """
        Enhanced quantity calculation with sophisticated risk controls and position sizing.
        """
        try:
            # Get current price with validation
            current_price = self.get_current_price(symbol)
            if not current_price or current_price <= 0:
                logging.error(f"Invalid price for {symbol}: {current_price}")
                return 0

            # Get account info
            account = self.api.get_account()
            portfolio_value = float(account.portfolio_value)
            available_cash = float(account.cash)

            # Get current position if exists
            try:
                position = self.api.get_position(symbol)
                current_position_qty = float(position.qty)
                current_position_value = current_position_qty * current_price
            except Exception:
                current_position_qty = 0
                current_position_value = 0

            # Price-based quantity tiers
            quantity_tiers = {
                'micro': {'max_price': 1, 'max_quantity': 100000},
                'small': {'max_price': 10, 'max_quantity': 10000},
                'medium': {'max_price': 100, 'max_quantity': 1000},
                'large': {'max_price': 1000, 'max_quantity': 100},
                'xlarge': {'max_price': float('inf'), 'max_quantity': 10}
            }

            # Determine tier based on price
            selected_tier = None
            for tier, params in quantity_tiers.items():
                if current_price <= params['max_price']:
                    selected_tier = params
                    break
            if not selected_tier:
                selected_tier = quantity_tiers['xlarge']

            # Handle sell orders
            if order_type.lower() == 'sell':
                if current_position_qty == 0:
                    logging.info(f"No position to sell for {symbol}")
                    return 0
                # For sells, return the entire position quantity
                return current_position_qty

            # Calculate maximum allocation for asset type
            if symbol.endswith('USD'):
                max_allocation = float(self.risk_params['max_crypto_equity'])
                current_crypto_value = sum(
                    float(p.market_value) for p in self.api.list_positions()
                    if p.symbol.endswith('USD')
                )
                available_allocation = max_allocation - current_crypto_value
            else:
                max_allocation = float(self.risk_params['max_equity_equity'])
                available_allocation = max_allocation

            # Calculate base quantity from the smaller of:
            # 1. Available cash
            # 2. Available allocation
            # 3. Tier maximum
            max_qty_by_cash = available_cash / current_price
            max_qty_by_allocation = available_allocation / current_price
            max_qty_by_tier = selected_tier['max_quantity']

            base_qty = min(max_qty_by_cash, max_qty_by_allocation, max_qty_by_tier)

            # Apply position value constraints
            max_position_value = float(self.risk_params['max_position_size'])
            if (base_qty * current_price) > max_position_value:
                base_qty = max_position_value / current_price

            # Apply risk per trade constraint
            max_risk_per_trade = float(self.risk_params['max_risk_per_trade'])
            max_trade_value = portfolio_value * max_risk_per_trade
            if (base_qty * current_price) > max_trade_value:
                base_qty = max_trade_value / current_price

            # Apply minimum transaction value
            min_transaction_qty = self.MIN_TRANSACTION_VALUE / current_price
            base_qty = max(min_transaction_qty, base_qty)

            # Asset-specific minimum quantities
            min_quantities = {
                'BTCUSD': 0.001,
                'ETHUSD': 0.01,
                'DOGEUSD': 1.0,
                'default': 1.0
            }
            min_required = min_quantities.get(symbol, min_quantities['default'])
            final_qty = max(min_required, base_qty)

            # Market impact adjustment
            volume = self.get_recent_volumes(symbol, 1)[0] if hasattr(self, 'get_recent_volumes') else None
            if volume:
                max_volume_ratio = 0.01  # Maximum 1% of volume
                volume_constrained_qty = volume * max_volume_ratio
                final_qty = min(final_qty, volume_constrained_qty)

            # Final rounding based on asset type
            if symbol.endswith('USD'):
                if 'BTC' in symbol:
                    final_qty = round(final_qty, 3)  # 0.001 precision for BTC
                elif 'ETH' in symbol:
                    final_qty = round(final_qty, 2)  # 0.01 precision for ETH
                else:
                    final_qty = int(final_qty)  # Integer quantities for other crypto
            else:
                final_qty = int(final_qty)  # Integer quantities for non-crypto

            logging.info(f"""
            Quantity calculation for {symbol}:
            - Current Price: ${current_price:.2f}
            - Max by Cash: {max_qty_by_cash:.2f}
            - Max by Allocation: {max_qty_by_allocation:.2f}
            - Max by Tier: {max_qty_by_tier}
            - Final Quantity: {final_qty}
            """)

            return final_qty

        except Exception as e:
            logging.error(f"Error calculating quantity: {e}")
            return 0

    def get_current_price(self, symbol):
        """
        Get current price for any asset type with comprehensive error handling
        """
        try:
            # Handle options contracts
            if 'OPT' in symbol:
                price = self.get_option_price_from_alpaca(symbol)
                if price is not None:
                    return price
                logging.warning(f"Could not get option price for {symbol}, falling back to last trade")
                return None

            # Handle regular equities
            if not symbol.endswith('USD'):
                try:
                    bar = self.api.get_latest_bar(symbol)
                    if bar and hasattr(bar, 'c'):
                        return float(bar.c)
                    logging.warning(f"No latest bar data for {symbol}")
                    return None
                except Exception as e:
                    logging.error(f"Error getting equity price for {symbol}: {e}")
                    return None

            # Handle crypto
            try:
                base_crypto = symbol[:-3] if not '/' in symbol else symbol.split('/')[0]

                # First try digital currency endpoint
                url = (
                    f"https://www.alphavantage.co/query"
                    f"?function=DIGITAL_CURRENCY_DAILY"
                    f"&symbol={base_crypto}"
                    f"&market=USD"
                    f"&apikey={self.ALPHA_VANTAGE_API}"
                )

                response = requests.get(url)
                data = response.json()

                if "Digital Currency Daily" in data:
                    latest_day = list(data["Digital Currency Daily"].keys())[0]
                    price = float(data["Digital Currency Daily"][latest_day]["4a. close (USD)"])

                    # Basic sanity check
                    if price > 0 and price < 1_000_000:
                        return price
                    logging.warning(f"Price sanity check failed for {symbol}: {price}")

                # Fallback to exchange rate endpoint
                url = (
                    f"https://www.alphavantage.co/query"
                    f"?function=CURRENCY_EXCHANGE_RATE"
                    f"&from_currency={base_crypto}"
                    f"&to_currency=USD"
                    f"&apikey={self.ALPHA_VANTAGE_API}"
                )

                response = requests.get(url)
                data = response.json()

                if "Realtime Currency Exchange Rate" in data:
                    price = float(data["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
                    if price > 0:
                        return price

                # Final fallback - try Alpaca crypto endpoint
                try:
                    quote = self.api.get_latest_crypto_quote(symbol)
                    if quote and hasattr(quote, 'ap'):
                        return float(quote.ap)
                except Exception as e:
                    logging.error(f"Alpaca crypto quote failed for {symbol}: {e}")

                logging.error(f"All price fetching methods failed for {symbol}")
                return None

            except Exception as e:
                logging.error(f"Error in crypto price fetch for {symbol}: {e}")
                return None

        except Exception as e:
            logging.error(f"Error in get_current_price for {symbol}: {e}")
            return Non

    def apply_crypto_tiers(self, quantity, current_price):
        if current_price > 4001:
            return quantity * 0.01
        elif 3001 < current_price <= 4000:
            return quantity * 0.0354
        elif 1000 < current_price <= 3000:
            return quantity * 0.0334
        elif 201 < current_price <= 999:
            return quantity * 0.04534
        elif 20 < current_price <= 200:
            return quantity * 0.09434
        elif -0.000714 < current_price <= 20.00:
            return quantity * 0.031434
        else:
            return quantity

    def execute_profit_taking(self, symbol, base_pct_gain=0.015):
        """Enhanced profit taking with dynamic thresholds"""
        try:
            position = self.get_position(symbol)
            if not position:
                return

            current_price = self.get_current_price(symbol)
            volume = self.get_recent_volumes(symbol, 1)[0]  # Get most recent volume
            self.update_price_history(symbol, current_price, volume)  # Track metrics

            avg_entry = float(position.avg_entry_price)
            unrealized_gain = (current_price - avg_entry) / avg_entry

            # Calculate dynamic threshold
            volatility = self.calculate_volatility(symbol)
            vol_adjusted_threshold = base_pct_gain * (1 + volatility)

            # Volume check
            volume_ratio = self.get_volume_ratio(symbol)
            if volume_ratio < 0.7:
                vol_adjusted_threshold *= 1.2

            qty = float(position.qty)
            if unrealized_gain > vol_adjusted_threshold:
                sell_ratio = min(0.5, unrealized_gain / (2 * vol_adjusted_threshold))
                sell_qty = int(qty * sell_ratio)

                if self.validate_trade(symbol, sell_qty, "sell"):
                    self.api.submit_order(
                        symbol=symbol,
                        qty=sell_qty,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                    logging.info(f"Profit taking executed for {symbol}: {sell_qty} units at {unrealized_gain:.2%} gain")

        except Exception as e:
            logging.error(f"Error in enhanced profit taking for {symbol}: {e}")

    def execute_stop_loss(self, symbol, base_pct_loss=0.01):
        """Enhanced stop loss with dynamic thresholds"""
        try:
            position = self.get_position(symbol)
            if not position:
                return

            current_price = self.get_current_price(symbol)
            volume = self.get_recent_volumes(symbol, 1)[0]
            self.update_price_history(symbol, current_price, volume)

            avg_entry = float(position.avg_entry_price)
            unrealized_loss = (current_price - avg_entry) / avg_entry

            volatility = self.calculate_volatility(symbol)
            vol_adjusted_threshold = base_pct_loss * (1 + volatility)

            volume_ratio = self.get_volume_ratio(symbol)
            position_value = float(position.qty) * current_price

            if unrealized_loss < -vol_adjusted_threshold:
                exit_ratio = 1.0 if volume_ratio > 1.5 else min(0.5, volume_ratio / 2)
                sell_qty = int(float(position.qty) * exit_ratio)

                if self.validate_trade(symbol, sell_qty, "sell"):
                    self.api.submit_order(
                        symbol=symbol,
                        qty=sell_qty,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                    logging.info(f"Stop loss executed for {symbol}: {sell_qty} units at {unrealized_loss:.2%} loss")

        except Exception as e:
            logging.error(f"Error in enhanced stop loss for {symbol}: {e}")

    def calculate_position_risk(self, symbol):
        """Calculate risk-adjusted position size"""
        try:
            current_price = self.get_current_price(symbol)
            volume = self.get_recent_volumes(symbol, 1)[0]
            self.update_price_history(symbol, current_price, volume)

            volatility = self.calculate_volatility(symbol)
            volume_ratio = self.get_volume_ratio(symbol)

            account = self.api.get_account()
            base_size = float(account.equity) * float(self.risk_params['max_risk_per_trade'])

            vol_adjustment = 1 / (1 + volatility)
            vol_scalar = min(1.0, volume_ratio / 2)

            adjusted_size = base_size * vol_adjustment * vol_scalar

            return min(adjusted_size, float(self.risk_params['max_position_size']))

        except Exception as e:
            logging.error(f"Error calculating position risk for {symbol}: {e}")
            return 0

    def update_price_history(self, symbol, price, volume):
        """Updates price and volume history for a symbol"""
        try:
            if not hasattr(self, 'market_metrics'):
                self.market_metrics = {}
            if symbol not in self.market_metrics:
                self.market_metrics[symbol] = self.MarketMetrics()
            self.market_metrics[symbol].update(price, volume)
        except Exception as e:
            logging.error(f"Error updating price history for {symbol}: {e}")

    def get_recent_volumes(self, symbol, window=20):
        """Fetches recent trading volumes"""
        try:
            url = f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol.split('/')[0]}&market=USD&apikey={ALPHA_VANTAGE_API}"
            response = requests.get(url)
            data = response.json()

            if "Time Series (Digital Currency Daily)" in data:
                volumes = []
                for date in list(data["Time Series (Digital Currency Daily)"].keys())[:window]:
                    volume = float(data["Time Series (Digital Currency Daily)"][date]["5. volume"])
                    volumes.append(volume)
                return volumes
            return [0] * window
        except Exception as e:
            logging.error(f"Error fetching volumes for {symbol}: {e}")
            return [0] * window

    def get_recent_prices(self, symbol, window=20):
        """Fetches recent prices"""
        try:
            url = f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol.split('/')[0]}&market=USD&apikey={ALPHA_VANTAGE_API}"
            response = requests.get(url)
            data = response.json()

            if "Time Series (Digital Currency Daily)" in data:
                prices = []
                for date in list(data["Time Series (Digital Currency Daily)"].keys())[:window]:
                    price = float(data["Time Series (Digital Currency Daily)"][date]["4a. close (USD)"])
                    prices.append(price)
                return prices
            return None
        except Exception as e:
            logging.error(f"Error fetching prices for {symbol}: {e}")
            return None

    def calculate_volatility(self, symbol, window=14):
        """Calculate recent price volatility using ATR or standard deviation"""
        try:
            # Get recent price data
            prices = self.get_recent_prices(symbol, window + 1)
            if not prices or len(prices) < window:
                return 0

            # Calculate daily returns
            returns = np.diff(np.log(prices))
            return np.std(returns) * np.sqrt(252)  # Annualized volatility

        except Exception as e:
            logging.error(f"Error calculating volatility for {symbol}: {str(e)}")
            return 0

    def get_average_volatility(self, symbol, lookback=30):
        """Get the average historical volatility"""
        try:
            prices = self.get_recent_prices(symbol, lookback + 1)
            if not prices or len(prices) < lookback:
                return 0

            returns = np.diff(np.log(prices))
            return np.mean([np.std(returns[i:i + 14]) * np.sqrt(252)
                            for i in range(len(returns) - 14)])

        except Exception as e:
            logging.error(f"Error calculating average volatility for {symbol}: {str(e)}")
            return 0

    def get_volume_ratio(self, symbol, window=20):
        """Calculate current volume relative to average"""
        try:
            volumes = self.get_recent_volumes(symbol, window)
            if not volumes or len(volumes) < window:
                return 1

            current_volume = volumes[-1]
            avg_volume = np.mean(volumes[:-1])
            return current_volume / avg_volume if avg_volume > 0 else 1

        except Exception as e:
            logging.error(f"Error calculating volume ratio for {symbol}: {str(e)}")
            return 1

    def get_max_profit_reached(self, symbol):
        """Get the maximum profit percentage reached for this position"""
        try:
            position = self.api.get_position(symbol)
            cost_basis = float(position.avg_entry_price)
            high_price = float(position.highest_price_since_entry)
            return (high_price - cost_basis) / cost_basis if cost_basis > 0 else 0

        except Exception as e:
            logging.error(f"Error getting max profit for {symbol}: {str(e)}")
            return 0

    def record_trade_exit(self, symbol, exit_type, loss_pct):
        """Record trade exit details for analysis"""
        try:
            exit_record = {
                'symbol': symbol,
                'exit_type': exit_type,
                'loss_percentage': loss_pct,
                'timestamp': datetime.utcnow(),
                'market_conditions': {
                    'volatility': self.calculate_volatility(symbol),
                    'volume_ratio': self.get_volume_ratio(symbol)
                }
            }

            # Store in MongoDB for analysis
            self.db.trade_exits.insert_one(exit_record)

        except Exception as e:
            logging.error(f"Error recording trade exit for {symbol}: {str(e)}")

    @staticmethod
    def calculate_profitability(current_price, avg_entry_price):
        """
        Calculate profitability percentage between current and entry price
        with improved price validation and error handling.

        Args:
            current_price: Current price of the asset
            avg_entry_price: Average entry price of the position

        Returns:
            float: Profitability percentage. Returns actual calculation if possible,
                  returns -100.0 for losses requiring exit, returns 0.0 if calculation impossible
        """
        try:
            logging.info(f"Calculating profitability - Current: {current_price}, Entry: {avg_entry_price}")

            # Special case: If we have an entry price but no current price,
            # return a value that will trigger exit (assume significant loss)
            if avg_entry_price and (current_price is None or current_price == 0):
                logging.warning(
                    f"Missing current price with valid entry price ({avg_entry_price}). "
                    "Assuming significant loss position."
                )
                return -100.0  # This will trigger position exit

            # Special case: If we have a current price but no entry price,
            # use current price as entry price (conservative approach)
            if current_price and (avg_entry_price is None or avg_entry_price == 0):
                logging.warning(
                    f"Missing entry price with valid current price ({current_price}). "
                    "Using current price as entry price."
                )
                avg_entry_price = current_price

            # If both prices are missing or zero, we can't calculate
            if (not current_price and not avg_entry_price) or \
                    (current_price == 0 and avg_entry_price == 0):
                logging.warning("Both prices missing or zero - cannot calculate profitability")
                return 0.0

            # Convert to float if needed
            try:
                if isinstance(current_price, str):
                    current_price = float(current_price)
                if isinstance(avg_entry_price, str):
                    avg_entry_price = float(avg_entry_price)
            except (ValueError, TypeError) as e:
                logging.error(f"Error converting prices to float: {e}")
                return -100.0  # Error in price conversion, trigger exit

            # Type checking
            if not isinstance(current_price, (int, float)) or not isinstance(avg_entry_price, (int, float)):
                logging.error(
                    f"Invalid price types: current_price={type(current_price)}, "
                    f"avg_entry_price={type(avg_entry_price)}"
                )
                return -100.0  # Invalid types, trigger exit

            # Calculate profitability
            if avg_entry_price > 0:  # Normal case
                profitability = ((current_price - avg_entry_price) / avg_entry_price) * 100
            else:  # Edge case where entry price is 0
                profitability = 0.0 if current_price == 0 else 100.0

            # Log result
            logging.info(f"Calculated profitability: {profitability:.2f}%")

            # Sanity check on result
            if abs(profitability) > 1000:
                logging.warning(
                    f"Extreme profitability detected ({profitability:.2f}%). "
                    f"Prices: current={current_price}, entry={avg_entry_price}"
                )

            return float(profitability)

        except ZeroDivisionError:
            logging.error("Zero division error in profitability calculation")
            return -100.0  # Division error, trigger exit
        except Exception as e:
            logging.error(f"Error calculating profitability: {e}")
            return -100.0  # Unknown error, trigger exit

    def enforce_diversification(self, symbol, max_pct_portfolio=0.30):
        """
        Enforces diversification by ensuring that no crypto makes up more than a certain percentage of the portfolio.
        """
        alpaca_symbol = self.convert_symbol(symbol, to_alpaca=True)
        portfolio = self.api.list_positions()
        position_list = [position for position in portfolio if position.symbol == alpaca_symbol]

        if len(position_list) == 0:
            print(f"No position exists for {symbol}.")
            return

        position = position_list[0]
        position_value = float(position.current_price) * float(position.qty)

        # If the value of this position exceeds the maximum percentage of the portfolio, sell enough shares to get below the maximum
        if position_value / portfolio_value > max_pct_portfolio:
            excess_value = position_value - (portfolio_value * max_pct_portfolio)
            qty_to_sell = int(excess_value / float(position.current_price))

            if self.validate_trade(symbol, qty_to_sell, "sell"):
                self.api.submit_order(
                    symbol=symbol,
                    qty=qty_to_sell,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"Selling {qty_to_sell} shares of {symbol} to maintain diversification.")

    def generate_momentum_signal(self, symbol):
        """
        Generate a momentum signal for the given symbol.

        Returns "Buy" if the symbol has increased in value by 1.5% or more since purchase,
        and "Sell" if it has decreased by 0.98% or more. Otherwise, returns "Hold".
        """
        # Get the purchase price for this stock
        purchase_price = self.get_purchase_price(symbol)

        # Get the current price for this stock
        current_price = self.get_avg_entry_price(symbol)

        # Calculate the percentage change since purchase
        pct_change = (current_price - purchase_price) / purchase_price * 100

        # Generate the momentum signal
        if pct_change >= 1.5:
            return "Buy"
        elif pct_change <= -0.98:
            return "Sell"
        else:
            return "Hold"



    def get_purchase_price(self, symbol):
        """
        Retrieve the purchase price of the given symbol.
        """
        trades = download_trades()

        # Filter trades for the given symbol
        trades = [trade for trade in trades if trade[0] == symbol]

        if not trades:
            return None

        # Get the last trade for the symbol
        last_trade = trades[-1]

        # The price is the third element in the trade
        return float(last_trade[2])

    def get_avg_entry_price(self, symbol):
        try:
            position = self.api.get_position(symbol)
            avg_entry_price = float(position.avg_entry_price)
            print(f"For symbol {symbol}, average entry price is {avg_entry_price}.")
            return avg_entry_price
        except Exception as e:
            print(f"No position in {symbol} to calculate average entry price. Error: {str(e)}")
            return 0

    ## this section provides logic to pull the latest price of something and use that
    @staticmethod
    def fetch_exchange_rate(base_currency, quote_currency):
        url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={base_currency}&to_currency={quote_currency}&apikey={ALPHA_VANTAGE_API}"
        try:
            response = requests.get(url).json()
            exchange_rate = response["Realtime Currency Exchange Rate"]["5. Exchange Rate"]
            return float(exchange_rate)
        except KeyError:
            print(f"No exchange rate found from {base_currency} to {quote_currency}")
            return None
        except Exception as e:
            print(f"Error while fetching exchange rate: {e}")
            return None

    @staticmethod
    def fetch_and_process_data(base_crypto, quote='USD'):
        exchange_rate = RiskManagement.fetch_exchange_rate(base_crypto, quote)

        if exchange_rate is None:
            print(f"No exchange rate found for {base_crypto} to {quote}")
            return None

        url = f"https://www.alphavantage.co/query?function=CRYPTO_INTRADAY&symbol={base_crypto}&market={quote}&interval=5min&outputsize=compact&apikey={ALPHA_VANTAGE_API}"
        try:
            response = requests.get(url).json()
            intraday_data = response.get('Time Series Crypto (5min)', {})
            if not intraday_data:
                print(f"No intraday data found for {base_crypto}/{quote}")
                return None

            latest_intraday_data = list(intraday_data.items())[0]  # Get the latest 5-minute interval
            latest_price = float(latest_intraday_data[1]['4. close']) * exchange_rate
            return latest_price
        except Exception as e:
            print(f"Error while fetching intraday stats: {e}")
            return None


    def analyze_trend(self, symbol):
        try:
            # Fetch historical data
            data, _ = alpha_vantage_ts.get_daily(symbol=symbol, outputsize='compact')

            # Calculate moving averages
            data['SMA_20'] = data['4. close'].rolling(window=20).mean()
            data['SMA_50'] = data['4. close'].rolling(window=50).mean()

            # Determine trend
            if data['SMA_20'].iloc[-1] > data['SMA_50'].iloc[-1]:
                return 'uptrend'
            elif data['SMA_20'].iloc[-1] < data['SMA_50'].iloc[-1]:
                return 'downtrend'
            else:
                return 'neutral'
        except Exception as e:
            logging.error(f"Error analyzing trend for {symbol}: {e}")
            return 'neutral'

    def monitor_and_close_expiring_options(self):
        try:
            positions = self.api.list_positions()
            if not positions:
                logging.info("No positions to monitor.")
                return

            current_date = datetime.now().date()

            for position in positions:
                if 'OPT' in position.symbol:
                    expiration_date = datetime.strptime(position.expiration_date, '%Y-%m-%d').date()
                    days_to_expiration = (expiration_date - current_date).days

                    if days_to_expiration <= 7:
                        # Analyze the trend
                        trend = self.analyze_trend(
                            position.symbol.replace('OPT', ''))  # Adjust symbol for trend analysis

                        # Decision based on trend
                        if trend == 'downtrend':
                            # Close options position
                            qty = position.qty
                            self.api.submit_order(
                                symbol=position.symbol,
                                qty=qty,
                                side='sell',
                                type='market',
                                time_in_force='gtc'
                            )
                            logging.info(
                                f"Closed options position {position.symbol} due to approaching expiration and downtrend.")
                            print(
                                f"Closed options position {position.symbol} due to approaching expiration and downtrend.")
                        else:
                            logging.info(f"Decided not to sell {position.symbol} due to trend: {trend}")
                            print(f"Decided not to sell {position.symbol} due to trend: {trend}")
        except Exception as e:
            logging.error(f"Error in monitor_and_close_expiring_options: {e}")
            print(f"Error in monitor_and_close_expiring_options: {e}")



    TARGET_ALLOCATION = {
        'options': 0.20,  # 20%
        'crypto': 0.30,  # 30%
        'equities': 0.50  # 50%
    }


def get_alpha_vantage_data(base_currency, quote_currency):
    url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={base_currency}&to_currency={quote_currency}&apikey={ALPHA_VANTAGE_API}"

    response = requests.get(url)
    data = response.json()

    if "Realtime Currency Exchange Rate" in data:
        # Get the exchange rate
        exchange_rate = data["Realtime Currency Exchange Rate"]["5. Exchange Rate"]
        return exchange_rate
    else:
        print("Error getting data from Alpha Vantage")
        return None

def send_teams_message(teams_url, message):
    headers = {
        "Content-type": "application/json",
    }
    response = requests.post(teams_url, headers=headers, data=json.dumps(message))
    return response.status_code


def get_profit_loss(positions):
    profit_loss = 0
    for position in positions:
        profit_loss += (position['current_price'] - position['avg_entry_price']) * float(position['quantity'])
    return profit_loss



facts = []

if __name__ == "__main__":
    try:
        risk_manager = RiskManagement(api, risk_params)

        # Initialize portfolio metrics
        print("\n=== Portfolio Initialization ===")
        account = risk_manager.api.get_account()
        portfolio = risk_manager.api.list_positions()

        # Calculate and log initial allocations
        initial_allocation = risk_manager.calculate_current_allocation()
        print("\nInitial Portfolio Allocation:")
        for asset_class, allocation in initial_allocation.items():
            print(f"{asset_class}: {allocation:.2%}")

        # Perform rebalancing with enhanced monitoring
        print("\n=== Portfolio Rebalancing ===")
        try:
            risk_manager.rebalance_portfolio()
        except Exception as e:
            logging.error(f"Rebalancing error: {e}")
            print(f"Warning: Portfolio rebalancing encountered an error: {e}")

        # Monitor options with improved error handling
        print("\n=== Options Monitoring ===")
        try:
            risk_manager.monitor_and_close_expiring_options()
        except Exception as e:
            logging.error(f"Options monitoring error: {e}")
            print(f"Warning: Options monitoring encountered an error: {e}")

        # Refresh account and position data after rebalancing
        account = risk_manager.api.get_account()
        portfolio = risk_manager.api.list_positions()

        # Enhanced portfolio summary
        portfolio_summary = {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'positions': [],
            'market_regimes': {},
            'correlations': None,
            'asset_class_values': {
                'crypto': 0.0,
                'options': 0.0,
                'equities': 0.0
            }
        }

        print("\n=== Position Analysis ===")
        # Enhanced position analysis with error handling
        for position in portfolio:
            try:
                symbol = position.symbol
                avg_entry_price = float(position.avg_entry_price) if position.avg_entry_price else 0.0
                current_price = risk_manager.get_current_price(symbol)
                qty = float(position.qty) if position.qty else 0.0

                # Determine asset class
                asset_class = 'options' if 'OPT' in symbol else 'crypto' if symbol.endswith('USD') else 'equities'
                position_value = (current_price * qty) if current_price else 0.0
                portfolio_summary['asset_class_values'][asset_class] += position_value

                # Get market regime and volatility
                try:
                    regime = risk_manager.detect_market_regime(symbol)
                    volatility = risk_manager.calculate_volatility(symbol)
                except Exception as e:
                    logging.warning(f"Could not calculate metrics for {symbol}: {e}")
                    regime = 'unknown'
                    volatility = 0.0

                portfolio_summary['market_regimes'][symbol] = regime

                # Calculate position metrics
                profitability = risk_manager.calculate_profitability(current_price, avg_entry_price)

                position_data = {
                    'symbol': symbol,
                    'asset_class': asset_class,
                    'avg_entry_price': avg_entry_price,
                    'current_price': current_price,
                    'profitability': float(profitability) if profitability is not None else 0.0,
                    'quantity': qty,
                    'market_regime': regime,
                    'volatility': volatility,
                    'value': position_value
                }

                portfolio_summary['positions'].append(position_data)

            except Exception as e:
                logging.error(f"Error analyzing position {position.symbol}: {e}")
                continue

        # Calculate correlations only for relevant assets
        valid_positions = [p for p in portfolio_summary['positions']
                           if p['current_price'] is not None and p['current_price'] > 0]
        if len(valid_positions) >= 2:
            try:
                symbols = [pos['symbol'] for pos in valid_positions]
                portfolio_summary['correlations'] = risk_manager.calculate_asset_correlations(symbols)
            except Exception as e:
                logging.error(f"Error calculating correlations: {e}")

        # Calculate total P/L
        portfolio_summary['profit_loss'] = sum(
            (pos['current_price'] - pos['avg_entry_price']) * pos['quantity']
            for pos in portfolio_summary['positions']
            if pos['current_price'] is not None and pos['avg_entry_price'] is not None
        )

        # Update risk parameters
        risk_manager.update_risk_parameters()

        # Print comprehensive portfolio summary
        print("\n=== Portfolio Summary ===")
        print(f"Total Equity: ${portfolio_summary['equity']:,.2f}")
        print(f"Cash: ${portfolio_summary['cash']:,.2f}")
        print(f"Total P/L: ${portfolio_summary['profit_loss']:,.2f}")

        print("\nAsset Class Allocation:")
        total_portfolio_value = sum(portfolio_summary['asset_class_values'].values())
        for asset_class, value in portfolio_summary['asset_class_values'].items():
            allocation = value / total_portfolio_value if total_portfolio_value > 0 else 0
            print(f"{asset_class}: ${value:,.2f} ({allocation:.2%})")

        print("\nPosition Details:")
        for pos in sorted(portfolio_summary['positions'], key=lambda x: -x['value']):
            print(f"{pos['symbol']} ({pos['asset_class']}): "
                  f"Value: ${pos['value']:,.2f}, "
                  f"Regime: {pos['market_regime']}, "
                  f"Vol: {pos['volatility']:.2%}, "
                  f"P/L: {pos['profitability']:.2%}")

        # Log high correlations if any
        if portfolio_summary['correlations'] is not None and not portfolio_summary['correlations'].empty:
            high_corr_pairs = []
            corr_matrix = portfolio_summary['correlations']
            for i in range(len(corr_matrix.index)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > risk_manager.correlation_threshold:
                        high_corr_pairs.append((
                            corr_matrix.index[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j]
                        ))

            if high_corr_pairs:
                print("\nHighly Correlated Pairs:")
                for sym1, sym2, corr in high_corr_pairs:
                    print(f"{sym1} - {sym2}: {corr:.2f}")

    except Exception as e:
        logging.error(f"Critical error in main execution: {e}")
        print(f"Critical error occurred: {e}")