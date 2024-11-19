import os
import json
from dotenv import load_dotenv
import numpy as np
from scipy.stats import norm
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor, as_completed
import alpaca_trade_api as tradeapi
import requests
import random
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time
import traceback
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from datetime import datetime, timedelta
import logging
from typing import Optional
import pymongo


# Setup logging
logging.basicConfig(filename='logfile.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Alpaca API setup
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url='https://paper-api.alpaca.markets')

# Add the parent directory containing `risk_strategy.py` to the Python path
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import risk strategy components
try:
    from risk_strategy_options import OptionsRiskManagement, risk_params
    # Initialize risk management
    rm = OptionsRiskManagement(api, risk_params)
    #logging.debug("Risk strategy module loaded successfully.")
except ModuleNotFoundError as e:
    logging.error(f"Module not found: {e}")
    print(f"Module not found: {e}")
    sys.exit(1)

# Get the path to the risk_params.json file
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
risk_params_path = os.path.join(root_dir, 'risk_params.json')

# Database configuration
MONGO_DB_CONN_STRING = os.getenv("MONGO_DB_CONN_STRING")

# Initialize MongoDB client
try:
    mongo_client = MongoClient(MONGO_DB_CONN_STRING)
    #logging.info("MongoDB client initialized successfully.")
    #logging.debug("MongoDB client initialized successfully.")
except Exception as e:
    logging.info(f"Failed to initialize MongoDB client: {e}")
    logging.error(f"Failed to initialize MongoDB client: {e}")
    sys.exit(1)

# Initialize MongoDB collections
stock_data_db = mongo_client["stock_data"]
options_contracts_collection = stock_data_db["options_contracts"]

# Load risk parameters from JSON file
try:
    with open(risk_params_path, 'r') as f:
        risk_params = json.load(f)

    # Portfolio and risk configuration
    portfolio_balance = risk_params.get('max_portfolio_size')
    maximum_risk_per_trade = risk_params.get('max_risk_per_trade')

    logging.info(f"Loaded portfolio balance: {portfolio_balance}")
    logging.info(f"Loaded maximum risk per trade: {maximum_risk_per_trade}")
except FileNotFoundError:
    logging.error(f"risk_params.json not found at {risk_params_path}")
    sys.exit(1)
except json.JSONDecodeError:
    logging.error(f"Error decoding risk_params.json")
    sys.exit(1)


def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2)


def calculate_greeks(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return {
        'delta': norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1),
        'gamma': norm.pdf(d1) / (S * sigma * np.sqrt(T)),
        'theta': -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2) if option_type == "call" else -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2),
        'vega': S * norm.pdf(d1) * np.sqrt(T)
    }


def get_predicted_prices(mongo_client):
    """Fetch the latest predicted prices for each symbol from the MongoDB collection."""
    current_month_year = datetime.now().strftime("%B_%Y")
    predictions_db = mongo_client["predictions"]
    predictions_collection = predictions_db[current_month_year]

    try:
        pipeline = [
            {"$sort": {"entry_date": -1}},
            {"$group": {
                "_id": "$symbol",
                "symbol": {"$first": "$symbol"},
                "predicted_price": {"$first": "$predicted_price"},
                "prediction_date": {"$first": "$prediction_date"},
                "entry_date": {"$first": "$entry_date"}
            }}
        ]
        predictions = list(predictions_collection.aggregate(pipeline))
        logging.debug(f"Predicted prices fetched: {len(predictions)}")
        print(f"Predicted prices fetched: {len(predictions)} symbols.")
        return predictions
    except Exception as e:
        logging.error(f"Error fetching predicted prices: {e}")
        print(f"Error fetching predicted prices: {e}")
        return []


def is_valid_expiration(expiration_date, prediction_date):
    expiration = datetime.strptime(expiration_date, '%Y-%m-%d').date()
    prediction = datetime.strptime(prediction_date, '%Y-%m-%d').date()
    return (expiration - prediction).days


def get_latest_trade_price(symbol: str) -> Optional[float]:
    try:
        latest_trade = api.get_latest_trade(symbol)
        return latest_trade.price
    except Exception as e:
        logging.error(f"Error fetching latest trade price for {symbol}: {e}")
        return None


def place_order(api, symbol, option_type, strike_price, multiplier, max_risk_per_trade, portfolio_balance, take_profit_price, stop_loss_price):
    try:
        # Calculate the maximum trade value based on risk per trade
        max_trade_value = portfolio_balance * max_risk_per_trade
        logging.debug(f"Initialized max_trade_value: {max_trade_value}")

        # Fetch the current price of the option
        last_price = get_current_option_price(symbol)
        logging.debug(f"Initial last_price fetched: {last_price}")

        if last_price is None:
            logging.error(f"Cannot place order: no current price available for {symbol}")
            return False

        try:
            print(f'Here is the last price string value: {last_price}')
            logging.debug(f'Here is the last price string value: {last_price}')
            print(f'Here is the multiplier string value: {multiplier}')
            logging.debug(f'Here is the multiplier string value: {multiplier}')

            last_price = float(last_price)
            multiplier = float(multiplier)
        except (TypeError, ValueError) as e:
            logging.error(f"Error converting price or multiplier to float for {symbol}: {e}")
            return False

        logging.debug(f"Converted last_price: {last_price}, Multiplier: {multiplier}")

        if last_price <= 0 or multiplier <= 0:
            logging.error(f"Invalid price or multiplier for {symbol}: price={last_price}, multiplier={multiplier}")
            return False

        # Define the initial requested quantity
        requested_qty = int(max_trade_value / (last_price * multiplier))
        logging.debug(f"Initial requested quantity: {requested_qty}")

        if requested_qty <= 0:
            logging.error(f"Invalid number of shares to order for {symbol}: {requested_qty}")
            return False

        # Get the permissible quantity from the validate_options_trade method
        permissible_qty = rm.validate_options_trade(symbol, requested_qty)
        if permissible_qty <= 0:
            logging.warning(f"Order for {symbol} cannot be placed due to risk constraints.")
            return False

        # Proceed with placing the order using permissible_qty
        logging.debug(f"Trade validated for {symbol}, proceeding with order creation for {permissible_qty} shares.")

        # Step 1: Place the initial order
        initial_order_payload = create_order_payload(symbol, permissible_qty, last_price)
        if initial_order_payload is None:
            logging.error(f"Failed to create order payload for {symbol}")
            return False

        logging.debug(f"Initial order payload created: {json.dumps(initial_order_payload, indent=2)}")
        initial_order_response = submit_order(initial_order_payload)
        if not initial_order_response:
            return False

        # Wait for the initial order to be filled
        filled = wait_for_order_fill(initial_order_response['id'])
        if not filled:
            logging.error(f"Initial order for {symbol} was not filled within the expected timeframe")
            return False

        # Step 2: Place the take profit order
        take_profit_payload = create_order_payload(
            symbol, permissible_qty, take_profit_price,
            order_type="limit", side="sell", time_in_force="gtc"
        )
        if take_profit_payload is None:
            logging.error(f"Failed to create take profit order payload for {symbol}")
            return False

        # Step 3: Place the stop loss order
        stop_loss_payload = create_order_payload(
            symbol, permissible_qty, stop_loss_price,
            order_type="stop_limit", side="sell", time_in_force="gtc"
        )
        if stop_loss_payload is None:
            logging.error(f"Failed to create stop loss order payload for {symbol}")
            return False

        # Step 3: Place the stop loss order
        stop_loss_payload = create_order_payload(symbol, permissible_qty, stop_loss_price, order_type="stop_limit", side="sell")
        if stop_loss_payload is None:
            logging.error(f"Failed to create stop loss order payload for {symbol}")
            return False

        logging.debug(f"Stop loss order payload created: {json.dumps(stop_loss_payload, indent=2)}")
        stop_loss_response = submit_order(stop_loss_payload)
        if not stop_loss_response:
            return False

        # All orders placed successfully
        logging.debug(f"All orders placed successfully for {symbol}")
        return True
    except Exception as e:
        logging.error(f"Error in place_order for {symbol}: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return False

def find_best_option(
    symbol: str,
    predicted_price: float,
    prediction_date: datetime,
    mongo_client: pymongo.MongoClient,
    min_days_to_expire: int = 7,
    max_risk_per_trade: float = 0.05,
    portfolio_balance: float = 500000,
    risk_free_rate: float = 0.01,
    default_volatility: float = 0.20
) -> Optional[str]:
    options_collection = mongo_client["stock_data"]["options_contracts"]

    try:
        latest_trade_price = get_latest_trade_price(symbol)
        if latest_trade_price is None:
            logging.error(f"No latest trade price found for {symbol}")
            return None  # Skip if no trade price is found

        # Determine option type based on predicted price vs current price
        if predicted_price > latest_trade_price:
            option_type = "call"
        else:
            option_type = "put"

        logging.info(
            f"Symbol: {symbol}, Latest trade price: {latest_trade_price}, "
            f"Predicted price: {predicted_price}, Selecting {option_type.upper()} options."
        )

        # Define strike price range based on option type
        if option_type == "call":
            lower_bound = latest_trade_price
            upper_bound = predicted_price * 1.1  # 10% above predicted price
        else:  # put
            lower_bound = predicted_price * 0.9  # 10% below predicted price
            upper_bound = latest_trade_price

        logging.info(f"Strike price range: {lower_bound} to {upper_bound}")

        # Define minimum expiration date
        min_expiration_date = datetime.now() + timedelta(days=min_days_to_expire)
        logging.info(f"Minimum expiration date: {min_expiration_date}")

        # Prepare query for options
        query = {
            'root_symbol': symbol,
            'tradable': True,
            'type': option_type.lower(),
            'expiration_date': {'$gte': min_expiration_date},
            'strike_price': {'$gte': lower_bound, '$lte': upper_bound}
        }

        # Fetch options with additional filters
        options_cursor = options_collection.find(query)
        options_found = False
        enriched_options = []
        total_options = 0
        options_without_last_price = 0
        options_expiring_too_soon = 0

        for option in options_cursor:
            total_options += 1
            options_found = True
            try:
                strike_price = float(option['strike_price'])

                # Update this section
                last_price = option.get('last_price')
                if last_price is None:
                    last_price = get_current_option_price(option['symbol'])
                    if last_price is None:
                        options_without_last_price += 1
                        logging.warning(f"Option {option.get('symbol', 'N/A')} skipped: Unable to fetch current price")
                        continue

                expiry_date = option['expiration_date']
                if not isinstance(expiry_date, datetime):
                    expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d')  # Adjust format as needed

                days_until_expiration = (expiry_date - datetime.now()).days
                if days_until_expiration < min_days_to_expire:
                    options_expiring_too_soon += 1
                    logging.warning(f"Option {option.get('symbol', 'N/A')} skipped: Expires in {days_until_expiration} days (minimum: {min_days_to_expire})")
                    continue

                T = days_until_expiration / 365.25
                r = risk_free_rate
                sigma = default_volatility

                greeks = calculate_greeks(
                    S=latest_trade_price,
                    K=strike_price,
                    T=T,
                    r=r,
                    sigma=sigma,
                    option_type=option_type
                )
                option['greeks'] = greeks

                price_difference = abs(strike_price - predicted_price)
                option['price_difference'] = price_difference

                enriched_options.append(option)
                logging.info(f"Option {option.get('symbol', 'N/A')} added to enriched options. "
                             f"Strike: {strike_price}, Expiry: {expiry_date}, "
                             f"Greeks: {greeks}, Price Difference: {price_difference}")
            except Exception as e:
                logging.error(f"Error processing option {option.get('symbol', 'N/A')} for {symbol}: {e}")

        if not options_found:
            logging.error(f"No tradable options found for {symbol} matching initial criteria.")
            return None

        logging.info(f"Total options processed: {total_options}")
        logging.info(f"Options without last price: {options_without_last_price}")
        logging.info(f"Options expiring too soon: {options_expiring_too_soon}")
        logging.info(f"Enriched options count: {len(enriched_options)}")

        if enriched_options:
            best_option = min(
                enriched_options,
                key=lambda x: (x['price_difference'], abs(x['greeks']['delta']))
            )
            logging.info(f"Best option selected for {symbol}: {best_option['symbol']}")
            return best_option["symbol"]
        else:
            logging.error(f"No suitable {option_type.upper()} options found for {symbol} after all filtering.")
            return None

    except pymongo.errors.PyMongoError as e:
        logging.error(f"Database error finding option for symbol {symbol}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error finding option for symbol {symbol}: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None


def handle_symbol(prediction, mongo_client):
    symbol = prediction['symbol']
    predicted_price = prediction['predicted_price']
    prediction_date = prediction['prediction_date']

    logging.debug(f"Processing symbol: {symbol}")
    print(f"Processing symbol: {symbol}")

    if predicted_price is None:
        logging.info(f"No prediction for {symbol}, skipping")
        print(f"No prediction for {symbol}, skipping.")
        return

    if prediction_date is None:
        prediction_date = (datetime.now() + timedelta(days=180)).strftime('%Y-%m-%d')
        logging.debug(f"No prediction date for {symbol}, using date 6 months from now: {prediction_date}")
        print(f"No prediction date for {symbol}, using date 6 months from now: {prediction_date}")

    # Check if we already have an active order or position for this symbol or any of its options
    try:
        # Check active orders
        active_orders = api.list_orders(status='open')
        for order in active_orders:
            if order.symbol.startswith(symbol):
                logging.info(f"Active order already exists for {order.symbol}, skipping {symbol}")
                print(f"Active order already exists for {order.symbol}, skipping {symbol}")
                return

        # Check existing positions
        positions = api.list_positions()
        for position in positions:
            if position.symbol.startswith(symbol):
                logging.info(f"Existing position found for {position.symbol}, skipping {symbol}")
                print(f"Existing position found for {position.symbol}, skipping {symbol}")
                return
    except Exception as e:
        logging.error(f"Error checking existing orders/positions for {symbol}: {str(e)}")
        print(f"Error checking existing orders/positions for {symbol}: {str(e)}")
        return

    logging.debug(f"Finding best option for {symbol} with predicted price {predicted_price} and prediction date {prediction_date}")
    print(f"Finding best option for {symbol} with predicted price {predicted_price} and prediction date {prediction_date}")

    try:
        best_option_symbol = find_best_option(symbol, predicted_price, prediction_date, mongo_client,
                                              portfolio_balance=portfolio_balance,
                                              max_risk_per_trade=maximum_risk_per_trade)

        if best_option_symbol:
            options_collection = mongo_client["stock_data"]["options_contracts"]
            best_option = options_collection.find_one({"symbol": best_option_symbol})

            if best_option:
                last_price = best_option.get('last_price')
                if last_price is None:
                    last_price = get_latest_trade_price(symbol)
                    if last_price is None:
                        logging.info(f"Skipping {symbol}: No last price available for option {best_option_symbol}.")
                        print(f"Skipping {symbol}: No last price available for option {best_option_symbol}.")
                        return

                shares = int((portfolio_balance * maximum_risk_per_trade) / float(last_price))
                logging.debug(f"Calculated shares to order: {shares}")

                # Calculate take profit and stop loss prices (example calculations, adjust as needed)
                take_profit_price = last_price * 1.2  # 20% profit target
                stop_loss_price = last_price * 0.9  # 10% stop loss

                if place_order(api, best_option['symbol'], best_option['type'], best_option['strike_price'],
                               best_option['multiplier'], maximum_risk_per_trade, portfolio_balance,
                               take_profit_price, stop_loss_price):
                    print_order_details(symbol, best_option)
                else:
                    logging.info(f"Order not placed for {symbol}, risk parameters not met")
                    print(f"Order not placed for {symbol}, risk parameters not met.")
            else:
                logging.info(f"No valid option data found for {symbol} in the database.")
                print(f"No valid option data found for {symbol} in the database.")
        else:
            logging.info(f"No suitable options found for {symbol}")
            print(f"No suitable options found for {symbol}.")
    except Exception as e:
        logging.error(f"Error processing symbol {symbol}: {str(e)}")
        print(f"Error processing symbol {symbol}: {str(e)}")


def requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def get_current_option_price(symbol):
    """Fetch the current ask price of the given option symbol from Alpaca API."""
    url = f"https://data.alpaca.markets/v1beta1/options/quotes/latest?symbols={symbol}&feed=indicative"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY"),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY")
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        #logging.debug(f"API response for {symbol}: {data}")

        if symbol in data['quotes']:
            price_str = data['quotes'][symbol].get('ap')
            if price_str is None or float(price_str) == 0:
                logging.warning(f"No valid ask price available for {symbol}. Trying bid price.")
                price_str = data['quotes'][symbol].get('bp')

            if price_str is None or float(price_str) == 0:
                logging.error(f"No valid price available for {symbol}")
                return None

            try:
                current_price = float(price_str)
                logging.debug(f"Current price for {symbol}: {current_price}")
                return current_price
            except (ValueError) as e:
                logging.error(f"Error converting price to float for {symbol}: {e}")
                return None
        else:
            logging.error(f"No quote data available for {symbol}")
            return None
    except Exception as e:
        logging.error(f"Failed to fetch current price for {symbol}: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        return None


def place_order(api, symbol, option_type, strike_price, multiplier, max_risk_per_trade, portfolio_balance, take_profit_price, stop_loss_price):
    try:
        # Calculate the maximum trade value based on risk per trade
        max_trade_value = portfolio_balance * max_risk_per_trade
        logging.debug(f"Initialized max_trade_value: {max_trade_value}")

        # Fetch the current price of the option
        last_price = get_current_option_price(symbol)

        logging.debug(f"Initial last_price fetched: {last_price}")

        if last_price is None:
            logging.error(f"Cannot place order: no current price available for {symbol}")
            return False

        try:
            print(f'Here is the last price string value: {last_price}')
            logging.debug(f'Here is the last price string value: {last_price}')
            print(f'Here is the multiplier string value: {multiplier}')
            logging.debug(f'Here is the multiplier string value: {multiplier}')

            last_price = float(last_price)
            multiplier = float(multiplier)
        except (TypeError, ValueError) as e:
            logging.error(f"Error converting price or multiplier to float for {symbol}: {e}")
            return False


        logging.debug(f"Converted last_price: {last_price}, Multiplier: {multiplier}")

        if last_price <= 0 or multiplier <= 0:
            logging.error(f"Invalid price or multiplier for {symbol}: price={last_price}, multiplier={multiplier}")
            return False

        # Define the initial requested quantity
        requested_qty = int(max_trade_value / (last_price * multiplier))
        logging.debug(f"Initial requested quantity: {requested_qty}")

        if requested_qty <= 0:
            logging.error(f"Invalid number of shares to order for {symbol}: {requested_qty}")
            return False

        # Get the permissible quantity from the validate_options_trade method
        permissible_qty = rm.validate_options_trade(symbol, requested_qty)
        if permissible_qty <= 0:
            logging.warning(f"Order for {symbol} cannot be placed due to risk constraints.")
            return False

        # Proceed with placing the order using permissible_qty
        logging.debug(f"Trade validated for {symbol}, proceeding with order creation for {permissible_qty} shares.")

        # Step 1: Place the initial order
        initial_order_payload = create_order_payload(symbol, permissible_qty, last_price)
        if initial_order_payload is None:
            logging.error(f"Failed to create order payload for {symbol}")
            return False

        logging.debug(f"Initial order payload created: {json.dumps(initial_order_payload, indent=2)}")
        initial_order_response = submit_order(initial_order_payload)
        if not initial_order_response:
            return False

        # Wait for the initial order to be filled
        filled = wait_for_order_fill(initial_order_response['id'])
        if not filled:
            logging.error(f"Initial order for {symbol} was not filled within the expected timeframe")
            return False

        # Step 2: Place the take profit order
        take_profit_payload = create_order_payload(symbol, permissible_qty, take_profit_price, order_type="limit", side="sell")
        if take_profit_payload is None:
            logging.error(f"Failed to create take profit order payload for {symbol}")
            return False

        logging.debug(f"Take profit order payload created: {json.dumps(take_profit_payload, indent=2)}")
        take_profit_response = submit_order(take_profit_payload)
        if not take_profit_response:
            return False

        # Step 3: Place the stop loss order
        stop_loss_payload = create_order_payload(symbol, permissible_qty, stop_loss_price, order_type="stop_limit", side="sell")
        if stop_loss_payload is None:
            logging.error(f"Failed to create stop loss order payload for {symbol}")
            return False

        logging.debug(f"Stop loss order payload created: {json.dumps(stop_loss_payload, indent=2)}")
        stop_loss_response = submit_order(stop_loss_payload)
        if not stop_loss_response:
            return False

        # All orders placed successfully
        logging.debug(f"All orders placed successfully for {symbol}")
        store_order_details(symbol, option_type, strike_price, multiplier, permissible_qty, last_price, take_profit_price, stop_loss_price, initial_order_response)
        return True
    except Exception as e:
        logging.error(f"Error in place_order for {symbol}: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return False


def store_order_details(symbol, option_type, strike_price, multiplier, quantity, entry_price, take_profit_price, stop_loss_price, order_response):
    try:
        order_details = {
            "symbol": symbol,
            "option_type": option_type,
            "strike_price": strike_price,
            "multiplier": multiplier,
            "quantity": quantity,
            "entry_price": entry_price,
            "take_profit_price": take_profit_price,
            "stop_loss_price": stop_loss_price,
            "order_id": order_response['id'],
            "client_order_id": order_response['client_order_id'],
            "order_status": order_response['status'],
            "filled_qty": order_response['filled_qty'],
            "filled_avg_price": order_response['filled_avg_price'],
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }

        # Connect to MongoDB
        db = mongo_client["stock_data"]
        options_order_details = db["options_order_details"]

        # Insert the order details
        result = options_order_details.insert_one(order_details)

        logging.info(f"Order details for {symbol} stored in MongoDB with ID: {result.inserted_id}")
    except Exception as e:
        logging.error(f"Error storing order details for {symbol} in MongoDB: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")


def wait_for_order_fill(order_id, timeout=15):
    start_time = time.time()
    while time.time() - start_time < timeout:
        order_status = api.get_order(order_id)
        if order_status.status == 'filled':
            logging.debug(f"Order {order_id} has been filled")
            return True
        elif order_status.status in ['canceled', 'expired', 'rejected']:
            logging.error(f"Order {order_id} has status: {order_status.status}")
            return False
        time.sleep(5)
    logging.error(f"Order {order_id} was not filled within {timeout} seconds")
    return False


def create_order_payload(symbol, shares_to_order, price, order_type="limit", side="buy", time_in_force="day"):
    try:
        client_order_id = f"gcos_{random.randrange(100000000)}"

        if price is None:
            logging.error(f"Cannot create order payload: price is None for {symbol}")
            return None

        # Convert to Decimal for precise calculation
        price_decimal = Decimal(str(price))

        # Format to two decimal places
        price_formatted = price_decimal.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

        # Construct the order payload based on the type
        payload = {
            "symbol": symbol,
            "qty": str(shares_to_order),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
            "limit_price": str(price_formatted) if order_type in ["limit", "stop_limit"] else None,
            "stop_price": str(price_formatted) if order_type == "stop_limit" else None,
            "client_order_id": client_order_id
        }

        # Remove any keys that are not applicable to the order type
        payload = {k: v for k, v in payload.items() if v is not None}

        logging.debug(f"Created order payload for {symbol}: {payload}")
        return payload
    except Exception as e:
        logging.error(f"Error creating order payload for {symbol}: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None



def submit_order(payload):
    headers = {
        "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY"),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY"),
        "Content-Type": "application/json"
    }

    try:
        response = requests.post("https://paper-api.alpaca.markets/v2/orders", json=payload, headers=headers)
        logging.debug(f"Order submission response status code: {response.status_code}")

        if response.status_code == 200:
            order_response = response.json()
            logging.debug(f"Order placed successfully: {order_response}")
            print(f"Order placed successfully: {order_response}")
            return order_response  # Return the full response for further processing
        else:
            logging.error(f"Failed to place order: {response.json()}")
            print(f"Failed to place order: {response.json()}")
            return None  # Return None to indicate failure
    except requests.RequestException as e:
        logging.error(f"Error during order submission: {e}")
        print(f"Error during order submission: {e}")
        return None  # Return None in case of an exception


def print_order_details(symbol, option, order_response=None):
    try:
        last_price = option.get('last_price')
        if last_price is None:
            logging.warning(f"Last price is None for {symbol}. Using 0 as a placeholder.")
            last_price = 0

        order_details = {
            "symbol": symbol,
            "option_symbol": option.get('symbol'),
            "strike_price": option.get('strike_price'),
            "expiration_date": option.get('expiration_date'),
            "option_type": option.get('type'),
            "last_price": last_price,
            "quantity": 1,
            "multiplier": option.get('multiplier'),
            "order_type": "limit",
            "time_in_force": "day",
            "limit_price": round(float(last_price), 2),
            "order_status": "pending",
            "inserted_at": datetime.utcnow()
        }

        if order_response:
            order_details.update({
                "order_id": order_response.get("id"),
                "client_order_id": order_response.get("client_order_id"),
                "order_status": order_response.get("status"),
                "filled_qty": order_response.get("filled_qty"),
                "filled_avg_price": order_response.get("filled_avg_price"),
                "order_response": order_response
            })

        logging.info(f"Order Details: {order_details}")
        print(f"Order Details: {order_details}")

        try:
            db = mongo_client["stock_data"]
            orders_collection = db["options_order_details"]
            orders_collection.insert_one(order_details)
            logging.debug(f"Order details for {symbol} saved to MongoDB.")
            print(f"Order details for {symbol} saved to MongoDB.")
        except Exception as e:
            logging.error(f"Failed to save order details for {symbol} to MongoDB: {e}")
            print(f"Failed to save order details for {symbol} to MongoDB: {e}")

    except Exception as e:
        logging.error(f"Error in print_order_details for {symbol}: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    logging.debug("Starting the options bracket order script.")
    print("Starting the options bracket order script.")

    with open(risk_params_path, 'r') as f:
        risk_params = json.load(f)

    portfolio_balance = risk_params.get('max_portfolio_size')
    maximum_risk_per_trade = risk_params.get('max_risk_per_trade')

    logging.info(f"Loaded portfolio balance: {portfolio_balance}")
    logging.info(f"Loaded maximum risk per trade: {maximum_risk_per_trade}")

    rm = OptionsRiskManagement(api, risk_params)

    predicted_prices = get_predicted_prices(mongo_client)
    logging.debug(f"Predicted prices fetched for {len(predicted_prices)} symbols")
    print(f"Predicted prices fetched for {len(predicted_prices)} symbols")

    all_symbols = set(options_contracts_collection.distinct('root_symbol'))
    all_symbols.update(p['symbol'] for p in predicted_prices)

    logging.debug(f"Total unique symbols to process: {len(all_symbols)}")
    print(f"Total unique symbols to process: {len(all_symbols)}")

    prediction_dict = {p['symbol']: p for p in predicted_prices}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for symbol in all_symbols:
            prediction = prediction_dict.get(symbol, {'symbol': symbol, 'predicted_price': None, 'prediction_date': None})
            futures.append(executor.submit(handle_symbol, prediction, mongo_client))

        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    logging.debug(f"Successfully processed symbol: {result}")
                else:
                    logging.debug(f"Symbol processing completed without notable result")
            except Exception as exc:
                logging.error(f"An exception occurred while processing: {exc}")
                print(f"An exception occurred while processing: {exc}")

    logging.debug("Options bracket order script completed.")
    print("Options bracket order script completed.")