import alpaca_trade_api as tradeapi
import requests
import json
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import numpy as np
from scipy.stats import norm
from pymongo import MongoClient
import logging
from time import sleep

# Setup logging
logging.basicConfig(filename='risk_strategy_options.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Teams webhook URL
TEAMS_URL = os.getenv("TEAMS_WEBHOOK_URL")

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
            return risk_params
    else:
        logging.error(f"Error: 'risk_params.json' not found at {risk_params_file_path}")
        return None

risk_params = load_risk_params()
if risk_params:
    print(f"Risk parameters loaded successfully. This is total max size right now: {risk_params['max_position_size']}")
else:
    print("Failed to load risk parameters.")


class OptionsRiskManagement:
    def __init__(self, api, risk_params):
        self.api = api
        self.risk_params = risk_params
        self.total_trades_today = 0
        self.max_options_allocation = 0.90  # Maximum 90% of portfolio allocated to options
        self.max_options_loss_threshold = 0.10  # Maximum 10% loss threshold per option trade
        self.min_options_profit_threshold = 0.10  # Minimum 10% profit threshold for notification
        self.current_options_value = 0  # Placeholder, will be calculated

    def get_portfolio_value(self):
        """Get the current portfolio value."""
        try:
            account = self.api.get_account()
            return float(account.portfolio_value)
        except Exception as e:
            logging.error(f"Error fetching portfolio value: {e}")
            return None

    def get_open_option_positions(self):
        """Fetch current open options positions."""
        try:
            positions = self.api.list_positions()
            options_positions = [p for p in positions if 'C' in p.symbol or 'P' in p.symbol]
            return options_positions
        except Exception as e:
            logging.error(f"Error fetching open option positions: {e}")
            return []

    def calculate_current_options_value(self):
        """Calculate the total value of currently held options."""
        options_positions = self.get_open_option_positions()
        total_value = sum(float(p.market_value) for p in options_positions if p.market_value is not None)
        return total_value

    def validate_options_trade(self, symbol, requested_qty, order_type='buy'):
        """Validate if an options trade adheres to risk management rules and return permissible quantity."""
        try:
            current_price = self.get_current_option_price(symbol)
            logging.debug(
                f"Validating trade for {symbol}: Current price: {current_price}, Requested Quantity: {requested_qty}")

            if current_price is None:
                logging.error(f"Cannot validate trade: no current price available for {symbol}")
                return 0  # Return 0 if validation fails due to missing price

            portfolio_value = self.get_portfolio_value()
            if portfolio_value is None:
                logging.error("Cannot validate trade: unable to fetch portfolio value")
                return 0  # Return 0 if validation fails due to missing portfolio value

            self.current_options_value = self.calculate_current_options_value()
            logging.debug(
                f"Portfolio value: {portfolio_value}, Current options value: {self.current_options_value}")

            # Calculate the maximum permissible quantity based on risk parameters
            max_trade_value = portfolio_value * self.risk_params.get('max_risk_per_trade', 0.05)
            max_position_size = self.risk_params.get('max_position_size', float('inf'))
            max_allocation = portfolio_value * self.max_options_allocation

            # Calculate permissible quantities based on different constraints
            max_qty_by_risk = max_trade_value / (current_price * 100)
            max_qty_by_position_size = max_position_size / (current_price * 100)
            max_qty_by_allocation = (max_allocation - self.current_options_value) / (current_price * 100)

            # Determine the final permissible quantity
            permissible_qty = min(max_qty_by_risk, max_qty_by_position_size, max_qty_by_allocation, requested_qty)

            logging.info(f"Permissible quantity for {symbol}: {permissible_qty}")
            return int(permissible_qty)  # Return the permissible quantity as an integer
        except Exception as e:
            logging.error(f"Error validating options trade for {symbol}: {e}")
            return 0  # Return 0 if an error occurs

    def get_current_option_price(self, symbol):
        """Fetch the current price of the option from Alpaca API."""
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

            if symbol in data['quotes']:
                price = float(data['quotes'][symbol]['ap'])
                logging.debug(f"Current price for {symbol}: {price}")
                return price
            else:
                logging.info(f"No quote data found for {symbol}.")
                return None
        except Exception as e:
            logging.error(f"Error fetching current price for {symbol}: {e}")
            return None

    def calculate_options_greeks(self, S, K, T, r, sigma, option_type):
        """Calculate and return option Greeks."""
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if option_type.lower() == 'call':
                delta = norm.cdf(d1)
                gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
                vega = S * norm.pdf(d1) * np.sqrt(T)
            else:  # Put option
                delta = -norm.cdf(-d1)
                gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
                vega = S * norm.pdf(d1) * np.sqrt(T)

            return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}
        except Exception as e:
            logging.error(f"Error calculating Greeks: {e}")
            return None


    def check_options_risk(self, symbol, quantity, price):
        """Implement risk checks for options trades."""
        try:
            portfolio_value = self.get_portfolio_value()
            if portfolio_value is None:
                return False

            position_value = quantity * price * 100  # Assuming standard options multiplier of 100

            if position_value > self.risk_params.get('max_position_size', float('inf')):
                logging.warning(f"Position size for {symbol} exceeds maximum allowed.")
                return False

            total_options_value = self.calculate_current_options_value() + position_value
            if total_options_value > (portfolio_value * self.max_options_allocation):
                logging.warning(f"Total options allocation would exceed maximum allowed for {symbol}.")
                return False

            return True
        except Exception as e:
            logging.error(f"Error checking options risk for {symbol}: {e}")
            return False

    def monitor_options_positions(self):
        """Monitor and manage open options positions."""
        try:
            options_positions = self.get_open_option_positions()
            for position in options_positions:
                symbol = position.symbol
                # Extract expiration date from the symbol
                expiration_date = self.extract_expiration_date(symbol)

                if expiration_date:
                    days_to_expiration = (expiration_date - datetime.now().date()).days
                    if days_to_expiration <= 7:
                        logging.info(f"Option {symbol} is nearing expiration (in {days_to_expiration} days).")
                        # Implement logic to decide whether to close the position
                else:
                    logging.warning(f"Could not determine expiration date for {symbol}")

                # Handle unrealized profit/loss
                try:
                    unrealized_plpc = float(position.unrealized_plpc) if position.unrealized_plpc is not None else 0.0
                except (ValueError, AttributeError):
                    logging.warning(f"Could not determine unrealized P/L for {symbol}. Assuming no impact.")
                    unrealized_plpc = 0.0

                if unrealized_plpc < -self.max_options_loss_threshold:
                    logging.warning(f"Option {symbol} has exceeded loss threshold: {unrealized_plpc:.2%}")
                    # Send notification to Teams for Loss
                    message = f"Option {symbol} has exceeded the loss threshold with an unrealized P/L of {unrealized_plpc:.2%}."
                    send_teams_notification(message)

                elif unrealized_plpc > self.min_options_profit_threshold:
                    logging.info(f"Option {symbol} has exceeded profit threshold: {unrealized_plpc:.2%}")
                    # Send notification to Teams for Profit
                    message = f"Option {symbol} has exceeded the profit threshold with an unrealized P/L of {unrealized_plpc:.2%}."
                    send_teams_notification(message)

                else:
                    logging.info(f"Option {symbol} unrealized P/L: {unrealized_plpc:.2%}")

        except Exception as e:
            logging.error(f"Error monitoring options positions: {e}")


    def extract_expiration_date(self, symbol):
        """Extract expiration date from option symbol."""
        try:
            # Assuming the symbol format is like 'AAPL210917C00150000'
            # where '210917' represents the expiration date (YYMMDD)
            date_str = symbol[-15:-9]  # Extract the date part
            expiration_date = datetime.strptime(date_str, '%y%m%d').date()
            return expiration_date
        except ValueError:
            logging.error(f"Could not extract expiration date from symbol: {symbol}")
            return None

def send_teams_notification(message):
    """Send a notification to a Microsoft Teams channel."""
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "text": message
    }
    try:
        response = requests.post(TEAMS_URL, headers=headers, json=payload)
        response.raise_for_status()
        logging.info("Notification sent to Teams successfully.")
    except Exception as e:
        logging.error(f"Failed to send notification to Teams: {e}")


def main():
    logging.info("Starting Options Performance Monitor")

    # Initialize OptionsRiskManagement
    risk_manager = OptionsRiskManagement(api, risk_params)

    try:
        # Get current portfolio value
        portfolio_value = risk_manager.get_portfolio_value()
        logging.info(f"Current portfolio value: ${portfolio_value:.2f}")

        # Calculate current options value
        current_options_value = risk_manager.calculate_current_options_value()
        logging.info(f"Current total options value: ${current_options_value:.2f}")

        # Get open options positions
        open_positions = risk_manager.get_open_option_positions()
        logging.info(f"Number of open option positions: {len(open_positions)}")

        # Check overall options allocation
        options_allocation_percentage = (current_options_value / portfolio_value) * 100 if portfolio_value else 0
        logging.info(f"Current options allocation: {options_allocation_percentage:.2f}%")

        if options_allocation_percentage > risk_manager.max_options_allocation * 100:
            logging.warning("Options allocation exceeds maximum allowed percentage!")

        # Analyze each open option position
        for position in open_positions:
            symbol = position.symbol
            quantity = int(position.qty)
            current_price = risk_manager.get_current_option_price(symbol)

            if current_price is not None:
                position_value = current_price * quantity
                logging.info(f"Analyzing position: {symbol}")
                logging.info(f"  Quantity: {quantity}")
                logging.info(f"  Current Price: ${current_price:.2f}")
                logging.info(f"  Position Value: ${position_value:.2f}")

                # Validate the existing position
                if risk_manager.validate_options_trade(symbol, quantity):
                    logging.info(f"  Position {symbol} is within risk parameters.")
                else:
                    logging.warning(f"  Position {symbol} exceeds risk parameters!")

                # Check options risk
                if risk_manager.check_options_risk(symbol, quantity, current_price):
                    logging.info(f"  Position {symbol} passes risk checks.")
                else:
                    logging.warning(f"  Position {symbol} fails risk checks!")

            else:
                logging.error(f"Unable to fetch current price for {symbol}")

        # Monitor options positions
        risk_manager.monitor_options_positions()

    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}")


if __name__ == "__main__":
    main()