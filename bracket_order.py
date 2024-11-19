from datetime import datetime
import alpaca_trade_api as tradeapi
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import json
import os
from dotenv import load_dotenv
import logging
import random
import time  # Added for rate limiting
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from risk_strategy import RiskManagement, risk_params  # Ensure this module is correctly implemented


@dataclass
class TradingConfig:
    """Configuration for trading parameters"""
    DEBUG_MODE: bool = False  # Set to False to process all symbols
    RSI_OVERSOLD: float = 37.0
    RSI_OVERBOUGHT: float = 62.0
    SMA_THRESHOLD: float = 1.04
    MIN_CASH_RATIO: float = 0.1
    TAKE_PROFIT_MULTIPLIER: float = 1.0243
    STOP_LOSS_MULTIPLIER: float = 0.9821
    MAX_WORKERS: int = 5  # Increased to process more symbols concurrently
    ALPHA_VANTAGE_CALLS_PER_MINUTE: int = 299  # premium limit intact when license active
    SLEEP_TIME: int = 1  # Sleep time between API calls in seconds


@dataclass
class MarketIndicators:
    """Store market indicators for a symbol"""
    rsi: float
    macd: float
    signal: float
    close: float
    sma: float


class BracketOrderTrader:
    def __init__(self, api_key: str, api_secret: str, mongo_conn: str, alpha_vantage_key: str, teams_url: str):
        # Set up logging
        self.logger = logging.getLogger('BracketOrderTrader')
        self.logger.setLevel(logging.INFO)

        # Create handlers
        file_handler = logging.FileHandler('trading.log')
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

        self.logger.info("Initializing BracketOrderTrader...")

        self.config = TradingConfig()
        self.api = tradeapi.REST(api_key, api_secret, base_url='https://paper-api.alpaca.markets')
        self.mongo_client = MongoClient(mongo_conn)
        self.db = self.mongo_client["stock_data"]
        self.alpha_vantage_key = alpha_vantage_key
        self.teams_url = teams_url

        # Initialize RiskManagement
        self.rm = RiskManagement(self.api, risk_params)

        # Initialize account values
        account = self.api.get_account()
        self.portfolio_balance = float(account.portfolio_value)
        self.cash_balance = float(account.cash)

        self.logger.info("Trading system initialized.")
        self.logger.info(f"Portfolio balance: {self.portfolio_balance}")
        self.logger.info(f"Cash balance: {self.cash_balance}")
        print("Trading system initialized.")
        print(f"Portfolio balance: {self.portfolio_balance}")
        print(f"Cash balance: {self.cash_balance}")

    def store_trade_decision(self, symbol: str, decision: str, reason: str, indicators: Dict[str, Any] = None) -> None:
        """Store trading decisions in MongoDB"""
        try:
            document = {
                'symbol': symbol,
                'decision': decision,
                'reason': reason,
                'indicators': indicators or {},
                'timestamp': datetime.now()
            }
            self.db.trade_decisions.insert_one(document)
            self.logger.info(f"Stored trade decision for {symbol}: {decision}, Reason: {reason}")
            print(f"Stored trade decision for {symbol}: {decision}, Reason: {reason}")
        except Exception as e:
            self.logger.error(f"Error storing trade decision for {symbol}: {e}")
            print(f"Error storing trade decision for {symbol}: {e}")

    def send_teams_notification(self, message: str) -> None:
        """Send notification to Microsoft Teams"""
        try:
            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": "0076D7",
                "summary": "Trade Orders Summary",
                "sections": [{
                    "activityTitle": "Trade Orders Placed",
                    "activitySubtitle": "Summary of Buy and Sell Orders",
                    "facts": [{
                        "name": "Orders",
                        "value": message
                    }],
                    "markdown": True
                }]
            }
            headers = {"Content-type": "application/json"}
            response = requests.post(self.teams_url, headers=headers, data=json.dumps(payload))
            if response.status_code != 200:
                self.logger.error(f"Failed to send Teams notification: {response.status_code}")
                print(f"Failed to send Teams notification: {response.status_code}")
            else:
                self.logger.info("Teams notification sent successfully.")
                print("Teams notification sent successfully.")
        except Exception as e:
            self.logger.error(f"Error sending Teams notification: {e}")
            print(f"Error sending Teams notification: {e}")

    def get_market_indicators(self, symbol: str) -> Optional[MarketIndicators]:
        """Fetch and calculate market indicators for a symbol with proper rate limiting"""
        try:
            self.logger.info(f"Fetching market indicators for {symbol}")
            print(f"Fetching market indicators for {symbol}")

            base_url = 'https://www.alphavantage.co/query'
            urls = {
                'daily': f'{base_url}?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={self.alpha_vantage_key}',
                'rsi': f'{base_url}?function=RSI&symbol={symbol}&interval=daily&time_period=14&series_type=close&apikey={self.alpha_vantage_key}',
                'macd': f'{base_url}?function=MACD&symbol={symbol}&interval=daily&series_type=close&apikey={self.alpha_vantage_key}',
                'sma': f'{base_url}?function=SMA&symbol={symbol}&interval=daily&time_period=30&series_type=close&apikey={self.alpha_vantage_key}'
            }

            responses = {}
            for name, url in urls.items():
                self.logger.info(f"Requesting {name} data for {symbol}")
                print(f"Requesting {name} data for {symbol}")

                # Implement exponential backoff for API calls
                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    response = requests.get(url).json()
                    if "Error Message" not in response and "Note" not in response:
                        responses[name] = response
                        break
                    retry_count += 1
                    if retry_count < max_retries:
                        sleep_time = self.config.SLEEP_TIME * (2 ** retry_count)
                        self.logger.info(f"Rate limit hit, waiting {sleep_time} seconds before retry")
                        print(f"Rate limit hit, waiting {sleep_time} seconds before retry")
                        time.sleep(sleep_time)

                if retry_count == max_retries:
                    self.logger.error(f"Failed to fetch {name} data for {symbol} after {max_retries} retries")
                    print(f"Failed to fetch {name} data for {symbol} after {max_retries} retries")
                    return None

                # Standard rate limit handling
                time.sleep(self.config.SLEEP_TIME)

            # Extract latest values with error checking
            if 'Time Series (Daily)' not in responses['daily']:
                self.logger.error(f"Time Series data not available for {symbol}")
                print(f"Time Series data not available for {symbol}")
                return None
            daily_point = list(responses['daily']['Time Series (Daily)'].values())[0]

            if 'Technical Analysis: RSI' not in responses['rsi']:
                self.logger.error(f"RSI data not available for {symbol}")
                print(f"RSI data not available for {symbol}")
                return None
            rsi_point = list(responses['rsi']['Technical Analysis: RSI'].values())[0]

            if 'Technical Analysis: MACD' not in responses['macd']:
                self.logger.error(f"MACD data not available for {symbol}")
                print(f"MACD data not available for {symbol}")
                return None
            macd_point = list(responses['macd']['Technical Analysis: MACD'].values())[0]

            if 'Technical Analysis: SMA' not in responses['sma']:
                self.logger.error(f"SMA data not available for {symbol}")
                print(f"SMA data not available for {symbol}")
                return None
            sma_point = list(responses['sma']['Technical Analysis: SMA'].values())[0]

            indicators = MarketIndicators(
                rsi=float(rsi_point['RSI']),
                macd=float(macd_point['MACD']),
                signal=float(macd_point['MACD_Signal']),
                close=float(daily_point['4. close']),
                sma=float(sma_point['SMA'])
            )

            self.logger.info(f"Market indicators for {symbol}: {indicators}")
            print(f"Market indicators for {symbol}: {indicators}")

            return indicators

        except Exception as e:
            self.logger.error(f"Error fetching indicators for {symbol}: {str(e)}")
            print(f"Error fetching indicators for {symbol}: {str(e)}")
            return None

    def check_trading_conditions(self, indicators: MarketIndicators) -> str:
        """Analyze trading conditions and return decision"""
        buy_conditions = [
            indicators.rsi <= self.config.RSI_OVERSOLD,
            indicators.macd >= indicators.signal,
            indicators.close >= indicators.sma * self.config.SMA_THRESHOLD
        ]

        sell_conditions = [
            indicators.rsi >= self.config.RSI_OVERBOUGHT,
            indicators.macd < indicators.signal,
            indicators.close < indicators.sma
        ]

        buy_count = sum(buy_conditions)
        sell_count = sum(sell_conditions)

        if buy_count >= 2:
            return 'buy'
        elif sell_count >= 2:
            return 'sell'
        return 'hold'

    def calculate_order_size(self, symbol: str, close_price: float) -> int:
        """Calculate the number of shares to order based on risk parameters"""
        try:
            # Get account info
            account = self.api.get_account()
            portfolio_value = float(account.portfolio_value)
            cash_balance = float(account.cash)

            # Calculate the maximum allowed trade value based on risk parameters
            max_allowable_risk = portfolio_value * self.rm.risk_params['max_risk_per_trade']  # This is 11% of portfolio
            max_asset_allocation = self.rm.risk_params['max_equity_equity']  # Maximum allocation for equity type

            # Use the smaller of the two limits
            max_trade_value = min(max_allowable_risk, max_asset_allocation)

            # Further adjust based on available cash
            min_cash_required = portfolio_value * self.config.MIN_CASH_RATIO
            if cash_balance - max_trade_value < min_cash_required:
                max_trade_value = max(0, cash_balance - min_cash_required)

            # Calculate shares based on adjusted max trade value
            shares = int(max_trade_value // close_price)

            # Log the calculation details
            self.logger.info(
                f"Order size calculation for {symbol}:\n"
                f"Portfolio Value: ${portfolio_value:,.2f}\n"
                f"Max Risk per Trade (11%): ${max_allowable_risk:,.2f}\n"
                f"Max Asset Allocation: ${max_asset_allocation:,.2f}\n"
                f"Selected Max Trade Value: ${max_trade_value:,.2f}\n"
                f"Close Price: ${close_price:,.2f}\n"
                f"Calculated Shares: {shares}"
            )

            return max(0, shares)
        except Exception as e:
            self.logger.error(f"Error calculating order size for {symbol}: {e}")
            print(f"Error calculating order size for {symbol}: {e}")
            return 0

    def place_bracket_order(self, symbol: str, shares: int, close_price: float) -> bool:
        """Place a bracket order with take profit and stop loss"""
        try:
            # Get account info and calculate trade parameters
            account = self.api.get_account()

            # Define trade parameters within local scope
            trade_params = {
                'portfolio_value': float(account.portfolio_value),
                'cash_balance': float(account.cash),
                'max_trade_value': float(account.portfolio_value) * self.rm.risk_params['max_risk_per_trade'],
                'min_cash_required': float(account.portfolio_value) * self.config.MIN_CASH_RATIO,
                'trade_value': float(shares) * float(close_price)
            }

            # Log initial parameters
            self.logger.info(
                f"Trade parameters for {symbol}:\n"
                f"Portfolio Value: {trade_params['portfolio_value']}\n"
                f"Cash Balance: {trade_params['cash_balance']}\n"
                f"Max Trade Value: {trade_params['max_trade_value']}\n"
                f"Min Cash Required: {trade_params['min_cash_required']}\n"
                f"Proposed Trade Value: {trade_params['trade_value']}"
            )

            # Check if trade would leave enough cash
            if trade_params['cash_balance'] - trade_params['trade_value'] < trade_params['min_cash_required']:
                adjusted_trade_value = trade_params['cash_balance'] - trade_params['min_cash_required']
                shares = int(adjusted_trade_value // close_price)
                trade_params['trade_value'] = float(shares) * float(close_price)
                self.logger.info(
                    f"Adjusted shares to {shares} to maintain minimum cash requirement. "
                    f"New trade value: {trade_params['trade_value']}"
                )

            # Validate final trade value and shares
            if shares <= 0:
                self.logger.info(f"No shares to trade for {symbol} after adjustments")
                return False

            # Validate with risk management
            if not self.rm.validate_trade(symbol, shares, 'buy'):
                self.logger.info(f"Trade validation failed for {symbol}")
                print(f"Trade validation failed for {symbol}")
                return False

            self.logger.info(f"Placing bracket order for {symbol}: {shares} shares at {close_price}")
            print(f"Placing bracket order for {symbol}: {shares} shares at {close_price}")

            # Calculate order parameters
            order_params = {
                'take_profit': {"limit_price": round(close_price * self.config.TAKE_PROFIT_MULTIPLIER, 2)},
                'stop_loss': {"stop_price": round(close_price * self.config.STOP_LOSS_MULTIPLIER, 2)},
                'client_order_id': f"gcos_{random.randrange(100000000)}"
            }

            # Place the order
            order = self.api.submit_order(
                symbol=symbol,
                qty=shares,
                side='buy',
                type='limit',
                limit_price=round(close_price, 2),
                order_class='bracket',
                take_profit=order_params['take_profit'],
                stop_loss=order_params['stop_loss'],
                client_order_id=order_params['client_order_id'],
                time_in_force='day'
            )

            self.logger.info(f"Order placed successfully for {symbol}: {order}")
            print(f"Order placed successfully for {symbol}: {order}")

            # Send notification
            self.send_teams_notification(
                f"Order placed successfully! Symbol: {symbol}, Shares: {shares}, Price: {close_price}"
            )
            return True

        except tradeapi.rest.APIError as e:
            self.logger.error(f"Alpaca API error for {symbol}: {str(e)}")
            print(f"Alpaca API error for {symbol}: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to place order for {symbol}: {str(e)}")
            print(f"Failed to place order for {symbol}: {str(e)}")
            return False



    def handle_symbol(self, symbol: str) -> None:
        """Process a single symbol for trading opportunities"""
        try:
            self.logger.info(f"Processing symbol: {symbol}")
            print(f"Processing symbol: {symbol}")

            # Check if we already have a position or open order
            current_positions = {p.symbol: p.qty for p in self.api.list_positions()}
            open_orders = {o.symbol for o in self.api.list_orders(status='open')}

            self.logger.info(f"Current positions: {current_positions}")
            self.logger.info(f"Open orders: {open_orders}")
            print(f"Current positions: {current_positions}")
            print(f"Open orders: {open_orders}")

            if symbol in open_orders:
                self.logger.info(f"Skipping {symbol} due to existing open order")
                print(f"Skipping {symbol} due to existing open order")
                self.store_trade_decision(symbol, 'skip', 'Existing open order')
                return

            # Get market indicators
            indicators = self.get_market_indicators(symbol)
            if not indicators:
                self.logger.info(f"Market indicators not available for {symbol}")
                print(f"Market indicators not available for {symbol}")
                return

            # Check trading conditions
            action = self.check_trading_conditions(indicators)
            self.logger.info(f"Trading decision for {symbol}: {action}")
            print(f"Trading decision for {symbol}: {action}")

            # Handle different trading actions
            if action == 'buy':
                self._handle_buy_action(symbol, indicators)
            elif action == 'sell':
                self._handle_sell_action(symbol, current_positions, indicators)
            else:  # action == 'hold'
                self._handle_hold_action(symbol, current_positions, indicators)

        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {str(e)}")
            print(f"Error processing {symbol}: {str(e)}")
            self.store_trade_decision(symbol, 'error', str(e))

    def _handle_buy_action(self, symbol: str, indicators: MarketIndicators) -> None:
        """Handle buy action for a symbol"""
        self.logger.info(f"Buy conditions met for {symbol}")
        print(f"Buy conditions met for {symbol}")
        shares = self.calculate_order_size(symbol, indicators.close)
        if shares > 0:
            success = self.place_bracket_order(symbol, shares, indicators.close)
            self.store_trade_decision(
                symbol,
                'buy',
                'Order placed' if success else 'Order failed',
                vars(indicators)
            )
        else:
            self.logger.info(f"Insufficient shares calculated for {symbol}")
            print(f"Insufficient shares calculated for {symbol}")
            self.store_trade_decision(symbol, 'hold', 'Insufficient shares to place order', vars(indicators))

    def _handle_sell_action(self, symbol: str, current_positions: Dict[str, str], indicators: MarketIndicators) -> None:
        """Handle sell action for a symbol"""
        if symbol in current_positions:
            self.logger.info(f"Sell conditions met for {symbol} - initiating sale")
            print(f"Sell conditions met for {symbol} - initiating sale")
            # Implement sell logic here if needed
            # Example: place a sell order
            try:
                qty = int(current_positions[symbol])
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                self.logger.info(f"Market sell order placed for {symbol}: {order}")
                print(f"Market sell order placed for {symbol}: {order}")
                self.store_trade_decision(symbol, 'sell', 'Sell order placed', vars(indicators))
            except Exception as e:
                self.logger.error(f"Failed to place sell order for {symbol}: {str(e)}")
                print(f"Failed to place sell order for {symbol}: {str(e)}")
                self.store_trade_decision(symbol, 'sell_failed', str(e), vars(indicators))
        else:
            self.logger.info(f"No existing position for {symbol} to sell.")
            print(f"No existing position for {symbol} to sell.")
            self.store_trade_decision(symbol, 'sell', 'Sell conditions met but no existing position', vars(indicators))

    def _handle_hold_action(self, symbol: str, current_positions: Dict[str, str], indicators: MarketIndicators) -> None:
        """
        Handle hold action for a symbol. If we don't have a position and conditions are favorable,
        treat it as a potential entry point.
        """
        if symbol not in current_positions:
            self.logger.info(f"Hold signal for {symbol} with no existing position - evaluating entry")
            print(f"Hold signal for {symbol} with no existing position - evaluating entry")

            # Basic trend confirmation
            trend_is_favorable = indicators.close >= indicators.sma  # Price above SMA indicates uptrend

            # Check if RSI is in a reasonable range (not overbought)
            rsi_is_favorable = indicators.rsi < self.config.RSI_OVERBOUGHT

            # Check if MACD is showing momentum
            macd_is_favorable = indicators.macd > indicators.signal

            if trend_is_favorable and rsi_is_favorable and macd_is_favorable:
                shares = self.calculate_order_size(symbol, indicators.close)
                if shares > 0:
                    self.logger.info(f"Attempting to enter position in {symbol} on hold signal")
                    print(f"Attempting to enter position in {symbol} on hold signal")

                    if self.rm.validate_trade(symbol, shares, 'buy'):
                        success = self.place_bracket_order(symbol, shares, indicators.close)
                        self.store_trade_decision(
                            symbol,
                            'buy',
                            'Order placed on hold signal' if success else 'Order failed on hold signal',
                            vars(indicators)
                        )
                        return
                    else:
                        self.logger.info(f"Trade validation failed for {symbol} on hold signal")
                        print(f"Trade validation failed for {symbol} on hold signal")
                else:
                    self.logger.info(f"Insufficient shares calculated for {symbol} on hold signal")
                    print(f"Insufficient shares calculated for {symbol} on hold signal")
            else:
                reasons = []
                if not trend_is_favorable:
                    reasons.append("price below SMA")
                if not rsi_is_favorable:
                    reasons.append("RSI unfavorable")
                if not macd_is_favorable:
                    reasons.append("MACD unfavorable")
                reason = f"No position - conditions not met ({', '.join(reasons)})"
                self.store_trade_decision(symbol, 'hold', reason, vars(indicators))
        else:
            self.store_trade_decision(
                symbol,
                'hold',
                'Maintaining existing position',
                vars(indicators)
            )

    def run(self):
        """Run the trading system"""
        try:
            self.logger.info("Starting trading system run.")
            print("Starting trading system run.")

            symbols_cursor = self.db.selected_pairs.find({}, {'_id': 0, 'Symbol': 1})
            symbols = [symbol['Symbol'] for symbol in symbols_cursor]
            if self.config.DEBUG_MODE:
                symbols = symbols[:10]  # Limit to 10 symbols for testing
                self.logger.info(f"DEBUG_MODE is ON. Processing only {len(symbols)} symbols.")
                print(f"DEBUG_MODE is ON. Processing only {len(symbols)} symbols.")

            with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
                futures = [executor.submit(self.handle_symbol, symbol) for symbol in symbols]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"Thread error: {str(e)}")
                        print(f"Thread error: {str(e)}")

            self.logger.info("Trading system run completed.")
            print("Trading system run completed.")

        except Exception as e:
            self.logger.error(f"Trading system error: {str(e)}")
            print(f"Trading system error: {str(e)}")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Set up root logger to capture any logs before BracketOrderTrader initializes
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Fetch environment variables with error checking
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    mongo_conn = os.getenv("MONGO_DB_CONN_STRING")
    alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API")
    teams_url = os.getenv("TEAMS_WEBHOOK_URL")

    missing_vars = []
    if not api_key:
        missing_vars.append("ALPACA_API_KEY")
    if not api_secret:
        missing_vars.append("ALPACA_SECRET_KEY")
    if not mongo_conn:
        missing_vars.append("MONGO_DB_CONN_STRING")
    if not alpha_vantage_key:
        missing_vars.append("ALPHA_VANTAGE_API")
    if not teams_url:
        missing_vars.append("TEAMS_WEBHOOK_URL")

    if missing_vars:
        logging.error(f"Missing environment variables: {', '.join(missing_vars)}")
        print(f"Missing environment variables: {', '.join(missing_vars)}")
        exit(1)

    # Initialize the trading system
    trader = BracketOrderTrader(
        api_key=api_key,
        api_secret=api_secret,
        mongo_conn=mongo_conn,
        alpha_vantage_key=alpha_vantage_key,
        teams_url=teams_url
    )

    # Optionally set DEBUG_MODE to True for testing
    trader.config.DEBUG_MODE = True  # Set to True for testing
    if trader.config.DEBUG_MODE:
        trader.logger.info("DEBUG_MODE is enabled.")
        print("DEBUG_MODE is enabled.")

    # Run the trading system
    trader.run()
