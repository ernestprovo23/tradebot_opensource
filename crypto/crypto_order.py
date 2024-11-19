import logging
from dotenv import load_dotenv
import pandas as pd
from pymongo import MongoClient
import alpaca_trade_api as tradeapi
import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from risk_strategy import RiskManagement, PortfolioManager

# Clear existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Enhanced logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('crypto_log.log', mode='w'),  # 'w' mode to clear previous logs
        logging.StreamHandler(sys.stdout)  # Explicitly write to stdout
    ]
)

# Immediate logging test
logging.info("Script started - Testing logging configuration")


def load_risk_params():
    """Load risk parameters with enhanced logging"""
    logging.info("Starting to load risk parameters")
    try:
        params_path = Path(__file__).parent / "risk_params.json"
        logging.debug(f"Looking for risk_params.json at: {params_path}")

        if not params_path.exists():
            logging.error(f"risk_params.json not found at {params_path}")
            return None

        with open(params_path, "r") as file:
            risk_params = json.load(file)
            logging.info(f"Risk parameters loaded: {risk_params}")
            return risk_params

    except Exception as e:
        logging.error(f"Error loading risk parameters: {str(e)}")
        return None


# Load environment variables with logging
logging.info("Loading environment variables")
load_dotenv()

# Verify environment variables
env_vars = {
    'ALPACA_API_KEY': os.getenv('ALPACA_API_KEY'),
    'ALPACA_SECRET_KEY': os.getenv('ALPACA_SECRET_KEY'),
    'MONGO_CONN_STRING': os.getenv('MONGO_DB_CONN_STRING')
}

missing_vars = [var for var, value in env_vars.items() if not value]
if missing_vars:
    logging.error(f"Missing environment variables: {missing_vars}")
    sys.exit(1)

logging.info("Environment variables loaded successfully")

# Initialize MongoDB connection
try:
    logging.info("Attempting to connect to MongoDB")
    client = MongoClient(env_vars['MONGO_CONN_STRING'])
    db = client.stock_data
    collection = db.crypto_data

    # Test connection
    client.server_info()
    logging.info("MongoDB connection successful")

    # Check collection contents
    doc_count = collection.count_documents({})
    logging.info(f"Found {doc_count} documents in crypto_data collection")

    # Sample document structure
    sample_doc = collection.find_one()
    if sample_doc:
        logging.info(f"Sample document structure: {list(sample_doc.keys())}")
    else:
        logging.warning("No documents found in collection")

except Exception as e:
    logging.error(f"MongoDB connection failed: {str(e)}")
    sys.exit(1)


def get_data_from_mongo():
    """Enhanced data retrieval with detailed logging"""
    logging.info("Starting data retrieval from MongoDB")
    try:
        # Get document count
        doc_count = collection.count_documents({})
        logging.info(f"Total documents in collection: {doc_count}")

        if doc_count == 0:
            logging.warning("No documents found in collection")
            return pd.DataFrame()

        # Retrieve documents
        documents = list(collection.find())
        logging.info(f"Retrieved {len(documents)} documents from MongoDB")

        # Convert to DataFrame
        df = pd.DataFrame(documents)
        logging.info(f"DataFrame created with shape: {df.shape}")
        logging.info(f"DataFrame columns: {df.columns.tolist()}")

        # Log sample data
        if not df.empty:
            logging.info(f"Sample data:\n{df.head()}")

        return df
    except Exception as e:
        logging.error(f"Error retrieving data from MongoDB: {str(e)}")
        return pd.DataFrame()


def initialize_trading_system():
    """Initialize trading system with enhanced logging"""
    logging.info("Starting trading system initialization")

    try:
        # Initialize Alpaca API
        logging.info("Initializing Alpaca API")
        api = tradeapi.REST(
            env_vars['ALPACA_API_KEY'],
            env_vars['ALPACA_SECRET_KEY'],
            base_url='https://paper-api.alpaca.markets'
        )

        # Test API connection
        account = api.get_account()
        logging.info(f"Alpaca API connection successful. Account status: {account.status}")

        # Load risk parameters
        risk_params = load_risk_params()
        if not risk_params:
            raise ValueError("Failed to load risk parameters")

        # Initialize components
        logging.info("Initializing PortfolioManager")
        manager = PortfolioManager(api)

        logging.info("Initializing RiskManagement")
        risk_management = RiskManagement(api, risk_params)

        logging.info("Initializing EnhancedCryptoTrader")
        trader = EnhancedCryptoTrader(api, risk_management, manager)

        logging.info("Trading system initialization completed successfully")
        return api, trader, risk_management

    except Exception as e:
        logging.error(f"Trading system initialization failed: {str(e)}")
        return None, None, None

def calculate_SMA(data, window=5):
    """Calculate Simple Moving Average"""
    try:
        if 'Close' not in data.columns:
            logging.error("'Close' column not found in data")
            return pd.Series(np.nan, index=data.index)
        return data['Close'].rolling(window=window).mean()
    except Exception as e:
        logging.error(f"Error calculating SMA: {e}")
        return pd.Series(np.nan, index=data.index)


def calculate_RSI(data, window=14):
    """Calculate Relative Strength Index"""
    try:
        if 'Close' not in data.columns:
            logging.error("'Close' column not found in data")
            return pd.Series(np.nan, index=data.index)

        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        logging.error(f"Error calculating RSI: {e}")
        return pd.Series(np.nan, index=data.index)


def calculate_MACD(data, short_window=12, long_window=26, signal_window=9):
    """Calculate MACD and Signal Line"""
    try:
        if 'Close' not in data.columns:
            logging.error("'Close' column not found in data")
            return pd.Series(np.nan, index=data.index), pd.Series(np.nan, index=data.index)

        ema_short = data['Close'].ewm(span=short_window, adjust=False).mean()
        ema_long = data['Close'].ewm(span=long_window, adjust=False).mean()
        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        return macd_line, signal_line
    except Exception as e:
        logging.error(f"Error calculating MACD: {e}")
        return pd.Series(np.nan, index=data.index), pd.Series(np.nan, index=data.index)


def get_data_from_mongo():
    """Enhanced data retrieval from MongoDB with proper error handling"""
    try:
        # Retrieve all documents
        documents = list(collection.find())
        if not documents:
            logging.warning("No documents found in MongoDB")
            return pd.DataFrame()

        # Convert to DataFrame
        data = pd.DataFrame(documents)

        # Verify required columns
        required_columns = ['Crypto', 'Quote', 'Close', 'Date']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logging.error(f"Missing required columns: {missing_columns}")
            return pd.DataFrame()

        # Process data
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date').reset_index(drop=True)

        # Create Symbol column
        data['Symbol'] = data.apply(lambda x: f"{x['Crypto']}{x['Quote']}", axis=1)

        logging.info(f"Retrieved {len(data)} records from MongoDB")
        logging.debug(f"Columns found: {data.columns.tolist()}")
        logging.debug(f"Sample data:\n{data.head()}")

        return data
    except Exception as e:
        logging.error(f"Error retrieving data from MongoDB: {e}")
        return pd.DataFrame()


class EnhancedCryptoTrader:
    def __init__(self, api, risk_management, manager):
        self.api = api
        self.risk_management = risk_management
        self.manager = manager
        self.MIN_TRADE_VALUE = 100
        self.MAX_POSITION_VALUE = float(risk_management.risk_params.get('max_position_size', 5000))
        self.MIN_SCORE_THRESHOLD = 3
        self.POSITION_SIZING_TIERS = {
            'tier1': {'max_price': 100, 'size_factor': 1.0},
            'tier2': {'max_price': 1000, 'size_factor': 0.75},
            'tier3': {'max_price': 10000, 'size_factor': 0.5},
            'tier4': {'max_price': float('inf'), 'size_factor': 0.25}
        }

    def calculate_technical_indicators(self, data):
        """Enhanced technical indicator calculation with validation"""
        try:
            if data.empty:
                logging.error("Empty dataset provided for technical analysis")
                return None

            if 'Close' not in data.columns:
                logging.error(f"'Close' column not found. Available columns: {data.columns.tolist()}")
                return None

            # Calculate indicators
            sma_short = calculate_SMA(data, 5)
            sma_long = calculate_SMA(data, 10)
            rsi = calculate_RSI(data)
            macd_line, signal_line = calculate_MACD(data)

            # Validate results
            if any(pd.isna([sma_short.iloc[-1], sma_long.iloc[-1], rsi.iloc[-1]])):
                logging.error("Invalid technical indicator calculations")
                return None

            indicators = {
                'sma_short': sma_short.iloc[-1],
                'sma_long': sma_long.iloc[-1],
                'rsi': rsi.iloc[-1],
                'macd': (macd_line, signal_line),
                'price': data['Close'].iloc[-1]
            }

            logging.debug(f"Calculated indicators: {indicators}")
            return indicators

        except Exception as e:
            logging.error(f"Error calculating technical indicators: {e}")
            return None

    def calculate_buy_score(self, technical_data):
        """Enhanced buy score calculation with detailed logging"""
        score = 0
        reasons = []

        try:
            # Trend analysis
            if technical_data['sma_short'] > technical_data['sma_long']:
                score += 1
                reasons.append("Upward trend detected")

            # RSI analysis
            rsi = technical_data['rsi']
            if 30 <= rsi <= 45:
                score += 1
                reasons.append(f"RSI in buy zone: {rsi:.2f}")
            elif rsi < 30:
                score += 2
                reasons.append(f"RSI oversold: {rsi:.2f}")

            # MACD analysis
            macd_line, signal_line = technical_data['macd']
            if macd_line.iloc[-1] > signal_line.iloc[-1]:
                score += 1
                reasons.append("MACD above signal line")
                if macd_line.iloc[-1] > 0:
                    score += 1
                    reasons.append("MACD positive")

            # Price vs SMA
            if technical_data['price'] > technical_data['sma_short']:
                score += 1
                reasons.append("Price above short-term SMA")

            logging.info(f"Buy score calculation - Score: {score}, Reasons: {reasons}")
            return score

        except Exception as e:
            logging.error(f"Error in buy score calculation: {e}")
            return 0

    def process_buy_signal(self, symbol, data_symbol, portfolio_value):
        """Process buy signal with enhanced allocation checks"""
        logging.info(f"\nProcessing buy signal for {symbol}")
        logging.debug(f"Data shape for {symbol}: {data_symbol.shape}")

        try:
            # Calculate technical indicators
            technical_data = self.calculate_technical_indicators(data_symbol)
            if not technical_data:
                logging.warning(f"No technical data available for {symbol}")
                return False

            # Calculate buy score
            score = self.calculate_buy_score(technical_data)
            logging.info(f"Buy score for {symbol}: {score}")

            if score >= self.MIN_SCORE_THRESHOLD:
                current_price = self.risk_management.get_current_price(symbol)
                if not current_price:
                    logging.error(f"Could not get current price for {symbol}")
                    return False

                logging.info(f"Current price for {symbol}: ${current_price:.2f}")

                # Get current crypto allocation
                try:
                    positions = self.api.list_positions()
                    total_portfolio_value = float(self.api.get_account().portfolio_value)
                    current_crypto_value = sum(
                        float(p.market_value) for p in positions
                        if p.symbol.endswith('USD')
                    )
                    current_allocation = current_crypto_value / total_portfolio_value
                    max_crypto_equity = float(self.risk_management.risk_params['max_crypto_equity'])

                    logging.info(f"Current crypto allocation: {current_allocation:.2%}")
                    logging.info(f"Maximum allowed crypto equity: ${max_crypto_equity:,.2f}")
                    logging.info(f"Current crypto value: ${current_crypto_value:,.2f}")

                    # Calculate available allocation
                    available_allocation = max_crypto_equity - current_crypto_value
                    logging.info(f"Available allocation: ${available_allocation:,.2f}")

                    if available_allocation <= 0:
                        logging.info("No available allocation for crypto")
                        return False

                    # Calculate maximum quantity based on available allocation
                    max_quantity = available_allocation / current_price
                    quantity = min(
                        max_quantity,
                        self.calculate_position_size(symbol, current_price, portfolio_value)
                    )

                    # Ensure minimum transaction value
                    if quantity * current_price < self.MIN_TRADE_VALUE:
                        logging.info(f"Trade value too small (${quantity * current_price:.2f})")
                        return False

                except Exception as e:
                    logging.error(f"Error calculating allocation for {symbol}: {e}")
                    return False

                quantity = int(quantity)  # Ensure integer quantity
                logging.info(f"Final calculated quantity for {symbol}: {quantity}")

                # Validate trade with updated quantity
                if self.risk_management.validate_trade(symbol, quantity, "buy"):
                    try:
                        order = self.api.submit_order(
                            symbol=symbol,
                            qty=quantity,
                            side='buy',
                            type='limit',
                            time_in_force='gtc',
                            limit_price=round(current_price * 1.002, 2)
                        )
                        logging.info(f"Buy order placed for {quantity} {symbol} at ${current_price:.2f}")
                        return True
                    except Exception as e:
                        logging.error(f"Error placing buy order for {symbol}: {e}")
                        return False
                else:
                    logging.info(f"Trade validation failed for {symbol}")
            else:
                logging.info(f"Buy conditions not met for {symbol} (score: {score})")

            return False

        except Exception as e:
            logging.error(f"Error in buy signal processing for {symbol}: {e}")
            return False

    def calculate_position_size(self, symbol, current_price, portfolio_value):
        """Calculate position size with enhanced allocation controls"""
        try:
            # Determine sizing tier
            tier = None
            for t, params in self.POSITION_SIZING_TIERS.items():
                if current_price <= params['max_price']:
                    tier = t
                    break

            if not tier:
                tier = 'tier4'

            size_factor = self.POSITION_SIZING_TIERS[tier]['size_factor']

            # Get risk parameters
            max_position = min(
                self.MAX_POSITION_VALUE,
                float(self.risk_management.risk_params.get('max_position_size', 5000))
            )

            # Calculate base position size
            base_size = (portfolio_value * 0.01) * size_factor  # 1% of portfolio * tier factor
            position_size = min(base_size, max_position)

            # Calculate quantity
            quantity = position_size / current_price

            # Apply minimum trade value
            if quantity * current_price < self.MIN_TRADE_VALUE:
                quantity = self.MIN_TRADE_VALUE / current_price

            # Round down to integer
            quantity = int(quantity)

            logging.info(f"""Position size calculation for {symbol}:
                Tier: {tier}
                Size Factor: {size_factor}
                Base Size: ${base_size:,.2f}
                Final Position Size: ${position_size:,.2f}
                Quantity: {quantity}""")

            return quantity

        except Exception as e:
            logging.error(f"Error calculating position size for {symbol}: {e}")
            return 0


def process_symbol_data(symbol_data, api, risk_management):
    """Process trading signals for a single symbol with enhanced risk management"""
    try:
        symbol = symbol_data['Symbol'].iloc[0]
        logging.info(f"\nAnalyzing {symbol}")

        # Check for buy signals
        if symbol_data['Buy Signal'].iloc[-1]:
            current_price = symbol_data['Close'].iloc[-1]
            rsi = symbol_data['RSI'].iloc[-1] if 'RSI' in symbol_data else None
            macd = symbol_data['MACD'].iloc[-1] if 'MACD' in symbol_data else None
            macd_signal = symbol_data['MACD Signal'].iloc[-1] if 'MACD Signal' in symbol_data else None

            logging.info(f"{symbol} Analysis:")
            logging.info(f"Current Price: ${current_price:.2f}")
            if all(x is not None for x in [rsi, macd, macd_signal]):
                logging.info(f"RSI: {rsi:.2f}")
                logging.info(f"MACD: {macd:.2f}")
                logging.info(f"MACD Signal: {macd_signal:.2f}")

                # Enhanced trading logic with risk validation
                if (30 <= rsi <= 70 and  # RSI in reasonable range
                        macd > macd_signal and  # MACD crossover
                        current_price > 0):

                    # Calculate position size based on risk parameters
                    quantity = risk_management.calculate_quantity(symbol)

                    if quantity > 0 and risk_management.validate_trade(symbol, quantity, "buy"):
                        logging.info(f"Buy signal confirmed for {symbol}")
                        try:
                            order = api.submit_order(
                                symbol=symbol,
                                qty=quantity,
                                side='buy',
                                type='limit',
                                time_in_force='gtc',
                                limit_price=round(current_price * 1.002, 2)
                            )
                            logging.info(f"Order placed for {symbol}: {order}")
                        except Exception as e:
                            logging.error(f"Order failed for {symbol}: {e}")

        # Check for sell signals
        if symbol_data['Sell Signal'].iloc[-1]:
            try:
                position = api.get_position(symbol)
                current_price = symbol_data['Close'].iloc[-1]

                if float(position.unrealized_plpc) > 0.02:  # 2% profit
                    if risk_management.validate_trade(symbol, position.qty, "sell"):
                        logging.info(f"Taking profit on {symbol}")
                        api.submit_order(
                            symbol=symbol,
                            qty=position.qty,
                            side='sell',
                            type='limit',
                            time_in_force='gtc',
                            limit_price=round(current_price * 0.998, 2)
                        )
            except Exception as e:
                if "position does not exist" not in str(e).lower():
                    logging.error(f"Error processing sell signal for {symbol}: {e}")

    except Exception as e:
        logging.error(f"Error processing symbol data: {e}")


def main():
    """Direct execution of trading logic without maintenance window"""
    logging.info("\n" + "=" * 50 + "\nStarting trading execution\n" + "=" * 50)

    try:
        # Initialize trading system
        api, trader, risk_management = initialize_trading_system()
        if not all([api, trader, risk_management]):
            logging.error("Failed to initialize trading system components")
            return

        # Get market data
        data = get_data_from_mongo()
        if data.empty:
            logging.warning("No data available for trading")
            return

        # Process data for trading
        data['Symbol'] = data.apply(lambda x: f"{x['Crypto']}{x['Quote']}", axis=1)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values(['Symbol', 'Date'])

        # Process each symbol
        symbols_processed = 0
        for symbol, symbol_data in data.groupby('Symbol'):
            try:
                process_symbol_data(symbol_data, api, risk_management)
                symbols_processed += 1
            except Exception as e:
                logging.error(f"Error processing {symbol}: {e}")
                continue

        logging.info(f"Completed processing {symbols_processed} symbols")

    except KeyboardInterrupt:
        logging.info("Trading system stopped by user")
    except Exception as e:
        logging.error(f"Critical error in main execution: {str(e)}")
    finally:
        logging.info("Trading system execution completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)