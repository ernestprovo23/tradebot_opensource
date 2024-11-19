import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from pymongo import MongoClient, UpdateOne
import alpaca_trade_api as tradeapi
from risk_strategy import RiskManagement, risk_params, PortfolioManager
from cachetools import TTLCache
import requests
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from typing import Dict, List, Optional

# Load environment variables
load_dotenv()


# Global configuration
class Config:
    ALPHA_VANTAGE_API = os.getenv('ALPHA_VANTAGE_API')
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
    MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')
    TEAMS_WEBHOOK_URL = os.getenv('TEAMS_WEBHOOK_URL')
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 0.3
    CACHE_TTL = 300
    MAX_WORKERS = 5


class CryptoAggregator:
    def __init__(self):
        # Initialize logging
        logging.basicConfig(
            filename='logfile.log',
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )

        # Initialize connections
        self.client = MongoClient(Config.MONGO_CONN_STRING)
        self.db = self.client.stock_data
        self.collection = self.db.crypto_data

        # Initialize API clients
        self.api = tradeapi.REST(
            Config.ALPACA_API_KEY,
            Config.ALPACA_SECRET_KEY,
            base_url='https://paper-api.alpaca.markets'
        )

        # Initialize managers
        self.portfolio_manager = PortfolioManager(self.api)
        self.risk_management = RiskManagement(self.api, risk_params)

        # Initialize cache
        self.exchange_rate_cache = TTLCache(maxsize=100, ttl=Config.CACHE_TTL)

        # Define supported cryptocurrencies
        self.cryptocurrencies = {
            "USD": ["AAVE", "AVAX", "BAT", "BCH", "BTC", "CRV", "DOGE", "DOT",
                    "ETH", "GRT", "LINK", "LTC", "MKR", "SHIB", "SUSHI", "UNI",
                    "USDC", "USDT", "XTZ", "YFI"]
        }

    def fetch_alpha_vantage_price(self, base_currency: str, quote_currency: str) -> Optional[float]:
        """Fetch price from Alpha Vantage with retry logic"""
        cache_key = f"{base_currency}/{quote_currency}"
        if cache_key in self.exchange_rate_cache:
            return self.exchange_rate_cache[cache_key]

        for attempt in range(Config.MAX_RETRIES):
            try:
                url = (
                    f"https://www.alphavantage.co/query"
                    f"?function=CURRENCY_EXCHANGE_RATE"
                    f"&from_currency={base_currency}"
                    f"&to_currency={quote_currency}"
                    f"&apikey={Config.ALPHA_VANTAGE_API}"
                )
                response = requests.get(url)
                data = response.json()

                if "Realtime Currency Exchange Rate" in data:
                    rate = float(data["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
                    self.exchange_rate_cache[cache_key] = rate
                    return rate

            except Exception as e:
                logging.error(
                    f"Error fetching price from Alpha Vantage for "
                    f"{base_currency}/{quote_currency}: {e}"
                )
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(Config.BACKOFF_FACTOR * (2 ** attempt))

        return None

    def get_current_prices(self) -> pd.DataFrame:
        """Fetch current prices using ThreadPoolExecutor"""
        data = []

        def fetch_price(quote_currency, base_currency):
            alpha_vantage_symbol = f"{base_currency}/{quote_currency}"
            price = self.fetch_alpha_vantage_price(base_currency, quote_currency)
            if price:
                return {
                    "Crypto": base_currency,
                    "Quote": quote_currency,
                    "Symbol": alpha_vantage_symbol,
                    "Close": price,
                    "Date": datetime.now()
                }
            return None

        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            futures = []
            for quote_currency, base_currencies in self.cryptocurrencies.items():
                for base_currency in base_currencies:
                    futures.append(
                        executor.submit(fetch_price, quote_currency, base_currency)
                    )

            for future in futures:
                result = future.result()
                if result:
                    data.append(result)

        return pd.DataFrame(data)

    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Calculate Bollinger Bands
        df['Mean'] = df['Close'].rolling(window=20).mean()
        df['Std Dev'] = df['Close'].rolling(window=20).std()
        df['Upper Band'] = df['Mean'] + (df['Std Dev'] * 2)
        df['Lower Band'] = df['Mean'] - (df['Std Dev'] * 2)

        # Calculate Momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(1)

        # Calculate RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        avg_loss = avg_loss.replace(0, np.nan)
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(0)

        # Calculate MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        return df

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading signals"""
        df = self.calculate_technical_indicators(df)

        # Generate signals
        df['Buy Signal'] = (
                                   (df['Close'] < df['Lower Band']) &
                                   (df['Momentum'] > 0)
                           ) | (df['RSI'] < 30) | (df['MACD'] > df['MACD Signal'])

        df['Sell Signal'] = (
                                    (df['Close'] > df['Upper Band']) &
                                    (df['Momentum'] < 0)
                            ) | (df['RSI'] > 70) | (df['MACD'] < df['MACD Signal'])

        df['Mean Reversion Signal'] = np.where(
            df['Close'] < df['Lower Band'],
            'Buy',
            np.where(df['Close'] > df['Upper Band'], 'Sell', 'Hold')
        )

        df['Momentum Signal'] = np.where(
            df['Momentum'] > 0,
            'Buy',
            np.where(df['Momentum'] < 0, 'Sell', 'Hold')
        )

        return df

    def update_mongodb(self, df: pd.DataFrame) -> None:
        """Update MongoDB with signals"""
        operations = []

        for _, row in df.iterrows():
            doc = row.to_dict()
            doc['Date'] = pd.to_datetime(doc['Date'])
            doc['Signal_Updated_At'] = datetime.now()

            if self.validate_document(doc):
                operations.append(
                    UpdateOne(
                        {
                            'Date': doc['Date'],
                            'Crypto': doc['Crypto'],
                            'Quote': doc['Quote']
                        },
                        {'$set': doc},
                        upsert=True
                    )
                )
            else:
                logging.error(f"Invalid document: {doc}")

        if operations:
            self.collection.bulk_write(operations)

    @staticmethod
    def validate_document(doc: Dict) -> bool:
        """Validate document fields"""
        required_fields = [
            'Buy Signal', 'Sell Signal',
            'Mean Reversion Signal', 'Momentum Signal'
        ]

        return all(
            field in doc for field in required_fields
        ) and isinstance(doc['Buy Signal'], bool) and isinstance(doc['Sell Signal'], bool)

    def process_signals(self) -> None:
        """Main processing function"""
        try:
            current_prices_df = self.get_current_prices()
            if current_prices_df.empty:
                logging.warning("No current prices retrieved.")
                return

            signals_df = self.calculate_signals(current_prices_df)
            self.update_mongodb(signals_df)

            logging.info(f"Processed {len(signals_df)} symbols")

        except Exception as e:
            logging.error(f"Error in process_signals: {e}")
            raise


def main():
    logging.info('Script started')
    aggregator = CryptoAggregator()
    aggregator.process_signals()
    logging.info('Script ended')
    print('Script ended')


if __name__ == "__main__":
    main()