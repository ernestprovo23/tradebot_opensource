import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# MongoDB connection
client = MongoClient(MONGO_DB_CONN_STRING)
db = client['stock_data']

# Source and destination collections
aggregated_collection = db['aggregated_stock_data']
features_collection = client['silvertables']['ML_features']

def fetch_stock_data(symbol):
    """
    Fetch stock data for a given stock symbol.
    """
    print(f"Fetching data for symbol: {symbol}")
    query = {'symbol': symbol}
    # Adjust projection to match actual key names in the documents
    cursor = aggregated_collection.find(query, {'_id': 0, 'symbol': 1, 'TIME_SERIES_DAILY_data': 1})
    data = list(cursor)
    if data:
        # Adjust to the correct key, respecting case sensitivity
        if 'TIME_SERIES_DAILY_data' in data[0]:
            time_series_data = data[0]['TIME_SERIES_DAILY_data']
            # Ensure your DataFrame creation logic matches the structure of the time series data
            df = pd.DataFrame(time_series_data)
            # Assuming 'timestamp' and 'close_price' are the fields you're interested in
            df.rename(columns={'timestamp': 'date', 'close_price': 'close'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            df['close'] = pd.to_numeric(df['close'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            return df
        else:
            print(f"No 'TIME_SERIES_DAILY_data' found for symbol: {symbol}.")
            return None
    else:
        print(f"No data found for symbol: {symbol}")
        return None

def calculate_technical_indicators(df):
    """
    Calculate technical indicators like Moving Averages (MA), Exponential Moving Averages (EMA),
    Relative Strength Index (RSI), etc., that might be useful for the model.
    """
    print("Calculating technical indicators...")
    # Moving Averages
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()

    # Exponential Moving Averages
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    print("Technical indicators calculated successfully.")
    return df

def prepare_features(symbol):
    """
    Prepare features for model training or prediction.
    """
    df_stock_data = fetch_stock_data(symbol)
    if df_stock_data is not None:
        df_features = calculate_technical_indicators(df_stock_data)
        print(f"Features prepared for symbol: {symbol}")
        return df_features
    else:
        print(f"Failed to prepare features for symbol: {symbol}")
        return None

def process_all_symbols():
    """
    Process all unique symbols in the aggregated_stock_data collection.
    """
    symbols = aggregated_collection.distinct('symbol')
    print(f"Found {len(symbols)} symbols to process.")
    for symbol in symbols:
        print(f"Processing symbol: {symbol}")
        df_features = prepare_features(symbol)
        if df_features is not None:
            features_data = {
                'symbol': symbol,
                'features': df_features.reset_index().to_dict(orient='records'),
                'datetime_processed': datetime.now()
            }
            features_collection.insert_one(features_data)
            print(f"Features stored for symbol: {symbol} in ML_features collection.")
        else:
            print(f"No valid data to store for symbol: {symbol}")

# Example usage
process_all_symbols()
