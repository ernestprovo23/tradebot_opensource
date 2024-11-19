import logging
import requests
from datetime import datetime, timedelta
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(filename='logfile.log',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# MongoDB setup
client = MongoClient(MONGO_DB_CONN_STRING)
stock_data_db = client.stock_data
predictions_db = client.predictions
options_contracts_collection = stock_data_db.options_contracts

# Function to get the predictions collection name based on current month and year
def get_predictions_collection_name():
    current_date = datetime.now()
    month_number = current_date.strftime('%B')
    year = current_date.year
    return f"{month_number}_{year}"

# Access the predictions collection for the current month and year
current_predictions_collection_name = get_predictions_collection_name()
current_predictions_collection = predictions_db[current_predictions_collection_name]
selected_pairs_collection = predictions_db[current_predictions_collection_name]

print(f"Using predictions collection: {current_predictions_collection_name}")

# Alpaca API headers
headers = {
    "Accept": "application/json",
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
}


def fetch_prediction_date(symbol):
    """Fetch the prediction date for a given symbol."""
    prediction = current_predictions_collection.find_one({'symbol': symbol}, sort=[('entry_date', -1)])
    if prediction:
        prediction_date = prediction['prediction_date'].strftime('%Y-%m-%d')
        print(f"Prediction date for {symbol}: {prediction_date}")
        return prediction_date
    else:
        print(f"No prediction date found for {symbol}")
        return None


def fetch_options_for_symbol(symbol, prediction_date):
    """Fetch options contracts for a given symbol using the prediction date."""
    pred_date = datetime.strptime(prediction_date, '%Y-%m-%d')

    # Calculate the exact expiration date 23 days from the current date
    current_date = datetime.now()
    target_offset = 75
    target_date = current_date + timedelta(days=target_offset)

    # Define the specific expiration date range around the target date (within 5 days)
    min_expiration = target_date - timedelta(days=5)
    max_expiration = target_date + timedelta(days=120)

    print(f"Prediction date for {symbol}: {pred_date.strftime('%Y-%m-%d')}")
    print(f"Target expiration date (90 days from current date): {target_date.strftime('%Y-%m-%d')}")
    print(f"Checking options within the date range: {min_expiration.strftime('%Y-%m-%d')} to {max_expiration.strftime('%Y-%m-%d')}")

    for days_offset in range(-25, 36):  # Check within 5 days before and after the target date
        current_expiration_date = target_date + timedelta(days=days_offset)

        url = f"https://paper-api.alpaca.markets/v2/options/contracts?underlying_symbols={symbol}&expiration_date={current_expiration_date.strftime('%Y-%m-%d')}&status=active&style=american&limit=1000"

        print(f"Checking options for {symbol} on {current_expiration_date.strftime('%Y-%m-%d')}")

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if 'option_contracts' in data and data['option_contracts']:
                option_contracts = data['option_contracts']
                print(f"Contracts found for {symbol} on {current_expiration_date.strftime('%Y-%m-%d')}: {len(option_contracts)}")
                return option_contracts  # Return immediately after finding the first valid chain
        else:
            print(f"Failed to fetch options for {symbol} on {current_expiration_date.strftime('%Y-%m-%d')}. Response: {response.status_code}")

    print(f"No contracts found for {symbol} within the specified date range.")
    return []


def store_options_data(symbol, options_data):
    """Store fetched options data in MongoDB."""
    results = []
    datetime_now = datetime.now()  # Current timestamp for datetime_imported

    for contract in options_data:
        contract['fetched_for_symbol'] = symbol
        contract['datetime_imported'] = datetime_now  # Add datetime_imported field

        # Convert and ensure the correct data types for specific fields
        contract['strike_price'] = float(contract.get('strike_price')) if contract.get('strike_price') else None
        contract['expiration_date'] = datetime.strptime(contract['expiration_date'], '%Y-%m-%d') if contract.get('expiration_date') else None
        contract['multiplier'] = float(contract.get('multiplier')) if contract.get('multiplier') else None
        contract['size'] = int(contract.get('size')) if contract.get('size') else None
        contract['open_interest'] = float(contract.get('open_interest')) if contract.get('open_interest') else None  # Fixed the parentheses here
        contract['close_price'] = float(contract.get('close_price')) if contract.get('close_price') else None
        contract['last_price'] = float(contract.get('last_price')) if contract.get('last_price') else None  # Ensure 'last_price' is stored
        contract['underlying_price'] = float(contract.get('underlying_price')) if contract.get('underlying_price') else None  # Ensure 'underlying_price' is stored

        if contract.get('open_interest_date'):
            contract['open_interest_date'] = datetime.strptime(contract['open_interest_date'], '%Y-%m-%d')
        else:
            contract['open_interest_date'] = None

        if contract.get('close_price_date'):
            contract['close_price_date'] = datetime.strptime(contract['close_price_date'], '%Y-%m-%d')
        else:
            contract['close_price_date'] = None

        # Check if the contract already exists in the database to avoid duplicates
        existing_contract = options_contracts_collection.find_one({
            'symbol': contract['symbol'],
            'expiration_date': contract['expiration_date'],
            'strike_price': contract['strike_price']
        })

        if not existing_contract:
            result = options_contracts_collection.insert_one(contract)
            results.append(result.inserted_id)
        else:
            print(f"Contract for {symbol} already exists in the database, skipping insert.")

    if results:
        print(f"Stored options data for {symbol} in MongoDB. Document IDs: {results}")
    else:
        print(f"No new data stored for {symbol}.")
    return results


def main():
    symbols = selected_pairs_collection.distinct('symbol')
    logging.info(f"Symbols to process: {symbols}")
    symbols_no_options = []
    symbols_with_options = []

    for symbol in symbols:
        logging.info(f"Processing symbol: {symbol}")
        print(f"Processing symbol: {symbol}")
        prediction_date = fetch_prediction_date(symbol)
        if not prediction_date:
            print(f"No prediction date found for {symbol}")
            symbols_no_options.append(symbol)
            continue

        options_data = fetch_options_for_symbol(symbol, prediction_date)
        if options_data:
            result_ids = store_options_data(symbol, options_data)
            if result_ids:
                print(f"Options data for {symbol} stored successfully with IDs: {result_ids}")
                symbols_with_options.append(symbol)
            else:
                print(f"No options data stored for {symbol}")
                symbols_no_options.append(symbol)
        else:
            print(f"No options data found for {symbol} within the specified date range.")
            symbols_no_options.append(symbol)

    print(f"Symbols with options found: {list(set(symbols_with_options))}")
    print(f"Symbols without options found: {list(set(symbols_no_options))}")
    print(f"Total options contracts stored: {options_contracts_collection.count_documents({})}")
    logging.debug(f"Symbols with options found: {list(set(symbols_with_options))}")
    logging.debug(f"Symbols without options found: {list(set(symbols_no_options))}")
    logging.debug(f"Total options contracts stored: {options_contracts_collection.count_documents({})}")
    stock_data_db.symbols_with_options.insert_one({"symbols": list(set(symbols_with_options))})
    stock_data_db.symbols_without_options.insert_one({"symbols": list(set(symbols_no_options))})

if __name__ == "__main__":
    main()
