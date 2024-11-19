import os
import requests
from dotenv import load_dotenv
import logging
from pymongo import MongoClient
import concurrent.futures
import time
from datetime import datetime

# Setup logging
logging.basicConfig(filename='../logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API')
MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Setup MongoDB connection
client = MongoClient(MONGO_CONN_STRING)
db = client['stock_data']  # Accessing the 'stock_data' database

# Accessing the 'selected_pairs' collection to fetch symbols
selected_pairs_collection = db['selected_pairs']

# Global variable to count API calls
api_calls_count = 0

def fetch_balance_sheet(symbol):
    global api_calls_count

    # Rate limiting: 150 calls per minute
    if api_calls_count >= 150:
        logging.info("API call limit reached, waiting...")
        time.sleep(60)  # Wait for 60 seconds before proceeding
        api_calls_count = 0  # Reset the counter

    url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
    response = requests.get(url)
    api_calls_count += 1

    if response.status_code == 200:
        data = response.json()
        logging.info(f"Successfully fetched balance sheet for {symbol}")
        return data
    else:
        logging.error(f"Failed to fetch balance sheet for {symbol} with status code {response.status_code}")
        return None


def process_symbol(document):
    symbol = document.get('symbol')
    if symbol:
        logging.info(f"Processing symbol: {symbol}")
        balance_sheet_data = fetch_balance_sheet(symbol)
        if balance_sheet_data:
            balance_sheet_collection = db['balance_sheet']
            document_to_upsert = {
                'symbol': symbol,
                'balance_sheet': balance_sheet_data,
                'datetime_imported': datetime.utcnow()
            }
            balance_sheet_collection.update_one(
                {'symbol': symbol},
                {'$set': document_to_upsert},
                upsert=True
            )
            logging.info(f"Inserted/Updated data for {symbol}")
        else:
            logging.error(f"Failed to fetch data for {symbol}")


# Clear the destination collection before processing
balance_sheet_collection = db['balance_sheet']
delete_result = balance_sheet_collection.delete_many({})
logging.info(f"Cleared {delete_result.deleted_count} records from the balance_sheet collection.")

# Fetch symbols from MongoDB 'selected_pairs' collection
symbols = list(selected_pairs_collection.find())

# Using ThreadPoolExecutor to manage multiple threads
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(process_symbol, symbol) for symbol in symbols]

    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as exc:
            logging.error(f'Generated an exception: {exc}')
