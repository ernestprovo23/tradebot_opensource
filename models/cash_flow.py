import os
import requests
from dotenv import load_dotenv
import logging
from pymongo import MongoClient
from datetime import datetime
import concurrent.futures
import time

# Setup logging
logging.basicConfig(filename='../cash_flow_logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API')
MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Setup MongoDB connection
client = MongoClient(MONGO_CONN_STRING)
db = client['stock_data']  # Accessing the 'stock_data' database
selected_pairs_collection = db['selected_pairs']  # Accessing the 'selected_pairs' collection
bad_performers_collection = client['trading_db']['bad_performers']  # Accessing the 'bad_performers' collection

# Global variable to count API calls
api_calls_count = 0

def fetch_cash_flow(symbol):
    global api_calls_count

    # Rate limiting: 75 calls per minute
    if api_calls_count >= 75:
        logging.info("API call limit reached, waiting...")
        time.sleep(60)  # Wait for 60 seconds before proceeding
        api_calls_count = 0  # Reset the counter

    url = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
    response = requests.get(url)
    api_calls_count += 1

    if response.status_code == 200:
        data = response.json()
        logging.info(f"Successfully fetched cash flow for {symbol}")
        return data
    else:
        logging.error(f"Failed to fetch cash flow for {symbol} with status code {response.status_code}")
        return None

def process_symbol(document):
    symbol = document.get('symbol')
    if symbol:
        cash_flow_data = fetch_cash_flow(symbol)
        if cash_flow_data:
            logging.info(f"Inserting/Updating data for {symbol}")
            cash_flow_collection = db['cash_flow']
            document_to_upsert = {
                'symbol': symbol,
                'cash_flow': cash_flow_data,
                'datetime_imported': datetime.utcnow()
            }
            cash_flow_collection.update_one(
                {'symbol': symbol},
                {'$set': document_to_upsert},
                upsert=True
            )
        else:
            logging.error(f"Failed to fetch data for {symbol}")

# Clear the destination collection before processing
cash_flow_collection = db['cash_flow']
delete_result = cash_flow_collection.delete_many({})
logging.info(f"Cleared {delete_result.deleted_count} records from the cash_flow collection.")

# Fetch bad performers
bad_performers_symbols = bad_performers_collection.distinct('symbol')

# Fetch symbols from MongoDB 'selected_pairs' collection, omitting bad performers
symbols = [doc for doc in selected_pairs_collection.find() if doc.get('symbol') not in bad_performers_symbols]

# Using ThreadPoolExecutor to manage multiple threads
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(process_symbol, symbol) for symbol in symbols]

    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as exc:
            logging.error(f'Generated an exception: {exc}')
