import os
from time import sleep
import requests
import pymongo
from dotenv import load_dotenv
from datetime import datetime
import logging
import time
import concurrent.futures
from pymongo import UpdateOne

# Setup logging
logging.basicConfig(filename='../logfile.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API')
MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

FUNCTIONS = ["TIME_SERIES_DAILY"]

def fetch_unique_symbols(mongo_conn_string):
    client = pymongo.MongoClient(mongo_conn_string)
    db = client['stock_data']
    collection = db['selected_pairs']
    unique_symbols = collection.distinct('symbol')
    client.close()
    return unique_symbols

def fetch_bad_performers(mongo_conn_string):
    client = pymongo.MongoClient(mongo_conn_string)
    db = client['trading_db']
    collection = db['bad_performers']
    bad_performers = collection.distinct('symbol')
    client.close()
    return bad_performers

def fetch_existing_dates(symbol, function, mongo_client):
    db = mongo_client['stock_data']
    collection = db['aggregated_stock_data']
    existing_dates = collection.distinct('timestamp', {'symbol': symbol, 'function': f'{function}_data'})
    return set(existing_dates)

def fetch_data(symbol, api_key, existing_dates):
    data_document = {'symbol': symbol, 'datetime_imported': datetime.now()}
    BASE_URL = "https://www.alphavantage.co/query"
    logging.debug(f"Fetching data for symbol: {symbol}")

    for function in FUNCTIONS:
        url = f"{BASE_URL}?function={function}&symbol={symbol}&apikey={api_key}&outputsize=full"

        retries = 3
        while retries > 0:
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                logging.error(f"Error fetching data for {symbol} with {function}: {e}")
                retries -= 1
                if retries == 0:
                    logging.error(f"Failed to fetch data for {symbol} with {function} after 3 attempts")
                    return None
                sleep(5)

        data = r.json()
        time_series_key = next((key for key in data if 'Time Series' in key or 'Global Quote' in key), None)

        if time_series_key:
            time_series_data = data.get(time_series_key, {})
            closing_prices = [
                {'timestamp': timestamp, 'close_price': float(values.get('4. close'))}
                for timestamp, values in time_series_data.items()
                if '4. close' in values and timestamp not in existing_dates
            ]
            data_document[f'{function}_data'] = closing_prices
            logging.debug(f"Fetched {len(closing_prices)} new entries for {symbol} with {function}")
        else:
            logging.error(f"Failed to fetch or parse data for {symbol} with {function}. Missing expected key.")
            data_document[f'{function}_data'] = None

        sleep(12)

    return data_document

def store_data_in_mongo(client, data_document, bad_performers):
    if data_document['symbol'] in bad_performers:
        logging.info(f"Skipping storage for bad performer: {data_document['symbol']}")
        return

    db = client['stock_data']
    collection = db['aggregated_stock_data']
    bulk_operations = []

    for function, new_data in data_document.items():
        if function.endswith('_data') and new_data:
            bulk_operations.extend([
                UpdateOne(
                    {'symbol': data_document['symbol'], 'timestamp': entry['timestamp'], 'function': function},
                    {'$set': {'symbol': data_document['symbol'], 'timestamp': entry['timestamp'], 'close_price': entry['close_price'], 'function': function, 'datetime_imported': datetime.now()}},
                    upsert=True
                ) for entry in new_data
            ])

            if len(bulk_operations) >= 500:
                try:
                    result = collection.bulk_write(bulk_operations)
                    logging.debug(f"Bulk write result for {data_document['symbol']}: {result.bulk_api_result}")
                    bulk_operations = []
                except Exception as e:
                    logging.error(f"Error during bulk write for {data_document['symbol']}: {e}")

    if bulk_operations:
        try:
            result = collection.bulk_write(bulk_operations)
            logging.debug(f"Final bulk write result for {data_document['symbol']}: {result.bulk_api_result}")
        except Exception as e:
            logging.error(f"Error during final bulk write for {data_document['symbol']}: {e}")

def remove_bad_performers_from_db(client, bad_performers):
    db = client['stock_data']
    collection = db['aggregated_stock_data']
    result = collection.delete_many({'symbol': {'$in': bad_performers}})
    logging.debug(f"Removed {result.deleted_count} documents for bad performers.")

def process_ticker(ticker, mongo_client, bad_performers):
    try:
        logging.info(f"Processing data for {ticker}...")
        existing_dates = fetch_existing_dates(ticker, FUNCTIONS[0], mongo_client)
        data_document = fetch_data(ticker, ALPHA_VANTAGE_API_KEY, existing_dates)
        if data_document:
            store_data_in_mongo(mongo_client, data_document, bad_performers)
        else:
            logging.warning(f"No data fetched for {ticker}")
    except Exception as e:
        logging.error(f"An error occurred during processing {ticker}: {e}")

def wipe_aggregated_data(client):
    db = client['stock_data']
    collection = db['aggregated_stock_data']
    result = collection.delete_many({})
    logging.debug(f"Wiped {result.deleted_count} documents from aggregated_stock_data collection")

if __name__ == "__main__":
    logging.info("Starting the data fetching process...")

    mongo_client = pymongo.MongoClient(MONGO_CONN_STRING)
    logging.info("MongoDB client established.")

    wipe_aggregated_data(mongo_client)
    logging.info("Existing data wiped from aggregated_stock_data collection.")

    bad_performers = fetch_bad_performers(MONGO_CONN_STRING)
    logging.info(f"Fetched bad performers list: {bad_performers}")

    remove_bad_performers_from_db(mongo_client, bad_performers)
    logging.info("Removed bad performers from the database.")

    unique_symbols = fetch_unique_symbols(MONGO_CONN_STRING)
    logging.info(f"Fetched {len(unique_symbols)} unique symbols.")

    api_calls_made = 0
    start_time = time.time()

    logging.info("Beginning processing of ticker symbols...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for ticker in unique_symbols:
            if api_calls_made >= 150:
                elapsed_time = time.time() - start_time
                if elapsed_time < 60:
                    sleep_time = 60 - elapsed_time
                    logging.info(f"API limit reached, taking a break for {sleep_time:.2f} seconds.")
                    time.sleep(sleep_time)

                api_calls_made = 0
                start_time = time.time()

            futures.append(executor.submit(process_ticker, ticker, mongo_client, bad_performers))
            api_calls_made += 1

        concurrent.futures.wait(futures)

    mongo_client.close()
    logging.info("Data fetching and storage process completed.")
