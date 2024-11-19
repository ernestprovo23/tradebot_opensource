import os
import requests
from dotenv import load_dotenv
import logging
from pymongo import MongoClient
from datetime import datetime, timezone
import concurrent.futures
import time
import csv

# Setup logging
logging.basicConfig(filename='../logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API')
MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Setup MongoDB connection
client = MongoClient(MONGO_CONN_STRING)
db = client['stock_data']
selected_pairs_collection = db['selected_pairs']
earnings_calendar_collection = db['earnings_calendar']

# Clear the `earnings_calendar` collection at the start of the script
earnings_calendar_collection.delete_many({})

# Global variable to count API calls
api_calls_count = 0

def fetch_earnings_calendar(symbol, horizon='3month'):
    global api_calls_count

    if api_calls_count >= 300:
        logging.info("API call limit reached, waiting...")
        time.sleep(60)
        api_calls_count = 0

    CSV_URL = f'https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&symbol={symbol}&horizon={horizon}&apikey={ALPHA_VANTAGE_API_KEY}'
    with requests.Session() as s:
        download = s.get(CSV_URL)
        api_calls_count += 1

        if download.status_code == 200:
            decoded_content = download.content.decode('utf-8')
            cr = csv.reader(decoded_content.splitlines(), delimiter=',')
            earnings_calendar_list = list(cr)
            logging.info(f"Successfully fetched earnings calendar for {symbol}")
            return earnings_calendar_list
        else:
            logging.error(f"Failed to fetch earnings calendar for {symbol} with status code {download.status_code}")
            return None

def process_symbol(document):
    symbol = document.get('symbol')
    if symbol:
        earnings_calendar_data = fetch_earnings_calendar(symbol)
        if earnings_calendar_data:
            document_to_upsert = {
                'symbol': symbol,
                'earnings_calendar': earnings_calendar_data,
                'datetime_imported': datetime.now(timezone.utc)
            }
            earnings_calendar_collection.update_one(
                {'symbol': symbol},
                {'$set': document_to_upsert},
                upsert=True
            )

def main():
    symbols = list(selected_pairs_collection.find())

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_symbol, symbol) for symbol in symbols]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logging.error(f'Generated an exception: {exc}')

    logging.info("Earnings calendar update completed successfully at %s", datetime.now(timezone.utc))

if __name__ == "__main__":
    main()
