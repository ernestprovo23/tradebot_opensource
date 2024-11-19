import sys
import asyncio
import aiohttp
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
from pymongo import MongoClient

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_test_log.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPHA_VANTAGE_API')
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Connect to MongoDB
client = MongoClient(MONGO_DB_CONN_STRING)
db = client['stock_data']
collection = db['company_overviews']

# Constants
API_URL = 'https://www.alphavantage.co/query'
FUNCTION = 'OVERVIEW'
TEST_DURATION_SECONDS = 60  # Test for 60 seconds
CONCURRENT_REQUESTS = 299    # Number of concurrent requests

async def fetch_overview(session, symbol):
    """Fetch the company overview for a given symbol."""
    params = {
        'function': FUNCTION,
        'symbol': symbol,
        'apikey': API_KEY
    }
    try:
        async with session.get(API_URL, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if 'Note' in data:
                    logging.warning(f"API Note received for {symbol}: {data['Note']}")
                    return 'rate_limited'
                elif 'Sector' in data:
                    logging.info(f"Successfully fetched data for {symbol}")
                    return 'success'
                else:
                    logging.info(f"No data for {symbol}")
                    return 'no_data'
            else:
                logging.error(f"HTTP Error {response.status} for symbol {symbol}")
                return 'http_error'
    except aiohttp.ClientError as e:
        logging.error(f"Client error for symbol {symbol}: {e}")
        return 'client_error'

async def test_api_rate_limit(symbols):
    """Test how many API calls can be made within the test duration."""
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=TEST_DURATION_SECONDS)
    success_count = 0
    rate_limited_count = 0
    http_error_count = 0
    client_error_count = 0
    no_data_count = 0

    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    timeout = aiohttp.ClientTimeout(total=None)  # No total timeout

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for symbol in symbols:
            if datetime.now() >= end_time:
                break
            task = asyncio.create_task(fetch_overview(session, symbol))
            tasks.append(task)

        for task in asyncio.as_completed(tasks):
            if datetime.now() >= end_time:
                break
            result = await task
            if result == 'success':
                success_count += 1
            elif result == 'rate_limited':
                rate_limited_count += 1
            elif result == 'http_error':
                http_error_count += 1
            elif result == 'client_error':
                client_error_count += 1
            elif result == 'no_data':
                no_data_count += 1

    logging.info("API Rate Limit Test Completed")
    logging.info(f"Test Duration: {TEST_DURATION_SECONDS} seconds")
    logging.info(f"Successful Calls: {success_count}")
    logging.info(f"Rate Limited Calls: {rate_limited_count}")
    logging.info(f"HTTP Errors: {http_error_count}")
    logging.info(f"Client Errors: {client_error_count}")
    logging.info(f"No Data Returned: {no_data_count}")

def main():
    # Fetch unique symbols from MongoDB
    symbols = collection.distinct('Symbol')

    # Run the test
    asyncio.run(test_api_rate_limit(symbols))

if __name__ == "__main__":
    main()
