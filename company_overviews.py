import sys
import time
import os
import asyncio
import aiohttp
import concurrent.futures
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
from sector_filters import get_sector_thresholds, get_market_condition_adjustments
from mongo_func import MongoManager  # mongo_func.py class module for interacting with mongo

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("logfile.log"), logging.StreamHandler(sys.stdout)])

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPHA_VANTAGE_API')
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Initialize the MongoManager
mongo_manager = MongoManager(MONGO_DB_CONN_STRING, 'stock_data')

# Access collections
db_trade = mongo_manager.client["trading_db"]
bad_performers_collection = db_trade["bad_performers"]

# Load sector thresholds and market condition adjustments
sector_thresholds = get_sector_thresholds()
market_condition_adjustments = get_market_condition_adjustments()

def adjust_filters_for_market_conditions(filters):
    adjusted_filters = filters.copy()
    for key, adjustment in market_condition_adjustments.items():
        if key in adjusted_filters:
            if isinstance(adjusted_filters[key], list) and len(adjusted_filters[key]) == 2:
                adjusted_filters[key] = [adjusted_filters[key][0] + adjustment, adjusted_filters[key][1] + adjustment]
            else:
                adjusted_filters[key] += adjustment
    return adjusted_filters

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

async def fetch_company_overview(session, api_key, symbol):
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}'
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            data = await response.json()

            if 'Sector' not in data or data['Sector'] in [None, 'None', '', ' ', 'N/A']:
                logging.info(f"Sector information is missing for {symbol}. Skipping.")
                return None

            sector = data['Sector']
            filters = sector_thresholds.get(sector, sector_thresholds['default'])
            filters = adjust_filters_for_market_conditions(filters)

            if not all([
                safe_float(data.get('MarketCapitalization')) >= filters.get('MarketCapitalization', float('-inf')),
                safe_float(data.get('EBITDA')) >= filters.get('EBITDA', float('-inf')),
                filters.get('PERatio', [float('-inf'), float('inf')])[0] <= safe_float(data.get('PERatio'), float('inf')) <= filters.get('PERatio', [float('-inf'), float('inf')])[1],
                safe_float(data.get('EPS')) >= filters.get('EPS', float('-inf')),
                safe_float(data.get('Beta')) <= filters.get('Beta', float('inf')),
                safe_float(data.get('ReturnOnAssetsTTM')) >= filters.get('ROA', float('-inf')),
                safe_float(data.get('DebtToEquity')) <= filters.get('DebtToEquity', float('inf')),
                safe_float(data.get('FreeCashFlowTTM')) >= filters.get('FreeCashFlow', float('-inf')),
                safe_float(data.get('PriceToBookRatio')) <= filters.get('PriceToBookRatio', float('inf')),
                safe_float(data.get('DividendYield')) >= filters.get('DividendYield', float('-inf'))
            ]):
                logging.error(f"Symbol {symbol} does not meet filter criteria. Skipping.")
                return None

            # Add historical performance checks
            if not check_historical_performance(data):
                logging.error(f"Symbol {symbol} does not meet historical performance criteria. Skipping.")
                return None

            data['date_added'] = datetime.now()
            return data

    except aiohttp.ClientError as e:
        logging.error(f"Error retrieving company overview for symbol '{symbol}': {e}")
        return None

def check_historical_performance(data):
    try:
        # Assuming we have historical data fields in the data
        revenue_growth = safe_float(data.get('QuarterlyRevenueGrowthYOY'))
        earnings_growth = safe_float(data.get('QuarterlyEarningsGrowthYOY'))
        price_growth = safe_float(data.get('PriceGrowthLastYear'))  # Hypothetical field

        # Adjusted to be less stringent
        return all([
            revenue_growth > -1e9,  # Allowing for negative revenue growth but not excessively negative
            earnings_growth > -2e9,  # Allowing for negative earnings growth but not excessively negative
            price_growth > -0.1  # Allowing for slight negative price growth
        ])
    except Exception as e:
        logging.error(f"Error checking historical performance for {data.get('Symbol', 'unknown')}: {e}")
        return False

async def fetch_and_store_company_overviews(api_key, tickers_list, mongo_manager, collection_name):
    RATE_LIMIT_CALLS = 285
    company_overviews_list = []
    total_tickers = len(tickers_list)
    start_time = datetime.now()
    request_timestamps = []

    async with aiohttp.ClientSession() as session:
        tasks = []
        for ticker in tickers_list:
            tasks.append(fetch_company_overview(session, api_key, ticker))

            if len(request_timestamps) >= RATE_LIMIT_CALLS:
                time_since_limit_last_request = datetime.now() - request_timestamps[-RATE_LIMIT_CALLS]
                if time_since_limit_last_request.total_seconds() < 60:
                    sleep_time = 60 - time_since_limit_last_request.total_seconds()
                    logging.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds.")
                    await asyncio.sleep(sleep_time)

            request_timestamps.append(datetime.now())
            request_timestamps = [ts for ts in request_timestamps if datetime.now() - ts <= timedelta(minutes=1)]

        for future in asyncio.as_completed(tasks):
            data = await future
            if data:
                company_overviews_list.append(data)

        for data in company_overviews_list:
            mongo_manager.db[collection_name].delete_many({'Symbol': data['Symbol']})
            result = mongo_manager.db[collection_name].insert_one(data)
            logging.info(f"Inserted document ID: {result.inserted_id}")
            print(f"Inserted document ID: {result.inserted_id}")

    total_time = (datetime.now() - start_time).total_seconds()
    logging.info(f"Process completed. Total elapsed time: {total_time:.2f} seconds")

def filter_bad_performers(symbols):
    try:
        bad_performers = list(bad_performers_collection.find())
        bad_symbols = []
        for bp in bad_performers:
            symbol = bp["symbol"]
            avoid_until = bp["avoid_until"]
            if datetime.now() < avoid_until:
                bad_symbols.append(symbol)
        logging.info(f"Filtering out {len(bad_symbols)} bad performers.")
        return [symbol for symbol in symbols if symbol not in bad_symbols]
    except Exception as e:
        logging.error(f"An error occurred while filtering bad performers: {e}")
        return symbols

def main():

    start_time = datetime.now()
    collection_name = 'company_overviews'

    # Clear the collection at the start of each run
    mongo_manager.db[collection_name].delete_many({})

    tickers_collection = mongo_manager.db['tickers']
    tickers_list = [ticker['symbol'] for ticker in tickers_collection.find({}, {'symbol': 1, '_id': 0})]

    # Filter out bad performers
    tickers_list = filter_bad_performers(tickers_list)

    # Optional: Limit the number of tickers for testing
    # tickers_list = tickers_list[:200]

    asyncio.run(fetch_and_store_company_overviews(API_KEY, tickers_list, mongo_manager, collection_name))

    total_time = (datetime.now() - start_time).total_seconds()
    logging.info(f"Process completed. Total elapsed time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
