import os
import sys
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pymongo
import requests
from functools import wraps

# Load environment variables
load_dotenv()
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("logfile.log"), logging.StreamHandler(sys.stdout)])

# Initialize MongoDB client
mongo_client = pymongo.MongoClient(MONGO_DB_CONN_STRING)
db = mongo_client['trading_db']


def retry(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    attempts += 1
                    if attempts == max_attempts:
                        logging.error(f"Max retry attempts reached. Error: {e}")
                        raise
                    logging.warning(f"Request failed. Retrying in {delay} seconds...")
                    time.sleep(delay)

        return wrapper

    return decorator


@retry(max_attempts=3, delay=1)
def get_account_info():
    url = "https://paper-api.alpaca.markets/v2/account"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def store_account_snapshot(account_info):
    account_info['timestamp'] = datetime.now()
    db.account_info.insert_one(account_info)
    logging.info("Account snapshot stored successfully")


def get_initial_balance():
    initial_snapshot = db.account_info.find_one(sort=[("timestamp", 1)])
    if initial_snapshot:
        return float(initial_snapshot.get("equity", 0))
    return 0


def store_daily_snapshot():
    account_info = get_account_info()

    # Check if we already have a snapshot for today
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    existing_snapshot = db.account_info.find_one({"timestamp": {"$gte": today}})

    if not existing_snapshot:
        store_account_snapshot(account_info)
        logging.info("Daily account snapshot stored")
    else:
        logging.info("Daily snapshot already exists, skipping")


def update_account_info():
    try:
        account_info = get_account_info()

        # Store the daily snapshot
        store_daily_snapshot()

        # Update the current account info
        current_info = {
            "timestamp": datetime.now(),
            "account_info": account_info,
            "initial_balance": get_initial_balance()
        }
        db.current_account_info.replace_one({}, current_info, upsert=True)

        logging.info("Account info updated successfully")
    except Exception as e:
        logging.error(f"Error updating account info: {e}")


def main():
    update_account_info()


if __name__ == "__main__":
    main()