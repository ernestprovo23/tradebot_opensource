import requests
import os
from dotenv import load_dotenv
import logging
from pymongo import MongoClient
from datetime import datetime, timedelta
import json

# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(filename='portfolio_history.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Alpaca API configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# MongoDB configuration
MONGO_DB_CONN_STRING = os.getenv("MONGO_DB_CONN_STRING")
DB_NAME = "stock_data"

# Ensure the API keys are loaded correctly
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    logging.error("Failed to load ALPACA_API_KEY or ALPACA_SECRET_KEY from environment variables.")
    exit(1)

# MongoDB connection
mongo_client = MongoClient(MONGO_DB_CONN_STRING)
db = mongo_client[DB_NAME]

def fetch_portfolio_history():
    url = "https://paper-api.alpaca.markets/v2/account/portfolio/history"
    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY
    }
    params = {
        'period': '1A',
        'intraday_reporting': 'continuous',
        'pnl_reset': 'per_day'
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred while fetching portfolio history: {e}")
        return None


def insert_portfolio_history(db, portfolio_history):
    try:
        collection = db.portfolio_history
        six_hours_ago = datetime.now() - timedelta(hours=6)

        # Check if a document has been inserted in the last 6 hours
        existing_document = collection.find_one({
            "datetime": {"$gte": six_hours_ago.isoformat()}
        })

        if existing_document:
            logging.info("A portfolio history document has been inserted in the last 6 hours. Skipping insertion.")
        else:
            portfolio_history['datetime'] = datetime.now().isoformat()
            collection.insert_one(portfolio_history)
            logging.info("Successfully inserted portfolio history into MongoDB.")
    except Exception as e:
        logging.error(f"An error occurred while inserting portfolio history into MongoDB: {e}")


def main():
    portfolio_history = fetch_portfolio_history()
    if portfolio_history:
        insert_portfolio_history(db, portfolio_history)
        # Print the portfolio history as JSON for debugging purposes
        print(json.dumps(portfolio_history, indent=2, default=str))
    else:
        logging.info("No portfolio history data retrieved.")


if __name__ == "__main__":
    main()
