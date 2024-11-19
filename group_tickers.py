import os
import logging
from pymongo import MongoClient
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("logfile.log"), logging.StreamHandler(sys.stdout)])

def group_and_create_collections(db, collection_name):
    # Get a list of unique exchanges
    exchanges = db[collection_name].distinct('exchange')

    # Group and insert tickers into separate collections based on the exchange
    for exchange in exchanges:
        # Clean exchange name to create a valid collection name
        clean_exchange_name = ''.join(e for e in exchange if e.isalnum())
        collection_name_exchange = f"tickers_{clean_exchange_name}"

        # Find all tickers for this exchange
        tickers = list(db[collection_name].find({'exchange': exchange}))

        if tickers:
            # Drop the collection if it already exists to avoid duplicates
            db[collection_name_exchange].drop()
            # Create a new collection and insert tickers
            db[collection_name_exchange].insert_many(tickers)
            logging.info(f"Created new collection '{collection_name_exchange}' with {len(tickers)} tickers from the exchange: {exchange}")
        else:
            logging.info(f"No tickers found for exchange: {exchange}")

def main():
    client = MongoClient(MONGO_DB_CONN_STRING)
    db = client['stock_data']

    group_and_create_collections(db, 'tickers')

if __name__ == "__main__":
    main()
