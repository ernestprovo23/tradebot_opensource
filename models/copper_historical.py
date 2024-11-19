import requests
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API')
MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Setup MongoDB connection
client = MongoClient(MONGO_CONN_STRING)
db = client['commodity_data']  # Accessing the 'commodity_data' database

# Accessing the 'copper_prices' collection
copper_prices_collection = db['copper_prices']

def fetch_and_store_copper_prices():
    try:
        # Fetch copper price data from API
        url = f'https://www.alphavantage.co/query?function=COPPER&interval=monthly&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        data = response.json()

        if 'data' in data:
            records = data['data']
            new_records = []

            for record in records:
                # Check if the record already exists in the collection
                existing_record = copper_prices_collection.find_one({
                    'date': record['date']
                })

                if not existing_record:
                    # Add datetime_imported field
                    record['datetime_imported'] = datetime.utcnow()
                    new_records.append(record)

            if new_records:
                copper_prices_collection.insert_many(new_records)
                logging.info(f"Inserted {len(new_records)} new records into 'copper_prices' collection.")
            else:
                logging.info("No new records to insert into 'copper_prices' collection.")
        else:
            logging.error("No data found in API response.")

    except Exception as e:
        logging.error(f"An error occurred while fetching and storing copper prices: {e}")

if __name__ == "__main__":
    logging.info("Starting copper price data fetching and storage...")
    fetch_and_store_copper_prices()