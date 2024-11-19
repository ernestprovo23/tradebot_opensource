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
db = client['economic_data']  # Accessing the 'economic_data' database

# Accessing the 'real_gdp' collection
real_gdp_collection = db['real_gdp']

def fetch_and_store_real_gdp():
    try:
        # Fetch real GDP data from API
        url = f'https://www.alphavantage.co/query?function=REAL_GDP&interval=annual&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        data = response.json()

        if 'data' in data:
            records = data['data']
            new_records = []

            for record in records:
                # Check if the record already exists in the collection
                existing_record = real_gdp_collection.find_one({
                    'date': record['date']
                })

                if not existing_record:
                    # Add datetime_imported field
                    record['datetime_imported'] = datetime.utcnow()
                    new_records.append(record)

            if new_records:
                real_gdp_collection.insert_many(new_records)
                logging.info(f"Inserted {len(new_records)} new records into 'real_gdp' collection.")
            else:
                logging.info("No new records to insert into 'real_gdp' collection.")
        else:
            logging.error("No data found in API response.")

    except Exception as e:
        logging.error(f"An error occurred while fetching and storing real GDP data: {e}")

if __name__ == "__main__":
    logging.info("Starting real GDP data fetching and storage...")
    fetch_and_store_real_gdp()