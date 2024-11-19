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

# Accessing the 'treasury_yields' collection
treasury_yields_collection = db['treasury_yields']

def fetch_and_store_treasury_yields(maturity):
    try:
        # Fetch treasury yield data from API
        url = f'https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=monthly&maturity={maturity}&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        data = response.json()

        if 'data' in data:
            records = data['data']
            new_records = []

            for record in records:
                # Check if the record already exists in the collection
                existing_record = treasury_yields_collection.find_one({
                    'date': record['date'],
                    'maturity': maturity
                })

                if not existing_record:
                    # Add datetime_imported and maturity fields
                    record['datetime_imported'] = datetime.utcnow()
                    record['maturity'] = maturity
                    new_records.append(record)

            if new_records:
                treasury_yields_collection.insert_many(new_records)
                logging.info(f"Inserted {len(new_records)} new records for {maturity} into 'treasury_yields' collection.")
            else:
                logging.info(f"No new records to insert for {maturity} into 'treasury_yields' collection.")
        else:
            logging.error(f"No data found in API response for {maturity}.")

    except Exception as e:
        logging.error(f"An error occurred while fetching and storing treasury yields for {maturity}: {e}")

if __name__ == "__main__":
    logging.info("Starting treasury yield data fetching and storage...")
    maturities = ['5year', '10year', '30year']
    for maturity in maturities:
        fetch_and_store_treasury_yields(maturity)