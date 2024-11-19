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

# Accessing the 'retail_sales' collection
retail_sales_collection = db['retail_sales']

def fetch_and_store_retail_sales_data():
    try:
        # Fetch retail sales data from API
        url = f'https://www.alphavantage.co/query?function=RETAIL_SALES&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        data = response.json()

        if 'data' in data:
            records = data['data']
            new_records = []

            for record in records:
                # Check if the record already exists in the collection
                existing_record = retail_sales_collection.find_one({
                    'date': record['date']
                })

                if not existing_record:
                    # Add datetime_imported field
                    record['datetime_imported'] = datetime.utcnow()
                    new_records.append(record)

            if new_records:
                retail_sales_collection.insert_many(new_records)
                logging.info(f"Inserted {len(new_records)} new records into 'retail_sales' collection.")
            else:
                logging.info("No new records to insert into 'retail_sales' collection.")
        else:
            logging.error("No data found in API response.")

    except Exception as e:
        logging.error(f"An error occurred while fetching and storing retail sales data: {e}")

if __name__ == "__main__":
    logging.info("Starting retail sales data fetching and storage...")
    fetch_and_store_retail_sales_data()