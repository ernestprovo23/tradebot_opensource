import os
import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
import logging

# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(asctime)s - %(levelname)s - %(message)s')

# Database configuration
MONGO_DB_CONN_STRING = os.getenv("MONGO_DB_CONN_STRING")

# MongoDB connection
client = MongoClient(MONGO_DB_CONN_STRING)
stock_data_db = client["stock_data"]
historic_data_db = client["historic_data"]

selected_pairs_collection = stock_data_db["selected_pairs"]
historic_data_collection = historic_data_db["stock_data_historic"]

def main():
    logging.info("Starting data migration script...")

    try:
        # Fetch all documents from the selected_pairs collection
        selected_pairs = list(selected_pairs_collection.find())

        if not selected_pairs:
            logging.info("No records found in 'selected_pairs' collection.")
            return

        current_date = datetime.datetime.now().strftime('%Y-%m-%d')

        for record in selected_pairs:
            # Check if a record with the same symbol and date_added already exists in the historic_data_collection
            existing_record = historic_data_collection.find_one({
                'symbol': record['symbol'],
                'date_added': record['date_added'],
                'date_imported_historic': {'$regex': f'^{current_date}'}
            })

            if not existing_record:
                record["date_imported_historic"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                historic_data_collection.insert_one(record)
                logging.info(f"Inserted new record for symbol: {record['symbol']} with date_added: {record['date_added']}")
            else:
                logging.info(f"Record for symbol: {record['symbol']} with date_added: {record['date_added']} already exists for today.")

        logging.info(f"Completed copying records to 'stock_data_historic' collection in 'historic_data' database.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
