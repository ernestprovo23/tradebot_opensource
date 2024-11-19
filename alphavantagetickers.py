import requests
import pandas as pd
import logging
from dotenv import load_dotenv
import os
from pymongo import MongoClient, UpdateOne
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Create a logger
logging.basicConfig(filename='logfile.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# MongoDB connection setup
MONGO_DB_CONN_STRING = os.getenv("MONGO_DB_CONN_STRING")
DB_NAME = "stock_data"  # Specify your database name
COLLECTION_NAME = "tickers"  # Specify your collection name

def fetch_tickers(api_key):
    """
    Fetches ticker data from Alpha Vantage API.
    """
    url = f'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        logging.info("Connected to Alpha Vantage API successfully.")
        return response.text
    except requests.exceptions.RequestException as e:
        logging.error(f"Error connecting to Alpha Vantage API: {e}")
        return None

def parse_tickers(data):
    """
    Parses the ticker data into a DataFrame.
    """
    lines = data.strip().split("\n")
    header, *rows = lines
    data_list = [dict(zip(header.split(","), row.split(","))) for row in rows]
    return pd.DataFrame(data_list)


def save_to_mongodb(dataframe, connection_string, db_name, collection_name):
    """
    Saves the DataFrame to a MongoDB collection with deduplication using bulk operations.
    Also adds a datetime_imported field to each document.
    """
    client = MongoClient(connection_string)
    db = client[db_name]
    collection = db[collection_name]

    # Prepare bulk update operations
    operations = []
    datetime_imported = datetime.now()  # Capture the current datetime for the import

    for record in dataframe.to_dict('records'):
        # Add/update the datetime_imported field in the record
        record['datetime_imported'] = datetime_imported
        # Assume 'symbol' is the unique identifier for each ticker
        filter_ = {'symbol': record['symbol']}
        update_ = {'$set': record}
        operations.append(UpdateOne(filter_, update_, upsert=True))

    try:
        # Execute all the operations in bulk
        if operations:
            result = collection.bulk_write(operations)
            logging.info(f"Bulk write completed with {result.modified_count} modifications.")
        else:
            logging.info("No operations to perform.")

    except Exception as e:
        logging.error(f"Error saving data to MongoDB with bulk write: {e}")

def update_tickers():
    """
    Main function to fetch ticker data from Alpha Vantage and save it to MongoDB.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API")
    data = fetch_tickers(api_key)
    if data:
        df = parse_tickers(data)
        save_to_mongodb(df, MONGO_DB_CONN_STRING, DB_NAME, COLLECTION_NAME)

if __name__ == "__main__":
    update_tickers()
