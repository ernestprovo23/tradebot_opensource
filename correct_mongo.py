from pymongo import MongoClient
import logging
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize logging
logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# MongoDB connection setup
MONGO_DB_CONN_STRING = os.getenv("MONGO_DB_CONN_STRING")
DB_NAME = "stock_data"
COLLECTION_NAME = "tickers"

client = MongoClient(MONGO_DB_CONN_STRING)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def delete_all_documents(collection):
    """
    Deletes all documents in the collection.
    """
    try:
        result = collection.delete_many({})  # An empty query object matches all documents
        logging.info(f"Deleted {result.deleted_count} documents from the collection.")
    except Exception as e:
        logging.error(f"Error deleting documents: {e}")

if __name__ == "__main__":
    start_time = datetime.now()
    delete_all_documents(collection)
    total_time = datetime.now() - start_time
    logging.info(f"Total deletion script completed. Total elapsed time: {total_time}")
