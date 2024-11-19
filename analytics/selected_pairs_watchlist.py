import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Initialize MongoDB client
mongo_client = MongoClient(MONGO_DB_CONN_STRING)
db = mongo_client['stock_data']
selected_pairs_collection = db['selected_pairs']

def get_symbols_for_watchlist():
    # Fetch symbols from the 'selected_pairs' collection
    symbols = selected_pairs_collection.find({}, {'symbol': 1, '_id': 0})

    # Extract and return symbols in a comma-separated format for TradingView
    symbols_list = [doc['symbol'] for doc in symbols]
    return ', '.join(symbols_list)

# Get symbols and print them
symbols_for_watchlist = get_symbols_for_watchlist()
print(symbols_for_watchlist)