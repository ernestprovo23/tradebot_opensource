import os
import pymongo
from datetime import datetime

# MongoDB connection string from environment variables
MONGO_CONN_STRING = os.getenv('MONGO_CONN_STRING')

# Create a MongoDB client
client = pymongo.MongoClient(MONGO_CONN_STRING)

# Select your database
db = client['your_database_name']  # Replace 'your_database_name' with the name of your database

# Select the collection
trades_collection = db.trades


def record_trade(symbol, qty, price, date=None):
    """
    Record a trade in MongoDB.
    """
    # Set the default date as current datetime if not provided
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Check if trade already exists
    existing_trade = trades_collection.find_one({'symbol': symbol, 'qty': qty, 'price': price, 'date': date})
    if existing_trade:
        return  # Trade already exists, so don't record it

    # Create a new trade document
    new_trade = {
        'symbol': symbol,
        'qty': qty,
        'price': price,
        'date': date
    }

    # Insert the new trade document into the collection
    trades_collection.insert_one(new_trade)


def download_trades():
    """
    Retrieve the trades from MongoDB and return them as a list of dictionaries.
    """
    trades = list(trades_collection.find({}, {'_id': 0}))  # Exclude the _id field
    return trades

