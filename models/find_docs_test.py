from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Connect to MongoDB
client = MongoClient(MONGO_DB_CONN_STRING)
db = client['silvertables']  # Assuming all relevant collections are in the 'silvertables' database

# List of collections to check
collections = [
    'ML_balancesheets',
    'ML_cashflows',
    'ML_incomestatements',
    'ML_sentiment_data',
    'ML_features'  # Including this if you are using it for storing results
]

def print_collection_schema(collection_name):
    collection = db[collection_name]
    document = collection.find_one()
    if document:
        print(f"Schema for collection '{collection_name}':")
        for key in document.keys():
            print(f"  - {key}")
    else:
        print(f"No documents found in collection '{collection_name}'. Schema cannot be determined.")

for coll in collections:
    print_collection_schema(coll)
