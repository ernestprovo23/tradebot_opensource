import os
from pymongo import MongoClient
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# MongoDB setup
client = MongoClient(MONGO_DB_CONN_STRING)
stock_data_db = client.stock_data
options_contracts_collection = stock_data_db.options_contracts

def extract_symbol(symbol_string):
    """Extract the unique symbol from the option symbol string."""
    match = re.match(r"([A-Z]+)", symbol_string)
    if match:
        return match.group(1)
    return None

def get_unique_symbols():
    """Get a list of unique symbols from the options contracts collection."""
    unique_symbols = set()
    for doc in options_contracts_collection.find({}, {'symbol': 1}):
        symbol_string = doc.get('symbol', '')
        unique_symbol = extract_symbol(symbol_string)
        if unique_symbol:
            unique_symbols.add(unique_symbol)
    return list(unique_symbols)

def analyze_options_data():
    """Analyze the options contracts data."""
    unique_symbols = get_unique_symbols()
    print(f"Unique symbols found: {unique_symbols}")

    # Example analysis of other fields
    sample_data = options_contracts_collection.find_one()
    if sample_data:
        fields = sample_data.keys()
        print(f"Sample document fields: {fields}")

        # Additional analysis: Count of contracts per symbol
        contract_counts = {}
        for symbol in unique_symbols:
            count = options_contracts_collection.count_documents({'symbol': {'$regex': f'^{symbol}'}})
            contract_counts[symbol] = count
        print(f"Contract counts per symbol: {contract_counts}")

def main():
    analyze_options_data()

if __name__ == "__main__":
    main()
