import pandas as pd
from pymongo import MongoClient
import logging
from dotenv import load_dotenv
import os
import hashlib
from pymongo import InsertOne, UpdateOne
from datetime import datetime

# Configure the logger
logging.basicConfig(filename='clean_balancesheet.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


# Load environment variables
load_dotenv()
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

def connect_to_mongodb():
    try:
        client = MongoClient(MONGO_DB_CONN_STRING)
        db = client['cleaned_data']  # Update this to your actual database name
        return db
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        return None

def generate_record_hash(record):
    """Generate a hash for a record based on its content."""
    record_str = str(sorted(record.items()))  # Convert record to a string of sorted items
    return hashlib.sha256(record_str.encode()).hexdigest()  # Generate SHA-256 hash

def process_balance_sheet_data(db):
    # Retrieve data from MongoDB collection
    source_client = MongoClient(MONGO_DB_CONN_STRING)
    db_source = source_client['stock_data']
    collection = db_source['balance_sheet']  # Update this to your actual collection name
    data = list(collection.find({}))

    annual_dfs = []
    quarterly_dfs = []

    for doc in data:
        symbol = doc['symbol']
        balance_sheet = doc['balance_sheet']

        # Process annual reports
        if 'annualReports' in balance_sheet:
            annual_df = pd.DataFrame(balance_sheet['annualReports'])
            annual_df['symbol'] = symbol
            annual_dfs.append(annual_df)

        # Process quarterly reports
        if 'quarterlyReports' in balance_sheet:
            quarterly_df = pd.DataFrame(balance_sheet['quarterlyReports'])
            quarterly_df['symbol'] = symbol
            quarterly_dfs.append(quarterly_df)

    # Combine all dataframes
    annual_combined = pd.concat(annual_dfs, ignore_index=True) if annual_dfs else pd.DataFrame()
    quarterly_combined = pd.concat(quarterly_dfs, ignore_index=True) if quarterly_dfs else pd.DataFrame()

    # Clean and convert data types
    for df in [annual_combined, quarterly_combined]:
        if not df.empty:
            for column in df.columns:
                if column not in ['symbol', 'fiscalDateEnding', 'reportedCurrency']:
                    df[column] = pd.to_numeric(df[column].replace('None', pd.NA), errors='coerce')

            # Set multi-index
            df.set_index(['symbol', 'fiscalDateEnding'], inplace=True)
            df.sort_index(inplace=True)

    return annual_combined, quarterly_combined

def save_to_mongodb(db, collection_name, data_frame):
    """Append only new or updated rows of data to the specified MongoDB collection."""
    collection = db[collection_name]
    operations = []

    # Prepare a list of documents to insert
    for record in data_frame.reset_index().to_dict(orient='records'):
        # Generate hash for the current record
        record_hash = generate_record_hash(record)
        record['record_hash'] = record_hash
        record['date_loaded'] = datetime.utcnow()  # Add the current UTC datetime

        symbol = record['symbol']
        fiscal_date = record['fiscalDateEnding']

        # Check if the record already exists with the same symbol and fiscal date
        existing_record = collection.find_one({'symbol': symbol, 'fiscalDateEnding': fiscal_date})

        if existing_record:
            # Check if the hash is the same
            if existing_record.get('record_hash') == record_hash:
                logging.info(f"Record for {symbol} on {fiscal_date} is unchanged. Skipping.")
            else:
                # Update the existing record if the hash is different, and update date_loaded
                operations.append(UpdateOne(
                    {'symbol': symbol, 'fiscalDateEnding': fiscal_date},
                    {'$set': record}
                ))
                logging.info(f"Record for {symbol} on {fiscal_date} updated.")
        else:
            # Insert the new record if it doesn't exist
            operations.append(InsertOne(record))
            logging.info(f"New record for {symbol} on {fiscal_date} inserted.")

    # Perform the bulk write only if there are new records to insert or update
    if operations:
        collection.bulk_write(operations)
        logging.info(f"Data successfully saved to MongoDB collection '{collection_name}'")
    else:
        logging.info(f"No new records to insert or update in collection '{collection_name}'")

def main():
    db = connect_to_mongodb()
    if db is None:
        logging.error("Could not connect to MongoDB. Exiting.")
        return

    annual_df, quarterly_df = process_balance_sheet_data(db)

    print("Annual Reports DataFrame:")
    print(annual_df.head())
    print("\nQuarterly Reports DataFrame:")
    print(quarterly_df.head())

    # Save to MongoDB under 'cleaned_financials' database
    if not annual_df.empty:
        save_to_mongodb(db, 'bs_annual', annual_df)

    if not quarterly_df.empty:
        save_to_mongodb(db, 'bs_quarterly', quarterly_df)

if __name__ == "__main__":
    main()
