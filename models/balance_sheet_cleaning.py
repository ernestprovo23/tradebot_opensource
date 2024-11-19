import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import numpy as np


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Setup MongoDB connection
client = MongoClient(MONGO_CONN_STRING)
db = client['stock_data']  # Accessing the 'stock_data' database

# Accessing the 'balance_sheet' collection
balance_sheet_collection = db['balance_sheet']


def clean_and_prepare_balance_sheet_data():
    try:
        # Fetch all documents from the collection
        documents = balance_sheet_collection.find()

        cleaned_data = []

        for doc in documents:
            symbol = doc.get('symbol')
            annual_reports = doc.get('balance_sheet', {}).get('annualReports', [])

            for report in annual_reports:
                # Convert string values to numeric, handling 'None' as NaN
                for key, value in report.items():
                    if value == 'None':
                        report[key] = None
                    else:
                        try:
                            report[key] = float(value)
                        except ValueError:
                            pass  # Keep the original value if it cannot be converted

                # Calculate financial ratios
                try:
                    current_ratio = report['totalCurrentAssets'] / report['totalCurrentLiabilities']
                except (TypeError, ZeroDivisionError):
                    current_ratio = None

                try:
                    quick_ratio = (report['cashAndCashEquivalentsAtCarryingValue'] + report['currentNetReceivables']) / \
                                  report['totalCurrentLiabilities']
                except (TypeError, ZeroDivisionError):
                    quick_ratio = None

                try:
                    debt_to_equity_ratio = report['totalLiabilities'] / report['totalShareholderEquity']
                except (TypeError, ZeroDivisionError):
                    debt_to_equity_ratio = None

                try:
                    long_term_debt_to_equity_ratio = report['longTermDebt'] / report['totalShareholderEquity']
                except (TypeError, ZeroDivisionError):
                    long_term_debt_to_equity_ratio = None

                # Append cleaned and calculated data
                cleaned_data.append({
                    'symbol': symbol,
                    'fiscalDateEnding': report['fiscalDateEnding'],
                    'current_ratio': current_ratio,
                    'quick_ratio': quick_ratio,
                    'debt_to_equity_ratio': debt_to_equity_ratio,
                    'long_term_debt_to_equity_ratio': long_term_debt_to_equity_ratio,
                    # Add more features as needed
                })

        # Convert to DataFrame for further processing or analysis
        cleaned_df = pd.DataFrame(cleaned_data)
        logging.info("Cleaned and prepared balance sheet data:")
        logging.info(cleaned_df.head())

        # Handle missing values by replacing NaN with zero
        cleaned_df.fillna(0, inplace=True)

        # Store the cleaned data into a new collection 'balance_sheet_ML'
        balance_sheet_ml_collection = db['balance_sheet_ML']

        # Convert DataFrame to dictionary records and insert into MongoDB
        records = cleaned_df.to_dict('records')
        new_records = []

        for record in records:
            # Check if the record already exists in the collection
            existing_record = balance_sheet_ml_collection.find_one({
                'symbol': record['symbol'],
                'fiscalDateEnding': record['fiscalDateEnding']
            })

            if not existing_record:
                # Add datetime_imported field
                record['datetime_imported'] = datetime.utcnow()
                new_records.append(record)

        if new_records:
            balance_sheet_ml_collection.insert_many(new_records)
            logging.info(f"Inserted {len(new_records)} new records into 'balance_sheet_ML' collection.")
        else:
            logging.info("No new records to insert into 'balance_sheet_ML' collection.")

    except Exception as e:
        logging.error(f"An error occurred while cleaning balance sheet data: {e}")
        return None


if __name__ == "__main__":
    logging.info("Starting balance sheet data cleaning and preparation...")
    clean_and_prepare_balance_sheet_data()