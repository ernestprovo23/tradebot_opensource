import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Setup MongoDB connection
client = MongoClient(MONGO_CONN_STRING)
db = client['stock_data']  # Accessing the 'stock_data' database

# Accessing the 'cash_flow' collection
cash_flow_collection = db['cash_flow']


def clean_and_prepare_cash_flow_data():
    try:
        # Fetch all documents from the collection
        documents = cash_flow_collection.find()

        cleaned_data = []

        for doc in documents:
            symbol = doc.get('symbol')
            annual_reports = doc.get('cash_flow', {}).get('annualReports', [])

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

                # Calculate additional features if needed
                # Example: Free Cash Flow = Operating Cash Flow - Capital Expenditures
                try:
                    free_cash_flow = report['operatingCashflow'] - report['capitalExpenditures']
                except (TypeError, KeyError):
                    free_cash_flow = None

                # Append cleaned and calculated data
                cleaned_data.append({
                    'symbol': symbol,
                    'fiscalDateEnding': report['fiscalDateEnding'],
                    'operatingCashflow': report.get('operatingCashflow', 0),
                    'capitalExpenditures': report.get('capitalExpenditures', 0),
                    'free_cash_flow': free_cash_flow if free_cash_flow is not None else 0,
                    # Add more features as needed
                })

        # Convert to DataFrame for further processing or analysis
        cleaned_df = pd.DataFrame(cleaned_data)
        logging.info("Cleaned and prepared cash flow data:")
        logging.info(cleaned_df.head())

        # Handle missing values by replacing NaN with zero
        cleaned_df.fillna(0, inplace=True)

        # Store the cleaned data into a new collection 'cash_flow_ML'
        cash_flow_ml_collection = db['cash_flow_ML']

        # Convert DataFrame to dictionary records and insert into MongoDB
        records = cleaned_df.to_dict('records')
        new_records = []

        for record in records:
            # Check if the record already exists in the collection
            existing_record = cash_flow_ml_collection.find_one({
                'symbol': record['symbol'],
                'fiscalDateEnding': record['fiscalDateEnding']
            })

            if not existing_record:
                # Add datetime_imported field
                record['datetime_imported'] = datetime.utcnow()
                new_records.append(record)

        if new_records:
            cash_flow_ml_collection.insert_many(new_records)
            logging.info(f"Inserted {len(new_records)} new records into 'cash_flow_ML' collection.")
        else:
            logging.info("No new records to insert into 'cash_flow_ML' collection.")

        return cleaned_df

    except Exception as e:
        logging.error(f"An error occurred while cleaning cash flow data: {e}")
        return None


if __name__ == "__main__":
    logging.info("Starting cash flow data cleaning and preparation...")
    clean_and_prepare_cash_flow_data()