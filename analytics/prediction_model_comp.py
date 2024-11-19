import os
import logging
import requests
from datetime import datetime
from dotenv import load_dotenv
import pymongo
import pandas as pd

# Load environment variables
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API')
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("prediction_analysis_logfile.log"), logging.StreamHandler()])

# Initialize MongoDB client
mongo_client = pymongo.MongoClient(MONGO_DB_CONN_STRING)
db = mongo_client['predictions']


def fetch_actual_price(symbol, date):
    date = date.split('T')[0]  # Extract only the date part
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'Time Series (Daily)' in data:
            daily_data = data['Time Series (Daily)']
            # Find the most recent available date not later than the given date
            available_dates = sorted(daily_data.keys(), reverse=True)
            for available_date in available_dates:
                if available_date <= date:
                    return float(daily_data[available_date]['4. close'])
            logging.error(f"No data available on or before the date {date} in the fetched results.")
            return None
        else:
            logging.error(f"Error fetching data: {data}")
            return None
    else:
        logging.error(f"Error fetching data from Alpha Vantage: {response.text}")
        return None


def analyze_prediction(symbol):
    # Fetch the most recent prediction document based on the entry_date
    prediction_doc = db.September_2024.find_one(
        {"symbol": symbol},
        sort=[("entry_date", pymongo.DESCENDING)]
    )

    if prediction_doc:
        entry_date = prediction_doc['entry_date'].date().isoformat()  # Convert to YYYY-MM-DD format
        predicted_price = prediction_doc['predicted_price']

        actual_price = fetch_actual_price(symbol, entry_date)
        if actual_price is not None:
            delta = actual_price - predicted_price
            logging.info(f"Symbol: {symbol}")
            logging.info(f"Entry Date: {entry_date}")
            logging.info(f"Predicted Price: {predicted_price}")
            logging.info(f"Actual Price: {actual_price}")
            logging.info(f"Delta: {delta}")

            result = {
                "symbol": symbol,
                "entry_date": entry_date,
                "predicted_price": predicted_price,
                "actual_price": actual_price,
                "delta": delta
            }
            return result
        else:
            logging.error(f"Could not fetch actual price for symbol: {symbol} on date: {entry_date}")
            return None
    else:
        logging.error(f"No prediction document found for symbol: {symbol}")
        return None


def main():
    symbols = db.September_2024.distinct("symbol")
    results = []

    for symbol in symbols:
        result = analyze_prediction(symbol)
        if result:
            results.append(result)

    if results:
        # Create a DataFrame from the results
        df = pd.DataFrame(results)

        # Sort the DataFrame alphabetically by the symbol
        df = df.sort_values(by='symbol')

        # Convert the DataFrame to a list of dictionaries for MongoDB insertion
        records = df.to_dict(orient='records')

        # Insert the records into the 'predictions_comparisons' collection
        db.predictions_comparisons.insert_many(records)

        logging.info("Prediction analysis saved to MongoDB collection 'predictions_comparisons'")

        print(df)


if __name__ == "__main__":
    main()
