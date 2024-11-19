import requests
from dotenv import load_dotenv
import os
import pprint
import logging

# Load environment variables from .env file
load_dotenv()

def get_latest_trade_price(symbol):
    """Fetch the latest trade price for a given symbol using the Alpha Vantage API."""
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API")
    base_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"

    try:
        response = requests.get(base_url)
        response.raise_for_status()
        data = response.json()
        pprint.pp(data)

        # Ensure the expected key is present in the response
        if "Time Series (Daily)" in data:
            # Get the most recent trading day (i.e., the latest date)
            latest_date = max(data["Time Series (Daily)"].keys())
            latest_data = data["Time Series (Daily)"][latest_date]

            # Extract the closing price from the latest data
            price = float(latest_data['4. close'])
            logging.debug(f"Latest trade price for {symbol} on {latest_date}: {price}")
            print(f"Latest trade price for {symbol} on {latest_date}: {price}")
            return price
        else:
            logging.info(f"No trade data found for {symbol} via Alpha Vantage. Skipping.")
            print(f"No trade data found for {symbol} via Alpha Vantage. Skipping.")
            return None
    except Exception as e:
        logging.error(f"Error fetching latest trade price for {symbol}: {e}")
        print(f"Error fetching latest trade price for {symbol}: {e}")
        return None

symbol = 'TM'
get_latest_trade_price(symbol)