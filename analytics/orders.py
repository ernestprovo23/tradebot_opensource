import os
import sys
import requests
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pymongo

# Load environment variables
load_dotenv()
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("orders_logfile.log"), logging.StreamHandler(sys.stdout)])

# Initialize MongoDB client
mongo_client = pymongo.MongoClient(MONGO_DB_CONN_STRING)
db = mongo_client['trading_db']
orders_collection = db['orders']

def get_orders(start_date, end_date, page_token=None):
    url = "https://paper-api.alpaca.markets/v2/orders"
    params = {
        "status": "all",
        "limit": 500,
        "after": start_date,
        "until": end_date,
        "direction": "asc"
    }
    if page_token:
        params['page_token'] = page_token

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
    }

    logging.info(f"Fetching orders with params: {params}")

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        logging.info("Fetched orders successfully")
        return response.json()
    else:
        logging.error(f"Error fetching orders: {response.status_code} - {response.text}")
        return None

def main():
    all_orders = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30*3)  # Start date 3 months ago

    # Clear the orders collection before inserting new data
    logging.info("Clearing existing orders collection")
    orders_collection.delete_many({})

    while start_date < end_date:
        current_end_date = start_date + timedelta(days=30)
        page_token = None
        last_order_date = None

        while True:
            start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            current_end_date_str = current_end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

            logging.info(f"Fetching orders from {start_date_str} to {current_end_date_str}...")
            orders_data = get_orders(start_date_str, current_end_date_str, page_token)
            if not orders_data:
                break

            all_orders.extend(orders_data)
            logging.info(f"Fetched {len(orders_data)} orders")

            if len(orders_data) > 0:
                new_last_order_date = orders_data[-1]["created_at"]
                if new_last_order_date == last_order_date:
                    logging.info("No new orders fetched, breaking the loop.")
                    break

                last_order_date = new_last_order_date
                logging.info(f"Last order date on this page: {last_order_date}")

                page_token = orders_data[-1].get("id")  # Use the last order's id as the page token for the next request
                logging.info(f"Setting page_token for next fetch: {page_token}")
                if len(orders_data) < 500:
                    break  # No more orders to fetch for this range
            else:
                break  # Exit if no more orders in the current page

        start_date = current_end_date

    logging.info(f"Total orders fetched: {len(all_orders)}")
    if all_orders:
        oldest_date = min(order["created_at"] for order in all_orders)
        logging.info(f"Oldest order date: {oldest_date}")

        # Insert fetched orders into MongoDB
        logging.info("Inserting fetched orders into MongoDB")
        orders_collection.insert_many(all_orders)
        logging.info("Orders inserted successfully")

if __name__ == "__main__":
    main()
