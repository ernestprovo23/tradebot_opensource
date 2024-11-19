import os
import pandas as pd
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize Alpaca API
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)

# Setup MongoDB connection
client = MongoClient(MONGO_CONN_STRING)
db = client.stock_data
collection = db.raw_trades_import

def get_all_trades(api):
    try:
        account_activities = api.get_activities(activity_types='FILL')
        trades_data = []
        for activity in account_activities:
            trades_data.append({
                'id': activity.id,
                'symbol': activity.symbol,
                'qty': activity.qty,
                'side': activity.side,
                'price': activity.price,
                'transaction_time': activity.transaction_time,
                'order_id': activity.order_id,
                'type': activity.type
            })
        trades_df = pd.DataFrame(trades_data)
        return trades_df
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()

def store_trades_to_mongo(trades_df):
    if not trades_df.empty:
        records = trades_df.to_dict('records')
        collection.insert_many(records)
        print("Trades stored successfully in MongoDB.")
    else:
        print("No trades to store.")

def main():
    trades_df = get_all_trades(api)
    store_trades_to_mongo(trades_df)

if __name__ == "__main__":
    main()
