import os
import sys
import json
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pymongo
import requests
import pandas as pd

# Load environment variables
load_dotenv()
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("logfile.log"), logging.StreamHandler(sys.stdout)])

# Initialize MongoDB client
mongo_client = pymongo.MongoClient(MONGO_DB_CONN_STRING)
db = mongo_client['trading_db']

def fetch_open_positions():
    url = "https://paper-api.alpaca.markets/v2/positions"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Error fetching open positions: {response.text}")
        return []

def fetch_current_price(symbol):
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return float(data['quote']['ap'])
    else:
        logging.error(f"Error fetching current price for {symbol}: {response.text}")
        return None

def calculate_pnl_by_symbol(start_date, end_date):
    pipeline = [
        {"$match": {
            "status": "filled",
            "filled_at": {"$gte": start_date.isoformat(), "$lte": end_date.isoformat()}
        }},
        {"$addFields": {
            "numeric_filled_qty": {"$toDouble": "$filled_qty"},
            "numeric_filled_avg_price": {"$toDouble": "$filled_avg_price"}
        }},
        {"$group": {
            "_id": "$symbol",
            "total_buy": {
                "$sum": {
                    "$cond": [
                        {"$eq": ["$side", "buy"]},
                        {"$multiply": ["$numeric_filled_qty", "$numeric_filled_avg_price"]},
                        0
                    ]
                }
            },
            "total_sell": {
                "$sum": {
                    "$cond": [
                        {"$eq": ["$side", "sell"]},
                        {"$multiply": ["$numeric_filled_qty", "$numeric_filled_avg_price"]},
                        0
                    ]
                }
            },
            "net_qty": {
                "$sum": {
                    "$cond": [
                        {"$eq": ["$side", "buy"]},
                        "$numeric_filled_qty",
                        {"$multiply": ["$numeric_filled_qty", -1]}
                    ]
                }
            }
        }}
    ]

    results = list(db.orders.aggregate(pipeline))

    open_positions = fetch_open_positions()
    symbol_pnl = []

    for result in results:
        symbol = result['_id']
        realized_pnl = result['total_sell'] - result['total_buy']
        unrealized_pnl = 0

        # Check if there's an open position for this symbol
        open_position = next((p for p in open_positions if p['symbol'] == symbol), None)
        if open_position and result['net_qty'] != 0:
            current_price = float(open_position['current_price'])
            if current_price is None:
                current_price = fetch_current_price(symbol)
            unrealized_pnl = result['net_qty'] * (
                    current_price - (result['total_buy'] - result['total_sell']) / result['net_qty'])

        total_pnl = realized_pnl + unrealized_pnl
        symbol_pnl.append({
            "symbol": symbol,
            "realized_pnl": realized_pnl if realized_pnl >= 0 else realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl
        })

    return pd.DataFrame(symbol_pnl)

def main():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # Last 3 months

    pnl_df = calculate_pnl_by_symbol(start_date, end_date)
    pnl_df.to_csv('pnl_by_symbol.csv', index=False)
    print(pnl_df)
    logging.info("PnL by symbol:\n" + pnl_df.to_string())

if __name__ == "__main__":
    main()
