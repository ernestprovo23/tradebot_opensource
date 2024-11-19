import os
import sys
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pymongo
import alpaca_trade_api as tradeapi
import requests

# Load environment variables
load_dotenv()
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')
ALPHA_VANTAGE_API = os.getenv('ALPHA_VANTAGE_API')

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("manage_bracket_orders.log"), logging.StreamHandler(sys.stdout)])

# Initialize MongoDB client
mongo_client = pymongo.MongoClient(MONGO_DB_CONN_STRING)
db = mongo_client['trading_db']
orders_collection = db['orders']
bad_performers_collection = db['bad_performers']

# Alpaca API setup
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url='https://paper-api.alpaca.markets')

def get_bracket_orders():
    try:
        logging.info("Fetching all bracket orders from MongoDB")
        orders = list(orders_collection.find({"order_class": "bracket"}))
        logging.info(f"Found {len(orders)} bracket orders")
        return orders
    except Exception as e:
        logging.error(f"Error fetching bracket orders: {e}")
        return []

def cancel_order(api, order_id):
    try:
        logging.info(f"Attempting to cancel order {order_id}")
        api.cancel_order(order_id)
        logging.info(f"Order {order_id} canceled successfully")
    except Exception as e:
        logging.error(f"Error canceling order {order_id}: {e}")

def place_stop_loss_order(api, symbol, qty, stop_price):
    logging.info(f"Preparing to place new stop loss order for {symbol}")
    logging.info(f"Qty: {qty}, Stop Price: {stop_price}")
    # Uncomment the following lines to actually place the order after verifying the output
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side='sell',
            type='stop',
            stop_price=stop_price,
            time_in_force='gtc'
        )
        logging.info(f"Placed new stop loss order: {order}")
        return order
    except Exception as e:
        logging.error(f"Error placing new stop loss order for {symbol}: {e}")
    return None

def place_market_sell_order(api, symbol, qty):
    logging.info(f"Preparing to place market sell order for {symbol}")
    logging.info(f"Qty: {qty}")
    # Uncomment the following lines to actually place the order after verifying the output
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side='sell',
            type='market',
            time_in_force='gtc'
        )
        logging.info(f"Placed market sell order: {order}")
        return order
    except Exception as e:
        logging.error(f"Error placing market sell order for {symbol}: {e}")
    return None

def get_current_price(symbol):
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={ALPHA_VANTAGE_API}"
        response = requests.get(url)
        data = response.json()
        if "Time Series (1min)" in data:
            last_refreshed = data["Meta Data"]["3. Last Refreshed"]
            current_price = float(data["Time Series (1min)"][last_refreshed]["4. close"])
            return current_price
        else:
            logging.error(f"Error fetching current price for {symbol}: {data}")
            return None
    except Exception as e:
        logging.error(f"Exception fetching current price for {symbol}: {e}")
        return None

def add_to_bad_performers(symbol):
    try:
        entry = {
            "symbol": symbol,
            "date_added": datetime.now(),
            "avoid_until": datetime.now() + timedelta(days=30)  # Avoid trading this symbol for 30 days
        }
        bad_performers_collection.update_one(
            {"symbol": symbol},
            {"$set": entry},
            upsert=True
        )
        logging.info(f"Added {symbol} to bad performers list")
    except Exception as e:
        logging.error(f"Error adding {symbol} to bad performers list: {e}")

def is_in_bad_performers(symbol):
    try:
        entry = bad_performers_collection.find_one({"symbol": symbol})
        if entry:
            if datetime.now() < entry["avoid_until"]:
                logging.info(f"{symbol} is in bad performers list until {entry['avoid_until']}")
                return True
            else:
                logging.info(f"{symbol} is no longer in the avoidance period")
                return False
        return False
    except Exception as e:
        logging.error(f"Error checking bad performers list for {symbol}: {e}")
        return False

def re_evaluate_and_replace_orders(api):
    orders = get_bracket_orders()
    for order in orders:
        try:
            symbol = order['symbol']
            if is_in_bad_performers(symbol):
                logging.info(f"{symbol} is in the bad performers list, skipping...")
                continue

            qty = float(order['qty'])
            limit_price = float(order['limit_price']) if order['limit_price'] else None
            stop_price = float(order['stop_price']) if order['stop_price'] else None
            take_profit_price = float(order.get('take_profit', {}).get('limit_price')) if order.get('take_profit', {}).get('limit_price') else None
            status = order['status']

            logging.info(f"Evaluating order {order['id']} for {symbol}")

            if status in ['new', 'open']:
                logging.info(f"Order {order['id']} is active, proceeding to cancel and replace")
                cancel_order(api, order['id'])
                place_bracket_order(api, symbol, qty, limit_price, stop_price, take_profit_price)
            else:
                logging.info(f"Order {order['id']} is not active, evaluating for re-placement")
                current_price = get_current_price(symbol)
                if current_price is not None and stop_price is not None:
                    if current_price < stop_price:
                        logging.info(f"Current price {current_price} is below stop price {stop_price}, placing market sell order and adding to bad performers")
                        place_market_sell_order(api, symbol, qty)
                        add_to_bad_performers(symbol)
                    else:
                        logging.info(f"Current price {current_price} is above stop price {stop_price}, placing stop loss order")
                        place_stop_loss_order(api, symbol, qty, stop_price)
                else:
                    logging.error(f"Could not fetch current price or stop price for {symbol}")

        except KeyError as ke:
            logging.error(f"KeyError while processing order {order['id']}: {ke}")
        except Exception as e:
            logging.error(f"Error while re-evaluating order {order['id']}: {e}")

if __name__ == "__main__":
    logging.info("Starting bracket order management process")
    re_evaluate_and_replace_orders(api)
    logging.info("Completed bracket order management process")
