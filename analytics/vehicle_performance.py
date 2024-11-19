import os
import sys
import json
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pymongo
import requests

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


def calculate_pnl_for_orders(vehicle, open_positions, start_date, end_date):
    pipeline = [
        {"$match": {
            "status": "filled",
            "asset_class": vehicle,
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

    realized_pnl = 0
    unrealized_pnl = 0

    for result in results:
        symbol = result['_id']
        realized_pnl += result['total_sell'] - result['total_buy']

        # Check if there's an open position for this symbol
        open_position = next((p for p in open_positions if p['symbol'] == symbol and p['asset_class'] == vehicle), None)
        if open_position and result['net_qty'] != 0:
            current_price = float(open_position['current_price'])
            unrealized_pnl += result['net_qty'] * (
                        current_price - (result['total_buy'] - result['total_sell']) / result['net_qty'])

    total_pnl = realized_pnl + unrealized_pnl
    return total_pnl


def calculate_win_rate_for_orders(vehicle, start_date, end_date, open_positions):
    pipeline = [
        {"$match": {
            "status": "filled",
            "asset_class": vehicle,
            "side": "sell",
            "filled_at": {"$gte": start_date.isoformat(), "$lte": end_date.isoformat()}
        }},
        {"$addFields": {
            "numeric_filled_avg_price": {"$toDouble": "$filled_avg_price"}
        }},
        {"$lookup": {
            "from": "orders",
            "let": {"symbol": "$symbol", "sell_time": "$filled_at"},
            "pipeline": [
                {"$match": {
                    "$expr": {
                        "$and": [
                            {"$eq": ["$symbol", "$$symbol"]},
                            {"$eq": ["$side", "buy"]},
                            {"$eq": ["$status", "filled"]},
                            {"$lte": ["$filled_at", "$$sell_time"]}
                        ]
                    }
                }},
                {"$addFields": {
                    "numeric_filled_avg_price": {"$toDouble": "$filled_avg_price"}
                }},
                {"$sort": {"filled_at": -1}},
                {"$limit": 1}
            ],
            "as": "buy_order"
        }},
        {"$unwind": "$buy_order"},
        {"$project": {
            "is_win": {"$cond": [{"$gt": ["$numeric_filled_avg_price", "$buy_order.numeric_filled_avg_price"]}, 1, 0]}
        }},
        {"$group": {
            "_id": None,
            "total_trades": {"$sum": 1},
            "wins": {"$sum": "$is_win"}
        }}
    ]

    result = list(db.orders.aggregate(pipeline))
    total_trades = 0
    wins = 0

    if result:
        total_trades = result[0]['total_trades']
        wins = result[0]['wins']

    # Include open positions in the win rate calculation
    for position in open_positions:
        if position['asset_class'] == vehicle:
            try:
                avg_entry_price = float(position['avg_entry_price'])
                current_price = float(position['current_price'])
                if current_price > avg_entry_price:
                    wins += 1
                total_trades += 1
            except ValueError as e:
                logging.error(f"Error calculating win rate for open position {position['asset_id']}: {e}")

    return (wins / total_trades) * 100 if total_trades > 0 else 0


def calculate_avg_trade_duration_for_orders(vehicle, start_date, end_date):
    pipeline = [
        {"$match": {
            "status": "filled",
            "asset_class": vehicle,
            "side": "sell",
            "filled_at": {"$gte": start_date.isoformat(), "$lte": end_date.isoformat()}
        }},
        {"$lookup": {
            "from": "orders",
            "let": {"symbol": "$symbol", "sell_time": "$filled_at"},
            "pipeline": [
                {"$match": {
                    "$expr": {
                        "$and": [
                            {"$eq": ["$symbol", "$$symbol"]},
                            {"$eq": ["$side", "buy"]},
                            {"$eq": ["$status", "filled"]},
                            {"$lte": ["$filled_at", "$$sell_time"]}
                        ]
                    }
                }},
                {"$sort": {"filled_at": -1}},
                {"$limit": 1}
            ],
            "as": "buy_order"
        }},
        {"$unwind": "$buy_order"},
        {"$project": {
            "duration": {"$subtract": [{"$toDate": "$filled_at"}, {"$toDate": "$buy_order.filled_at"}]}
        }},
        {"$group": {
            "_id": None,
            "avg_duration": {"$avg": "$duration"}
        }}
    ]

    result = list(db.orders.aggregate(pipeline))
    if result:
        return result[0]['avg_duration'] / (1000 * 60)  # Convert milliseconds to minutes
    return 0


def get_vehicle_metrics(vehicle, start_date, end_date, open_positions):
    return {
        "pnl": calculate_pnl_for_orders(vehicle, open_positions, start_date, end_date),
        "win_rate": calculate_win_rate_for_orders(vehicle, start_date, end_date, open_positions),
        "avg_trade_duration": calculate_avg_trade_duration_for_orders(vehicle, start_date, end_date)
    }



def store_performance_metrics(metrics):
    try:
        metrics['date_ingested'] = datetime.now()
        db.performance_metrics.insert_one(metrics)
        logging.info("Performance metrics stored successfully")
    except Exception as e:
        logging.error(f"Error storing performance metrics: {e}")


def main():
    vehicles = ["crypto", "us_option", "us_equity"]
    end_date = datetime.now()

    mtd_start = datetime(end_date.year, end_date.month, 1)
    ytd_start = datetime(end_date.year, 1, 1)

    open_positions = fetch_open_positions()

    metrics = {
        "timestamp": end_date.isoformat(),
        "mtd": {},
        "ytd": {}
    }

    for vehicle in vehicles:
        metrics["mtd"][vehicle] = get_vehicle_metrics(vehicle, mtd_start, end_date, open_positions)
        metrics["ytd"][vehicle] = get_vehicle_metrics(vehicle, ytd_start, end_date, open_positions)

    # Print nicely formatted JSON to console
    print(json.dumps(metrics, indent=2))

    # Log metrics in a business report format
    logging.info("----- Performance Metrics Report -----")
    for period in ["MTD", "YTD"]:
        logging.info(f"\n{period} Performance:")
        for vehicle, data in metrics[period.lower()].items():
            logging.info(f"  {vehicle.upper()}:")
            logging.info(f"    PnL: ${data['pnl']:.2f}")
            logging.info(f"    Win Rate: {data['win_rate']:.2f}%")
            logging.info(f"    Avg Trade Duration: {data['avg_trade_duration']:.2f} minutes")

    # Store metrics in MongoDB
    store_performance_metrics(metrics)

if __name__ == "__main__":
    main()
