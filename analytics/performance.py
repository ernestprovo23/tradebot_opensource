import os
import sys
import json
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pymongo

# Load environment variables
load_dotenv()
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("logfile.log"), logging.StreamHandler(sys.stdout)])

# Initialize MongoDB client
mongo_client = pymongo.MongoClient(MONGO_DB_CONN_STRING)
db = mongo_client['trading_db']

def calculate_pnl():
    orders = list(db.orders.find({"status": "filled"}))
    pnl = 0
    for order in orders:
        try:
            filled_qty = float(order["filled_qty"])
            filled_avg_price = float(order["filled_avg_price"])
            if order["side"] == "buy":
                pnl -= filled_avg_price * filled_qty
            elif order["side"] == "sell":
                pnl += filled_avg_price * filled_qty
        except ValueError as e:
            logging.error(f"Error calculating PnL for order {order['id']}: {e}")
    return pnl

def calculate_win_rate(start_date, end_date):
    orders = list(db.orders.find({
        "status": "filled",
        "filled_at": {"$gte": start_date.isoformat(), "$lte": end_date.isoformat()}
    }))
    wins = 0
    total_trades = 0
    for order in orders:
        if order["side"] == "sell":
            total_trades += 1
            try:
                if "price" in order and float(order["filled_avg_price"]) > float(order["price"]):
                    wins += 1
            except ValueError as e:
                logging.error(f"Error calculating win rate for order {order['id']}: {e}")
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    return win_rate


def calculate_average_trade_duration(start_date, end_date):
    orders = list(db.orders.find({
        "status": "filled",
        "filled_at": {"$gte": start_date.isoformat(), "$lte": end_date.isoformat()}
    }))
    total_duration = timedelta()
    trade_count = 0
    for order in orders:
        if order["side"] == "sell":
            buy_order = db.orders.find_one({
                "symbol": order["symbol"],
                "side": "buy",
                "status": "filled",
                "filled_at": {"$lte": order["filled_at"]}
            })
            if buy_order:
                try:
                    buy_time = datetime.fromisoformat(buy_order["filled_at"].replace("Z", "+00:00"))
                    sell_time = datetime.fromisoformat(order["filled_at"].replace("Z", "+00:00"))
                    total_duration += sell_time - buy_time
                    trade_count += 1
                except ValueError as e:
                    logging.error(f"Error calculating trade duration for order {order['id']}: {e}")
    average_duration = total_duration / trade_count if trade_count > 0 else timedelta()
    return average_duration.total_seconds() / 60  # Return in minutes


def calculate_roi():
    account_info = db.account_info.find_one(sort=[("timestamp", -1)])
    if account_info:
        initial_balance = float(account_info.get("last_equity", 0))
        current_balance = float(account_info.get("equity", 0))
        if initial_balance != 0:
            roi = ((current_balance - initial_balance) / initial_balance) * 100
            return roi
        else:
            logging.error("Initial balance is zero, cannot calculate ROI.")
            return 0
    else:
        logging.error("Account info not found.")
        return 0


def store_performance_metrics(metrics):
    try:
        metrics['date_ingested'] = datetime.now()
        db.performance_metrics.insert_one(metrics)
        logging.info("Performance metrics stored successfully")
    except Exception as e:
        logging.error(f"Error storing performance metrics: {e}")


def get_account_snapshots(start_date, end_date):
    return list(db.account_info.find({
        "timestamp": {"$gte": start_date, "$lte": end_date}
    }).sort("timestamp", 1))


def calculate_metrics(snapshots):
    if not snapshots:
        return None

    initial = snapshots[0]
    final = snapshots[-1]

    initial_equity = float(initial["equity"])
    final_equity = float(final["equity"])

    pnl = final_equity - initial_equity
    roi = (pnl / initial_equity) * 100 if initial_equity != 0 else 0

    start_date = initial["timestamp"]
    end_date = final["timestamp"]

    return {
        "start_date": start_date,
        "end_date": end_date,
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "pnl": pnl,
        "roi": roi,
        "win_rate": calculate_win_rate(start_date, end_date),
        "avg_trade_duration": calculate_average_trade_duration(start_date, end_date)
    }


def get_ytd_metrics():
    start_date = datetime(datetime.now().year, 1, 1)
    return calculate_metrics(get_account_snapshots(start_date, datetime.now()))


def get_mtd_metrics():
    start_date = datetime(datetime.now().year, datetime.now().month, 1)
    return calculate_metrics(get_account_snapshots(start_date, datetime.now()))


def get_qtd_metrics():
    current_month = datetime.now().month
    quarter_start_month = ((current_month - 1) // 3) * 3 + 1
    start_date = datetime(datetime.now().year, quarter_start_month, 1)
    return calculate_metrics(get_account_snapshots(start_date, datetime.now()))


def get_wtd_metrics():
    start_date = datetime.now() - timedelta(days=datetime.now().weekday())
    return calculate_metrics(get_account_snapshots(start_date, datetime.now()))


def get_all_time_metrics():
    first_snapshot = db.account_info.find_one(sort=[("timestamp", 1)])
    if first_snapshot:
        return calculate_metrics(get_account_snapshots(first_snapshot["timestamp"], datetime.now()))
    return None



def get_performance_metrics():
    account_info = db.account_info.find_one(sort=[("timestamp", -1)])
    if account_info:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Calculate for the last 30 days
        return {
            "timestamp": end_date.isoformat(),
            "pnl": calculate_pnl(),
            "win_rate": calculate_win_rate(start_date, end_date),
            "average_trade_duration": calculate_average_trade_duration(start_date, end_date),
            "roi": calculate_roi(),
            "equity": float(account_info.get("equity", 0)),
            "last_equity": float(account_info.get("last_equity", 0)),
            "buying_power": float(account_info.get("buying_power", 0)),
            "cash": float(account_info.get("cash", 0)),
            "portfolio_value": float(account_info.get("portfolio_value", 0))
        }
    else:
        logging.error("Account info not found.")
        return None


def main():
    metrics = {
        "all_time": get_all_time_metrics(),
        "ytd": get_ytd_metrics(),
        "qtd": get_qtd_metrics(),
        "mtd": get_mtd_metrics(),
        "wtd": get_wtd_metrics(),
        "current": get_performance_metrics()
    }

    # Print nicely formatted JSON to console
    print(json.dumps(metrics, indent=2, default=str))

    # Log metrics in a business report format
    logging.info("----- Performance Metrics Report -----")
    for period, data in metrics.items():
        if data:
            logging.info(f"\n{period.upper()} Performance:")
            if period == "current":
                logging.info(f"  Timestamp: {data['timestamp']}")
                logging.info(f"  PnL: ${data['pnl']:.2f}")
                logging.info(f"  ROI: {data['roi']:.2f}%")
                logging.info(f"  Win Rate: {data['win_rate']:.2f}%")
                logging.info(f"  Avg Trade Duration: {data['average_trade_duration']:.2f} minutes")
                logging.info(f"  Equity: ${data['equity']:.2f}")
                logging.info(f"  Last Equity: ${data['last_equity']:.2f}")
                logging.info(f"  Buying Power: ${data['buying_power']:.2f}")
                logging.info(f"  Cash: ${data['cash']:.2f}")
                logging.info(f"  Portfolio Value: ${data['portfolio_value']:.2f}")
            else:
                logging.info(f"  Period: {data['start_date']} to {data['end_date']}")
                logging.info(f"  PnL: ${data['pnl']:.2f}")
                logging.info(f"  ROI: {data['roi']:.2f}%")
                logging.info(f"  Win Rate: {data['win_rate']:.2f}%")
                logging.info(f"  Avg Trade Duration: {data['avg_trade_duration']:.2f} minutes")

    # Store metrics in MongoDB
    store_performance_metrics(metrics)


if __name__ == "__main__":
    main()