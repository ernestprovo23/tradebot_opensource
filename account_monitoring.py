import time
import datetime
import threading
from risk_strategy import RiskManagement, risk_params
from credentials import ALPACA_API_KEY, ALPACA_SECRET_KEY
import alpaca_trade_api as tradeapi

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url='https://paper-api.alpaca.markets')

risk_management = RiskManagement(api, risk_params)

def monitor_hourly():
    while True:
        risk_management.monitor_account_status()
        risk_management.monitor_positions()
        risk_management.report_profit_and_loss()
        risk_management.update_risk_parameters()
        print("Hourly monitoring completed. Pausing for 1 hour.")
        time.sleep(60*60)  # Pause for 1 hour

def monitor_daily():
    while True:
        drawdown = risk_management.calculate_drawdown()
        if drawdown is not None:
            print(f"Daily drawdown: {drawdown * 100}%")
        print("Daily monitoring completed. Pausing for 24 hours.")
        time.sleep(60*60*24)  # Pause for 24 hours


# Create threads
hourly_thread = threading.Thread(target=monitor_hourly)
daily_thread = threading.Thread(target=monitor_daily)

# Start threads
hourly_thread.start()
daily_thread.start()
