import os
import requests
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_API_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_ENDPOINT = 'https://paper-api.alpaca.markets/v2/account'

def fetch_account_details():
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET_KEY
    }

    response = requests.get(ALPACA_ENDPOINT, headers=headers)

    if response.status_code == 200:
        account_info = response.json()
        account_details = {
            "equity": float(account_info.get('equity', 0)),
            "cash": float(account_info.get('cash', 0)),
            "portfolio_value": float(account_info.get('portfolio_value', 0)),
            "buying_power": float(account_info.get('buying_power', 0)),
            "status": account_info.get('status', 'UNKNOWN')
        }

        print(f"Account Details: {account_details}")
        return account_details
    else:
        print(f"Failed to fetch account details: {response.text}")
        return None

def calculate_risk_parameters(account_size):
    max_position_size = account_size * 0.92
    max_portfolio_size = account_size * 0.98
    max_drawdown = 30
    max_risk_per_trade = 0.30
    max_crypto_equity = account_size * 0.84
    max_equity_equity = account_size * 0.4

    return {
        "max_position_size": max_position_size,
        "max_portfolio_size": max_portfolio_size,
        "max_drawdown": max_drawdown,
        "max_risk_per_trade": max_risk_per_trade,
        "max_crypto_equity": max_crypto_equity,
        "max_equity_equity": max_equity_equity
    }

def update_risk_parameters_json(new_risk_parameters):
    file_path = os.path.join(os.getcwd(), 'risk_params.json')

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            existing_risk_parameters = json.load(file)
    else:
        existing_risk_parameters = {}

    existing_risk_parameters.update(new_risk_parameters)

    with open(file_path, 'w') as file:
        json.dump(existing_risk_parameters, file, indent=4)

if __name__ == "__main__":
    account_details = fetch_account_details()

    if account_details:
        account_size = account_details['equity']
        risk_parameters = calculate_risk_parameters(account_size)
        print(risk_parameters)
        update_risk_parameters_json(risk_parameters)
        print("Risk parameters updated successfully.")
    else:
        print("Failed to fetch account details. Risk parameters not updated.")
