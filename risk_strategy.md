# Crypto Trading Bot

This Python script implements a crypto trading bot that interacts with the Alpaca trading API to manage a diversified portfolio of crypto assets, options, and equities. The bot incorporates various risk management strategies, portfolio rebalancing, and real-time monitoring of positions.

## Main Classes

1. `CryptoAsset`: Represents a single crypto asset holding, including symbol, quantity, and USD value.
2. `MarketRegime`: Defines different market regimes based on volatility levels (low, normal, high, crisis).
3. `PriceHistory`: Manages historical price data for assets with a configurable window size.
4. `PortfolioManager`: Handles the overall portfolio, including adding assets, updating values, calculating portfolio balance and value, and making sell decisions.
5. `RiskManagement`: The main class that integrates all components, handles risk calculations, portfolio rebalancing, trade validation, and interfaces with the Alpaca API.

## Key Functions

1. `load_risk_params()`: Loads risk parameters from a JSON file.
2. `get_exchange_rate(base_currency, quote_currency)`: Fetches the exchange rate between two currencies using the Alpha Vantage API.
3. `fetch_account_details()`: Retrieves account details from the Alpaca API.
4. `black_scholes(S, K, T, r, sigma, option_type)`: Calculates the theoretical price of an option using the Black-Scholes model.
5. `calculate_greeks(S, K, T, r, sigma, option_type)`: Calculates the option Greeks (delta, gamma, theta, vega) for risk management.
6. `rebalance_portfolio()`: Rebalances the portfolio based on target allocations and risk parameters.
7. `validate_trade(symbol, qty, order_type)`: Validates a potential trade against various risk checks before placing the order.
8. `calculate_quantity(symbol, order_type)`: Determines the appropriate quantity to buy or sell based on risk parameters and current prices.
9. `get_current_price(symbol)`: Fetches the current market price for a given symbol, handling different asset types (crypto, options, equities).
10. `execute_profit_taking(symbol)` and `execute_stop_loss(symbol)`: Implements profit-taking and stop-loss strategies based on predefined criteria.

## Usage

To use this script, you'll need to set up the following:

1. Install the required Python libraries: `alpaca-trade-api`, `requests`, `alpha_vantage`, `python-dotenv`, `pandas`, `pymongo`, `scipy`, `numpy`.
2. Set up environment variables for API keys and MongoDB connection string in a `.env` file.
3. Customize the `risk_params.json` file with your desired risk parameters and allocation targets.
4. Run the script with `python script_name.py`.

Please note that this script is provided as-is and may require additional error handling, testing, and modifications to suit your specific needs. Always use caution and proper risk management when trading with real funds.

## Disclaimer

This script is for educational purposes only and does not constitute financial advice. Use at your own risk.