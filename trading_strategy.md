# Algorithmic Trading Strategy Documentation

## Overview
This document provides a comprehensive overview of our algorithmic trading strategy, including the key components, data retrieval and filtering processes, pair selection, order placement, and risk management. The strategy aims to identify promising stock pairs based on fundamental analysis and growth potential, and execute trades using technical indicators and risk management parameters.

## Data Retrieval and Filtering
### Company Overview Retrieval
- Retrieve company overview data from the Alpha Vantage API for a list of stock tickers.
- Use concurrent.futures with a ThreadPoolExecutor to parallelize API requests and improve performance.

### Data Filtering
- Filter the retrieved company overview data based on sector-specific thresholds.
- Check various financial metrics (market capitalization, EBITDA, PE ratio, EPS, beta) against corresponding thresholds for each sector.
- Skip companies that fail to meet any of the criteria.

### Market Condition Adjustments
- Adjust filter thresholds based on current market conditions using the `adjust_filters_for_market_conditions` function.
- Take into account market condition adjustments defined in the `sector_filters` module.

### Data Storage
- Store the filtered company overview data in a MongoDB collection named 'company_overviews'.
- Use the `MongoManager` class to insert documents into the collection with deduplication, ensuring only unique documents are stored.

## Pair Selection
### Data Retrieval
- Fetch the filtered company overview data from the MongoDB database using the `fetch_company_overviews` function.
- Convert specific columns to numeric format and handle any errors that may occur during the process.

### Data Analysis
- Analyze the data before and after applying filters.
- Calculate descriptive statistics and generate histograms for visualizing the distribution of financial metrics (Profit Margin, P/E Ratio, Return on Equity, EV to EBITDA, Quarterly Earnings Growth YoY).

### Pair Selection Criteria
- Apply additional filters based on financial metrics and growth focus:
  - Profit Margin greater than -10% (allowing for growth-focused companies)
  - Positive P/E Ratio (ensuring positive earnings)
  - Return on Equity greater than -20% (including companies investing heavily in growth)
  - EV to EBITDA between 0 and 50 (allowing higher ratios typical of growth companies)
  - Positive Quarterly Earnings Growth YoY
- Sort the filtered companies by Market Capitalization in descending order and select the top 20 pairs.

### Data Storage
- Store the selected pairs in the MongoDB database using the `store_selected_pairs` function.
- Perform an upsert operation, updating existing documents or inserting new ones based on the symbol and date added.

### Financial Data Retrieval
- Fetch additional financial data for the selected pairs using the Alpha Vantage API.
- Retrieve the current price, Simple Moving Average (SMA), and Relative Strength Index (RSI) for each selected pair using multi-threading.

## Order Placement
### Symbol Selection
- Retrieve the selected stock symbols from the MongoDB database using the `get_symbols_from_mongodb` function.

### Technical Indicators
- Fetch technical indicator data for each symbol using the Alpha Vantage API:
  - Daily price data (close price)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Simple Moving Average (SMA)

### Order Placement Conditions
- Evaluate a set of conditions based on the technical indicators to determine whether to place a buy order for a symbol:
  - RSI less than or equal to 60
  - MACD greater than or equal to the MACD signal line
  - Close price greater than or equal to the 30-day SMA

### Position and Order Checks
- Before placing an order, check if there is already an existing position or an open order for the symbol.
- If either condition is true, skip placing a new order.

### Bracket Order Placement
- If all conditions are met and there is no existing position or open order, calculate the number of shares to buy based on the portfolio balance and maximum risk per trade.
- Place a bracket order using the Alpaca API, specifying the limit price, take profit, and stop loss parameters.

### Notifications
- Upon successful order placement, send a message to a Microsoft Teams channel using a webhook URL to notify about the trade details.

## Risk Management
### Portfolio Allocation
- Allocate up to x (max_crypto_equity) to cryptocurrencies and the remaining equity to commodities (rate calculated in risk_strategy.py).
- Maintain diversification by ensuring no single position exceeds 30% of the portfolio.

### Position Sizing
- Limit maximum position size to p (max_position_size) (rate calculated in risk_strategy.py).
- Adjust position sizes based on asset prices, with smaller positions for higher-priced assets.

### Trade Validation
- Validate trades against risk parameters, cash holdings, and equity allocations.
- Reject trades that exceed daily trade limits or violate risk parameters.

### Rebalancing
- Rebalance positions when crypto or commodity allocations exceed 75% of equity.
- Sell a portion of high-volatility positions to maintain balance.

### Risk Adjustment
- Dynamically adjust risk parameters based on account performance.
- Increase risk parameters when profitable, decrease when experiencing losses.
- Limit maximum portfolio size to a specified value (max_portfolio_size).
- Restrict maximum drawdown to a specified percentage (max_drawdown).
- Cap maximum risk per trade at a specified percentage (max_risk_per_trade).

### Profit Taking and Stop Loss
- Execute profit-taking by selling a portion of a position when it reaches a specified gain percentage.
- Implement stop-loss by selling the entire position when it reaches a specified loss percentage.

### Momentum and Trend Following
- Generate momentum signals based on price changes since purchase.
- Consider selling positions when momentum turns negative.

### Monitoring and Reporting
- Monitor account status, positions, and profit/loss.
- Generate portfolio summary reports with position details and risk parameters.
- Send notifications and updates to stakeholders via messaging platforms.

## Execution and Automation
- The `bracket_order.py` script is designed to run automatically on a cron job every 5 minutes during trading hours.
- This enables the trading strategy to continuously monitor the market and place orders based on the latest technical indicators and risk assessments.

## Conclusion
The algorithmic trading strategy outlined in this document combines fundamental analysis, technical indicators, and risk management to identify promising stock pairs and execute trades accordingly. By leveraging data retrieval, filtering, pair selection, order placement, and risk management techniques, the strategy aims to maximize returns while controlling risk exposure. The automated execution and monitoring components ensure efficient and systematic trade execution, contributing to the overall effectiveness and robustness of the trading strategy.

Here's a summary of the steps involved in building the price prediction modeling based on the provided information:

# Price Prediction Modeling

## Data Collection
1. Historical data is collected from various sources using the following scripts:
   - `balance_sheets.py`: Fetches balance sheet data.
   - `cash_flow.py`: Retrieves cash flow data.
   - `earnings_calendar.py`: Obtains earnings calendar information.
   - `income_statements.py`: Collects income statement data.
   - `news_sentiment.py`: Gathers news sentiment data.
   - `selected_pairs_ts_data.py`: Fetches time series data for selected pairs.

2. These scripts are designed to capture historical data from different sources and store it in a MongoDB database.

## Data Preparation
1. The `ML_data_prep.py` script is responsible for preparing the collected data for machine learning.

2. The script contains an `ETLPipeline` class that handles the data extraction, transformation, and loading process.

3. The `clean_and_transform` method of the `ETLPipeline` class performs data cleaning and transformation based on the source collection:
   - For balance sheet, cash flow, and income statement data:
     - Extracts the relevant data items from the source document.
     - Converts the data into a pandas DataFrame.
     - Converts numeric columns to numeric data type.
     - Checks for the presence of required columns.
     - Returns the cleaned data as a list of dictionaries.
   - For news sentiment data:
     - Extracts the relevant sentiment data from the source document.
     - Returns the cleaned sentiment data as a dictionary.

4. The `fetch_clean_store` method of the `ETLPipeline` class performs the following steps:
   - Fetches documents from the source collection in MongoDB.
   - Iterates over each document and applies the `clean_and_transform` method to clean and transform the data.
   - Creates a new document with the cleaned data, symbol, processing timestamp, and version.
   - Appends the processed document to a list.
   - Inserts the processed documents into the target collection in MongoDB.

5. The `ETLPipeline` class is instantiated with the MongoDB connection string obtained from environment variables.

6. The `fetch_clean_store` method is called for each source collection (`balance_sheet`, `cash_flow`, `technicals`, `news_sentiment_data`) to fetch, clean, and store the data in the corresponding target collections (`ML_balancesheets`, `ML_cashflows`, `ML_incomestatements`, `ML_sentiment_data`).

## Next Steps
1. With the historical data collected and prepared, the next step would be to further preprocess the data for machine learning tasks. This may involve:
   - Handling missing values.
   - Scaling or normalizing the data.
   - Encoding categorical variables.
   - Splitting the data into training and testing sets.

2. Once the data is preprocessed, various machine learning algorithms can be applied to build price prediction models. Some common approaches include:
   - Linear regression
   - Time series analysis (e.g., ARIMA, LSTM)
   - Ensemble methods (e.g., Random Forest, Gradient Boosting)
   - Deep learning models (e.g., Convolutional Neural Networks, Recurrent Neural Networks)

3. The performance of the trained models can be evaluated using appropriate evaluation metrics such as mean squared error (MSE), mean absolute error (MAE), or root mean squared error (RMSE).

4. The best-performing model can be selected and further fine-tuned or optimized using techniques like hyperparameter tuning or cross-validation.

5. Finally, the trained model can be used to make price predictions on new or unseen data.

It's important to note that building an accurate and reliable price prediction model requires careful data preprocessing, feature engineering, model selection, and validation. The specific techniques and algorithms used may vary depending on the characteristics of the data and the requirements of the prediction task.

Here's a summary of how the price prediction modeling is built out using the provided scripts:

# Price Prediction Modeling

## Data Retrieval and Preparation
1. `balance_sheets.py`, `cash_flow.py`, `earnings_calendar.py`, `income_statements.py`, `news_sentiment.py`, and `selected_pairs_ts_data.py` scripts are used to capture historical data from various sources and store it in a MongoDB database.

2. `ML_data_prep.py` script is responsible for preparing the collected data for machine learning. It performs the following steps:
   - Fetches data from the source collections in MongoDB.
   - Cleans and transforms the data based on the source collection type (balance sheet, cash flow, income statement, or news sentiment data).
   - Flattens nested structures and extracts relevant data.
   - Converts data types to numeric where possible and handles missing values.
   - Stores the cleaned and transformed data in target collections in MongoDB.

3. `phase1ML.py` script is used to further process the data for machine learning. It performs the following steps:
   - Loads data from the MongoDB collections.
   - Integrates sentiment data with time series data.
   - Replaces missing values and ensures proper data types.
   - Stores the processed data in a separate MongoDB collection for each symbol.

## Model Building and Training
1. `model_building.py` script is responsible for building and training the neural network model. It performs the following steps:
   - Loads the processed data from MongoDB collections.
   - Prepares the data for model training by scaling features, selecting relevant features, and splitting into train/test sets.
   - Performs hyperparameter optimization to find the best model configurations using GridSearchCV.
   - Builds the neural network model using the optimal hyperparameters.
   - Trains the model using K-Fold cross-validation and evaluates its performance on a test set.
   - Saves the trained model, scalers, and data with versioning and cleanup of old files.
   - Stores the predicted prices and datetime of calculation in a MongoDB collection.

2. `model_training.py` script is used to further train and update the models. It performs the following steps:
   - Retrieves the latest model files, scalers, and data for each symbol.
   - Loads the data files (X_train, X_test, y_train, y_test) and the trained model.
   - Retrains the model on the latest data.
   - Evaluates the model's performance using mean squared error (MSE).
   - Saves the updated model and training metadata (MSE, model version, train/test sizes, timestamp) for each symbol.

## Model Evaluation and Prediction
1. The `evaluate_and_predict` function in `model_building.py` script is used to evaluate the trained model on the test set and provide scaled predictions. It performs the following steps:
   - Evaluates the model's performance using metrics such as loss, mean absolute error (MAE), and mean absolute percentage error (MAPE).
   - Generates predictions using the trained model.
   - Rescales the predictions back to the original scale using the target scaler.
   - Returns the rescaled predictions and evaluation metrics.

2. The predicted prices and datetime of calculation are stored in a MongoDB collection for further analysis and use in the trading strategy.

## Logging and Monitoring
1. The scripts utilize the `logging` module to log important information, including progress updates, error messages, and model performance metrics.
2. Log files are generated to keep track of the script execution and any issues encountered.
3. Sample predictions and training metadata are logged for monitoring and debugging purposes.

Overall, the price prediction modeling process involves collecting historical data, preparing and processing the data for machine learning, building and training neural network models, evaluating model performance, and storing predictions for use in the trading strategy. The scripts are designed to handle multiple symbols and perform model versioning and cleanup to manage the model files efficiently.


## options trading section: 
Here's a summary of how the prediction and options trading components are integrated into the overall trading strategy:

# Prediction and Options Trading

## Price Prediction
1. The `prediction.py` script is responsible for generating price predictions using the trained models. It performs the following steps:
   - Retrieves the latest model files, scalers, and processed data for each symbol from the MongoDB database.
   - Loads the trained model, feature scaler, and target scaler for each symbol.
   - Processes the new input data by scaling the features using the loaded feature scaler.
   - Predicts the prices using the trained model and inverse transforms the predictions back to the original scale using the target scaler.
   - Saves the predicted prices, prediction dates, and metadata to a MongoDB collection named after the current month and year.
   - Logs the prediction details and any errors encountered during the process.

2. The script utilizes the `TeamsCommunicator` module to send notifications and logs to a designated Microsoft Teams channel for monitoring and logging purposes.

3. The script is designed to handle multiple symbols and retrieves the necessary files and data from the respective directories and MongoDB collections.

## Options Trading
1. The `options_bracket_order.py` script (currently under development) focuses on leveraging the price predictions to make informed options trading decisions. It performs the following steps:
   - Retrieves the predicted prices from the MongoDB collection populated by the `prediction.py` script.
   - Fetches the current stock prices for each symbol using the Alpha Vantage API.
   - Identifies the best options contract for each symbol based on the predicted price and current price.
     - If the predicted price is higher than the current price, it looks for call options.
     - If the predicted price is lower than the current price, it looks for put options.
   - Calculates the Greeks (delta, gamma, theta, vega) for the selected options contract using the Black-Scholes model.
   - Prepares the order details for each symbol, including the options contract symbol, strike price, and calculated Greeks.
   - Logs the order details and any errors encountered during the process.

2. The script utilizes concurrent processing with `ThreadPoolExecutor` to handle multiple symbols efficiently.

3. The options contracts data is stored in a separate MongoDB collection (`options_contracts`) for easy retrieval and analysis.

## Integration with Trading Strategy
1. The price prediction and options trading components are integral parts of the overall trading strategy.

2. The `prediction.py` script generates price predictions for a future date (e.g., 180 days) based on the trained models. These predictions are stored in a MongoDB collection for further analysis and decision-making.

3. The `options_bracket_order.py` script leverages the price predictions to identify the best options contracts to trade. By comparing the predicted prices with the current prices, it determines whether to focus on call options (if the predicted price is higher) or put options (if the predicted price is lower).

4. The script calculates the Greeks for the selected options contracts using the Black-Scholes model. These metrics provide insights into the sensitivity of the options prices to various factors such as underlying asset price, time to expiration, and volatility.

5. The prepared order details, including the options contract symbol, strike price, and calculated Greeks, are logged and can be used to execute the actual trades through a trading platform or API.

6. By leveraging options trading, the trading strategy aims to maximize investment returns while managing risk. Options provide the flexibility to profit from both bullish and bearish market conditions and allow for leveraged positions compared to holding long-term equity positions.

7. The integration of price predictions and options trading enables the trading strategy to make data-driven decisions and adapt to changing market conditions. The predictions provide a forward-looking view, while the options trading component allows for strategic positioning and risk management.

Overall, the prediction and options trading components work together to enhance the trading strategy's ability to generate profitable trades and manage risk effectively. The combination of machine learning-based predictions and options trading techniques allows for a more sophisticated and adaptive approach to trading in the financial markets.


## Price Prediction
1. The `prediction.py` script generates price predictions for a future date (e.g., 180 days) based on trained machine learning models.
2. The script retrieves the latest model files, scalers, and processed data for each symbol from the MongoDB database.
3. It processes the new input data, scales the features, and predicts the prices using the trained models.
4. The predicted prices, prediction dates, and metadata are stored in a MongoDB collection for further analysis and decision-making.
5. The script utilizes the `TeamsCommunicator` module to send notifications and logs to a designated Microsoft Teams channel for monitoring and logging purposes.

## Options Trading
1. The `options_bracket_order.py` script leverages the price predictions to identify the best options contracts to trade.
2. It retrieves the predicted prices from the MongoDB collection populated by the `prediction.py` script and fetches the current stock prices for each symbol using the Alpha Vantage API.
3. Based on the comparison between the predicted prices and current prices, the script determines whether to focus on call options (if the predicted price is higher) or put options (if the predicted price is lower).
4. It calculates the Greeks (delta, gamma, theta, vega) for the selected options contracts using the Black-Scholes model to assess the sensitivity of the options prices to various factors.
5. The script prepares the order details for each symbol, including the options contract symbol, strike price, and calculated Greeks.
6. The options contracts data is stored in a separate MongoDB collection (`options_contracts`) for easy retrieval and analysis.

## Order Placement
(No changes to this section)

## Risk Management
(No changes to this section)

## Execution and Automation
1. The `prediction.py` script can be scheduled to run automatically at regular intervals (e.g., daily) to generate price predictions based on the latest data.
2. The `options_bracket_order.py` script can be triggered to run after the `prediction.py` script completes, leveraging the generated price predictions to identify options trading opportunities.
3. The `bracket_order.py` script continues to run on a cron job every 5 minutes during trading hours to monitor the market and place orders based on the latest technical indicators and risk assessments.

## Conclusion
The refined algorithmic trading strategy incorporates machine learning-based price predictions and options trading techniques to enhance its ability to identify profitable trading opportunities and manage risk effectively. By leveraging data-driven insights and the flexibility of options trading, the strategy aims to adapt to changing market conditions and maximize returns while controlling risk exposure.

The integration of price predictions and options trading allows for a more sophisticated and adaptive approach to trading. The predictions provide a forward-looking view of potential price movements, while the options trading component enables strategic positioning and risk management.

The automated execution and monitoring components ensure efficient and systematic trade execution, contributing to the overall effectiveness and robustness of the trading strategy. The use of logging, notifications, and data storage in MongoDB facilitates monitoring, analysis, and continuous improvement of the strategy.

By combining fundamental analysis, technical indicators, machine learning, options trading, and risk management techniques, the refined algorithmic trading strategy seeks to capitalize on market opportunities and generate sustainable returns in the dynamic and competitive financial markets.