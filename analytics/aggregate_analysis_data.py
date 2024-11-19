import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
import pymongo
import traceback

# Load environment variables
load_dotenv()
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("logfile.log"), logging.StreamHandler(sys.stdout)])

# Initialize MongoDB client
try:
    mongo_client = pymongo.MongoClient(MONGO_DB_CONN_STRING)
    db_stock_data = mongo_client['stock_data']
    db_predictions = mongo_client['predictions']
    logging.info("Successfully connected to MongoDB")
except Exception as e:
    logging.error(f"Failed to connect to MongoDB: {str(e)}")
    sys.exit(1)


def calculate_financial_ratios(balance_sheet):
    logging.debug(f"Calculating financial ratios for balance sheet: {balance_sheet}")
    if balance_sheet:
        try:
            current_assets = balance_sheet.get('current_assets', 0)
            current_liabilities = balance_sheet.get('current_liabilities', 0)
            inventory = balance_sheet.get('inventory', 0)
            total_liabilities = balance_sheet.get('total_liabilities', 0)
            shareholders_equity = balance_sheet.get('shareholders_equity', 0)

            current_ratio = current_assets / current_liabilities if current_liabilities else 0
            quick_ratio = (current_assets - inventory) / current_liabilities if current_liabilities else 0
            debt_to_equity = total_liabilities / shareholders_equity if shareholders_equity else 0

            ratios = {
                "current_ratio": current_ratio,
                "quick_ratio": quick_ratio,
                "debt_to_equity": debt_to_equity
            }
            logging.debug(f"Calculated ratios: {ratios}")
            return ratios
        except Exception as e:
            logging.error(f"Error calculating financial ratios: {str(e)}")
    return {}

def aggregate_data(symbol):
    logging.info(f"Aggregating data for symbol: {symbol}")
    try:
        # Aggregating balance sheet data
        balance_sheet = db_stock_data.balance_sheets.find_one({"symbol": symbol})
        logging.debug(f"Balance sheet for {symbol}: {balance_sheet}")
        financial_ratios = calculate_financial_ratios(balance_sheet)

        # Aggregating cash flow data
        cash_flow = db_stock_data.cashflows.find_one({"symbol": symbol})
        logging.debug(f"Cash flow for {symbol}: {cash_flow}")

        # Aggregating company overview data
        company_overview = db_stock_data.company_overviews.find_one({"symbol": symbol})
        logging.debug(f"Company overview for {symbol}: {company_overview}")

        # Aggregating earnings calendar data
        earnings_calendar = db_stock_data.earnings_calendars.find_one({"symbol": symbol})
        logging.debug(f"Earnings calendar for {symbol}: {earnings_calendar}")

        # Aggregating news sentiment data
        news_sentiment = db_stock_data.news_sentiment_data.find_one({"symbol": symbol})
        if news_sentiment:
            logging.debug(f"News sentiment for {symbol}: {news_sentiment}")
            overall_sentiment = news_sentiment.get('overall_sentiment', {})
            sentiment_score = overall_sentiment.get('average_score', 0)
            ticker_sentiment = news_sentiment.get('ticker_sentiment', {}).get(symbol, {})
            ticker_score = ticker_sentiment.get('average_score', 0)
        else:
            logging.warning(f"No news sentiment data found for {symbol}")
            sentiment_score = 0
            ticker_score = 0

        logging.debug(f"Overall sentiment score for {symbol}: {sentiment_score}")
        logging.debug(f"Ticker-specific sentiment score for {symbol}: {ticker_score}")

        # Aggregating technicals data
        technicals = db_stock_data.technicals.find_one({"symbol": symbol})
        logging.debug(f"Technicals for {symbol}: {technicals}")

        # Aggregating time series data
        time_series_data = list(db_stock_data.aggregated_stock_data.find({"symbol": symbol}))
        logging.debug(f"Time series data count for {symbol}: {len(time_series_data)}")

        aggregated_data = {
            "symbol": symbol,
            "balance_sheet": balance_sheet,
            "cash_flow": cash_flow,
            "company_overview": company_overview,
            "earnings_calendar": earnings_calendar,
            "overall_sentiment_score": sentiment_score,
            "ticker_sentiment_score": ticker_score,
            "technicals": technicals,
            "time_series_data": time_series_data,
            "financial_ratios": financial_ratios,
            "timestamp": datetime.now()
        }

        logging.info(f"Data aggregation completed for {symbol}")
        return aggregated_data
    except Exception as e:
        logging.error(f"Error aggregating data for {symbol}: {str(e)}")
        logging.error(traceback.format_exc())
        return None


def store_aggregated_data(symbol):
    logging.info(f"Storing aggregated data for symbol: {symbol}")
    try:
        db_analytics = mongo_client['stock_analysis']
        aggregated_data = aggregate_data(symbol)
        if aggregated_data:
            result = db_analytics.aggregated_data.insert_one(aggregated_data)
            logging.info(f"Aggregated data for {symbol} stored successfully. Inserted ID: {result.inserted_id}")
        else:
            logging.warning(f"Skipping {symbol} due to missing aggregated data")
    except Exception as e:
        logging.error(f"Error storing aggregated data for {symbol}: {str(e)}")
        logging.error(traceback.format_exc())



def main():
    # Get the current month and year
    current_month_year = datetime.now().strftime("%B_%Y")

    # Retrieve symbols from predictions collection
    predictions_collection_name = f"{current_month_year}"
    logging.info(f"Retrieving symbols from collection: {predictions_collection_name}")

    try:
        # List all collections in the predictions database
        collections = db_predictions.list_collection_names()
        logging.info(f"Collections in the predictions database: {collections}")

        # Check if the predictions collection exists
        if predictions_collection_name not in collections:
            logging.warning(f"Collection {predictions_collection_name} does not exist in the predictions database.")
        else:
            # Print a sample document from the collection
            sample_doc = db_predictions[predictions_collection_name].find_one()
            logging.info(f"Sample document from {predictions_collection_name}: {sample_doc}")

        symbols = db_predictions[predictions_collection_name].distinct("symbol")
        logging.info(f"Retrieved {len(symbols)} symbols")
        logging.debug(f"Symbols: {symbols}")  # This will print all symbols if logging level is set to DEBUG

    except Exception as e:
        logging.error(f"Error retrieving symbols: {str(e)}")
        logging.error(traceback.format_exc())
        symbols = []

    # Call the function for each symbol
    for symbol in symbols:
        try:
            store_aggregated_data(symbol)
        except Exception as e:
            logging.error(f"Error processing symbol {symbol}: {str(e)}")
            logging.error(traceback.format_exc())

    logging.info("Script execution completed")

if __name__ == "__main__":
    main()