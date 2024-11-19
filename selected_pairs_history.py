import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import os
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("select_pairs_logfile.log"), logging.StreamHandler(sys.stdout)])

# Database configuration
MONGO_DB_CONN_STRING = os.getenv("MONGO_DB_CONN_STRING")
DB_NAME = "stock_data"
DB_NAME_TRADE = "trading_db"

# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API")
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
ti = TechIndicators(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

# MongoDB connection
client = MongoClient(MONGO_DB_CONN_STRING)
db = client[DB_NAME]
db_trade = client[DB_NAME_TRADE]
company_overviews_collection = db["company_overviews"]
selected_pairs_collection = db["selected_pairs"]
bad_performers_collection = db_trade["bad_performers"]

def get_current_price_and_sma(symbol, period=20):
    try:
        daily_data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
        sma_data, _ = ti.get_sma(symbol=symbol, interval='daily', time_period=period)
        current_price = daily_data['4. close'].iloc[-1]
        sma_value = sma_data['SMA'].iloc[-1]
        print(f"get_current_price_and_sma - {symbol}: current_price={current_price}, sma_value={sma_value}")
        return symbol, current_price, sma_value
    except Exception as e:
        logging.error(f"Error fetching price and SMA for {symbol}: {e}")
        return symbol, np.nan, np.nan

def get_rsi(symbol, period=14):
    try:
        rsi_data, _ = ti.get_rsi(symbol=symbol, interval='daily', time_period=period)
        rsi_value = rsi_data['RSI'].iloc[-1]
        print(f"get_rsi - {symbol}: rsi_value={rsi_value}")
        return symbol, rsi_value
    except Exception as e:
        logging.error(f"Error fetching RSI for {symbol}: {e}")
        return symbol, np.nan

def fetch_company_overviews():
    try:
        logging.info("Fetching company overviews...")
        documents = company_overviews_collection.find({}, {'_id': 0})
        df = pd.DataFrame(list(documents))
        if df.empty:
            logging.info("Fetched DataFrame is empty.")
        else:
            logging.info(f"Fetched {len(df)} documents.")
            numeric_cols = [
                "52WeekLow", "52WeekHigh", "50DayMovingAverage", "200DayMovingAverage",
                "AnalystRatingBuy", "AnalystRatingHold", "AnalystRatingSell", "AnalystRatingStrongBuy",
                "AnalystRatingStrongSell", "AnalystTargetPrice", "Beta", "BookValue", "DilutedEPSTTM",
                "DividendPerShare", "DividendYield", "EBITDA", "EPS", "EVToEBITDA", "EVToRevenue",
                "ForwardPE", "GrossProfitTTM", "MarketCapitalization", "OperatingMarginTTM", "PEGRatio",
                "PERatio", "PriceToBookRatio", "PriceToSalesRatioTTM", "ProfitMargin", "QuarterlyEarningsGrowthYOY",
                "QuarterlyRevenueGrowthYOY", "ReturnOnAssetsTTM", "ReturnOnEquityTTM", "RevenuePerShareTTM",
                "RevenueTTM", "SharesOutstanding", "TrailingPE"
            ]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        logging.error(f"An error occurred while fetching company overviews: {e}")
        return pd.DataFrame()


def filter_bad_performers(df):
    try:
        current_date = datetime.now()
        bad_performers = list(bad_performers_collection.find())
        bad_symbols_no_date = [bp["symbol"] for bp in bad_performers]
        bad_symbols = [bp["symbol"] for bp in bad_performers if bp.get("avoid_until", current_date) > current_date]

        logging.info(f"Filtering out {len(bad_symbols_no_date)} bad performers no date.")
        logging.info(f"Filtering out {len(bad_symbols)} bad performers.")
        df_filtered = df[~df['Symbol'].isin(bad_symbols)]
        return df_filtered
    except Exception as e:
        logging.error(f"An error occurred while filtering bad performers: {e}")
        return df


def classify_market_cap(df):
    df_top = df.nlargest(5, 'MarketCapitalization')
    df_top['MarketCapType'] = 'High'
    df_bottom = df.nsmallest(5, 'MarketCapitalization')
    df_bottom['MarketCapType'] = 'Low'
    return pd.concat([df_top, df_bottom])



def test_db_connection():
    try:
        count = company_overviews_collection.count_documents({})
        logging.info(f"Test query found {count} documents in 'company_overviews'.")
    except Exception as e:
        logging.error(f"Test query failed: {e}")

def store_selected_pairs(df):
    df['date_added'] = datetime.now().strftime('%Y-%m-%d')
    for _, row in df.iterrows():
        filter_ = {'symbol': row['Symbol'], 'date_added': row['date_added']}
        update = {'$set': row.to_dict()}
        result = selected_pairs_collection.update_one(filter_, update, upsert=True)
        if result.upserted_id:
            logging.info(f"Inserted new document for symbol: {row['Symbol']}")
        elif result.modified_count > 0:
            logging.info(f"Updated document for symbol: {row['Symbol']}")
        else:
            logging.info(f"No changes for symbol: {row['Symbol']}")
    logging.info("Selected pairs updated in MongoDB 'selected_pairs' collection.")

def fetch_financial_data(symbol):
    try:
        symbol, current_price, sma_value = get_current_price_and_sma(symbol)
        symbol, rsi_value = get_rsi(symbol)
        print(f"fetch_financial_data - {symbol}: current_price={current_price}, sma_value={sma_value}, rsi_value={rsi_value}")
        return symbol, current_price, sma_value, rsi_value
    except Exception as e:
        logging.error(f"Error fetching financial data for {symbol}: {e}")
        return symbol, np.nan, np.nan, np.nan


def main():
    test_db_connection()
    logging.info("Starting script...")

    # Clear out all records from the selected_pairs collection
    selected_pairs_collection.delete_many({})
    logging.info("Cleared all records from 'selected_pairs' collection.")

    df = fetch_company_overviews()
    if df.empty:
        logging.info("Exiting script due to no data.")
        return

    df_filtered = df[
        (df["ProfitMargin"] > -10) &
        (df["PERatio"] > 0) &
        (df["ReturnOnEquityTTM"] > -20) &
        (df["EVToEBITDA"].between(0, 50)) &
        (df["QuarterlyEarningsGrowthYOY"] > 0)
    ].copy()  # Avoiding SettingWithCopyWarning by creating a copy

    df_filtered["MarketCapitalization"] = pd.to_numeric(df_filtered["MarketCapitalization"], errors='coerce')

    # Apply bad performers filter
    df_filtered = filter_bad_performers(df_filtered)

    grouped = df_filtered.groupby('Sector', group_keys=False)
    selected_pairs_top_bottom = grouped.apply(classify_market_cap).reset_index(drop=True)

    store_selected_pairs(selected_pairs_top_bottom)

    symbols = selected_pairs_top_bottom['Symbol'].tolist()
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_symbol = {executor.submit(fetch_financial_data, symbol): symbol for symbol in symbols}

    for future in as_completed(future_to_symbol):
        symbol = future_to_symbol[future]
        try:
            data = future.result()
            logging.info(f"Fetched financial data for {symbol}: {data}")
        except Exception as exc:
            logging.error(f"Error fetching financial data for {symbol}: {exc}")

if __name__ == "__main__":
    main()
