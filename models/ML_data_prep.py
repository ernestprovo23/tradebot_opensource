import pandas as pd
import numpy as np
from pymongo import MongoClient, UpdateOne
from datetime import datetime, timezone
from dotenv import load_dotenv
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from functools import partial

# Set up logging
logging.basicConfig(
    filename='etl_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


@dataclass
class DatabaseCollections:
    """Collection names for MongoDB databases"""
    aggregated_stock_data: str = 'aggregated_stock_data'
    sentiment_data: str = 'ML_sentiment_data'
    balance_sheet: str = 'balance_sheet_ML'
    cash_flow: str = 'cash_flow_ML'
    raw_balance_sheet: str = 'balance_sheet'
    inflation: str = 'inflation'
    real_gdp: str = 'real_gdp'
    unemployment: str = 'unemployment'
    retail_sales: str = 'retail_sales'
    bad_performers: str = 'bad_performers'


class OptimizedETLPipeline:
    def __init__(self, mongo_conn_string: str):
        """Initialize ETL pipeline with MongoDB connection"""
        self.client = MongoClient(mongo_conn_string)
        self.collections = DatabaseCollections()

        # Initialize database connections
        self.db_stock = self.client['stock_data']
        self.db_silver = self.client['silvertables']
        self.db_ml = self.client['machinelearning']
        self.db_economic = self.client['economic_data']
        self.db_trading = self.client['trading_db']

        # Create indexes for better performance
        self._create_indexes()

    def _create_indexes(self):
        """Create necessary indexes for improved query performance"""
        try:
            # Create compound indexes for better query performance
            self.db_stock[self.collections.aggregated_stock_data].create_index(
                [('symbol', 1), ('timestamp', -1)]
            )
            self.db_silver[self.collections.sentiment_data].create_index(
                [('symbol', 1), ('datetime_processed', -1)]
            )
            logging.info("Successfully created indexes")
        except Exception as e:
            logging.error(f"Error creating indexes: {e}")

    def _prepare_dataframe(self, df: pd.DataFrame, sort_index: bool = True) -> pd.DataFrame:
        """Helper method to prepare and clean dataframe"""
        if df.empty:
            return df

        # Convert numeric columns
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception:
                continue

        # Sort index if requested
        if sort_index and isinstance(df.index, pd.DatetimeIndex):
            df.sort_index(inplace=True)

        return df

    def fetch_time_series(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch and process time series data for a symbol"""
        try:
            cursor = self.db_stock[self.collections.aggregated_stock_data].find(
                {'symbol': symbol},
                {'_id': 0, 'symbol': 1, 'timestamp': 1, 'close_price': 1}
            ).sort('timestamp', 1)  # Sort at database level

            df = pd.DataFrame(list(cursor))
            if df.empty:
                logging.warning(f"No time series data found for {symbol}")
                return None

            df['date'] = pd.to_datetime(df['timestamp'])
            df['close'] = pd.to_numeric(df['close_price'], errors='coerce')
            df = df.set_index('date')

            return self._prepare_dataframe(df[['close']])

        except Exception as e:
            logging.error(f"Error fetching time series for {symbol}: {e}")
            return None

    def _safe_merge(self, left_df: pd.DataFrame, right_df: pd.DataFrame,
                    how: str = 'left', sort: bool = True) -> pd.DataFrame:
        """Safely merge two dataframes with proper sorting"""
        if left_df.empty or right_df.empty:
            return left_df

        # Ensure both dataframes have datetime index
        for df in [left_df, right_df]:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

        # Sort both dataframes
        if sort:
            left_df.sort_index(inplace=True)
            right_df.sort_index(inplace=True)

        return pd.merge(
            left_df, right_df,
            left_index=True, right_index=True,
            how=how
        )

    def integrate_sentiment(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Integrate sentiment data with proper error handling and sorting"""
        try:
            if df is None or df.empty:
                return df

            doc = self.db_silver[self.collections.sentiment_data].find_one(
                {"symbol": symbol, "version": 1},
                sort=[("datetime_processed", -1)]
            )

            if doc and 'cleaned_data' in doc:
                sentiment_df = pd.DataFrame([doc['cleaned_data']])
                sentiment_df['date'] = pd.to_datetime(doc['datetime_processed'])
                sentiment_df.set_index('date', inplace=True)

                # Prepare sentiment columns
                sentiment_cols = ['overall_sentiment', 'ticker_sentiment']
                sentiment_df = sentiment_df[sentiment_cols]
                sentiment_df.columns = [f"sentiment_{col}" for col in sentiment_df.columns]

                # Forward fill sentiment data
                sentiment_df = sentiment_df.reindex(df.index, method='ffill')

                return self._safe_merge(df, sentiment_df)

            return df

        except Exception as e:
            logging.error(f"Error integrating sentiment for {symbol}: {e}")
            return df

    def integrate_financial_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Enhanced financial data integration with proper data validation and conversion"""
        try:
            if df is None or df.empty:
                return df

            # Process balance sheet data with explicit data type conversion
            balance_sheet = pd.DataFrame(list(self.db_stock[self.collections.balance_sheet].find(
                {'symbol': symbol},
                {'_id': 0, 'fiscalDateEnding': 1, 'current_ratio': 1, 'quick_ratio': 1, 'debt_to_equity_ratio': 1}
            )))

            if not balance_sheet.empty:
                # Convert and clean balance sheet data
                balance_sheet['fiscalDateEnding'] = pd.to_datetime(balance_sheet['fiscalDateEnding'])
                numeric_cols = ['current_ratio', 'quick_ratio', 'debt_to_equity_ratio']
                for col in numeric_cols:
                    balance_sheet[col] = pd.to_numeric(balance_sheet[col], errors='coerce')

                balance_sheet.set_index('fiscalDateEnding', inplace=True)
                balance_sheet.sort_index(inplace=True)

                # Rename columns
                balance_sheet.columns = [f"balance_sheet_{col}" for col in balance_sheet.columns]

                # Debug logging
                logging.info(f"Balance sheet data for {symbol}: {len(balance_sheet)} rows")

                df = pd.merge_asof(
                    df,
                    balance_sheet,
                    left_index=True,
                    right_index=True,
                    direction='backward',
                    tolerance=pd.Timedelta('30D')  # Allow matching within 30 days
                )

            # Process cash flow data with explicit data type conversion
            cash_flow = pd.DataFrame(list(self.db_stock[self.collections.cash_flow].find(
                {'symbol': symbol},
                {'_id': 0, 'fiscalDateEnding': 1, 'operatingCashflow': 1, 'free_cash_flow': 1}
            )))

            if not cash_flow.empty:
                # Convert and clean cash flow data
                cash_flow['fiscalDateEnding'] = pd.to_datetime(cash_flow['fiscalDateEnding'])
                numeric_cols = ['operatingCashflow', 'free_cash_flow']
                for col in numeric_cols:
                    cash_flow[col] = pd.to_numeric(cash_flow[col], errors='coerce')

                cash_flow.set_index('fiscalDateEnding', inplace=True)
                cash_flow.sort_index(inplace=True)

                # Rename columns
                cash_flow.columns = [f"cash_flow_{col}" for col in cash_flow.columns]

                # Debug logging
                logging.info(f"Cash flow data for {symbol}: {len(cash_flow)} rows")

                df = pd.merge_asof(
                    df,
                    cash_flow,
                    left_index=True,
                    right_index=True,
                    direction='backward',
                    tolerance=pd.Timedelta('30D')  # Allow matching within 30 days
                )

            # Verify data integration
            financial_cols = [col for col in df.columns if 'balance_sheet_' in col or 'cash_flow_' in col]
            for col in financial_cols:
                non_zero = (df[col] != 0).sum()
                logging.info(f"{symbol} - {col}: {non_zero} non-zero values")

            return df

        except Exception as e:
            logging.error(f"Error integrating financial data for {symbol}: {e}")
            return df

    def integrate_economic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced economic data integration with proper data validation and conversion"""
        try:
            if df is None or df.empty:
                return df

            economic_collections = [
                (self.collections.inflation, 'inflation'),
                (self.collections.real_gdp, 'gdp'),
                (self.collections.unemployment, 'unemployment'),
                (self.collections.retail_sales, 'retail')
            ]

            for collection_name, indicator_name in economic_collections:
                economic_df = pd.DataFrame(list(self.db_economic[collection_name].find(
                    {}, {'_id': 0, 'date': 1, 'value': 1}
                )))

                if not economic_df.empty:
                    # Convert and clean economic data
                    economic_df['date'] = pd.to_datetime(economic_df['date'])
                    economic_df['value'] = pd.to_numeric(economic_df['value'], errors='coerce')

                    economic_df.set_index('date', inplace=True)
                    economic_df.sort_index(inplace=True)

                    # Rename column
                    economic_df.columns = [f"economic_{indicator_name}"]

                    # Debug logging
                    logging.info(f"Economic {indicator_name} data: {len(economic_df)} rows")

                    # Merge with tolerance for economic data
                    df = pd.merge_asof(
                        df,
                        economic_df,
                        left_index=True,
                        right_index=True,
                        direction='backward',
                        tolerance=pd.Timedelta('30D')  # Allow matching within 30 days
                    )

                    # Verify data integration
                    col_name = f"economic_{indicator_name}"
                    non_zero = (df[col_name] != 0).sum()
                    logging.info(f"{col_name}: {non_zero} non-zero values")

            return df

        except Exception as e:
            logging.error(f"Error integrating economic data: {e}")
            return df

    def process_symbol(self, symbol: str) -> None:
        """Enhanced symbol processing with data validation"""
        try:
            # Fetch and prepare time series data
            df = self.fetch_time_series(symbol)
            if df is None:
                logging.warning(f"No time series data available for {symbol}")
                return

            # Log initial data state
            logging.info(f"Initial data for {symbol}: {len(df)} rows")

            # Integrate different data sources with validation
            df = self.integrate_sentiment(df, symbol)
            logging.info(f"After sentiment integration for {symbol}: {df.columns.tolist()}")

            df = self.integrate_financial_data(df, symbol)
            logging.info(f"After financial integration for {symbol}: {df.columns.tolist()}")

            df = self.integrate_economic_data(df)
            logging.info(f"After economic integration for {symbol}: {df.columns.tolist()}")

            # Verify data before final preparation
            before_fillna = df.isnull().sum()
            logging.info(f"Null values before fillna for {symbol}:\n{before_fillna}")

            # Final data preparation
            df = df.infer_objects(copy=False)
            df = df.ffill().fillna(0)

            # Verify data after preparation
            after_fillna = df.isnull().sum()
            logging.info(f"Null values after fillna for {symbol}:\n{after_fillna}")

            # Store processed data
            collection_name = f"{symbol}_processed_data"
            collection = self.db_ml[collection_name]

            # Clear existing data
            collection.delete_many({})

            # Prepare records for storage
            df.reset_index(inplace=True)
            records = df.to_dict('records')

            if records:
                # Add data validation before storage
                validated_records = []
                for record in records:
                    # Ensure all numeric fields are float
                    for key, value in record.items():
                        if key not in ['symbol', 'date', 'date_imported']:
                            try:
                                record[key] = float(value)
                            except (TypeError, ValueError):
                                record[key] = 0.0

                    validated_records.append(record)

                bulk_ops = [
                    UpdateOne(
                        {'symbol': symbol, 'date': record['date']},
                        {'$set': {
                            **record,
                            'symbol': symbol,
                            'date_imported': datetime.now(timezone.utc)
                        }},
                        upsert=True
                    )
                    for record in validated_records
                ]

                collection.bulk_write(bulk_ops)
                logging.info(f"Successfully processed and stored {len(validated_records)} records for {symbol}")
            else:
                logging.warning(f"No records to store for {symbol}")

        except Exception as e:
            logging.error(f"Error processing symbol {symbol}: {e}")

    def run_pipeline(self, testing_limit: Optional[int] = None) -> None:
        """Run the complete ETL pipeline with proper error handling"""
        try:
            # Get valid symbols with error handling
            valid_symbols = set(self.db_stock[self.collections.raw_balance_sheet].distinct('symbol'))
            trading_symbols = set(self.db_stock[self.collections.aggregated_stock_data].distinct('symbol'))
            bad_performers = set(self.db_trading[self.collections.bad_performers].distinct('symbol'))

            symbols_to_process = list((valid_symbols & trading_symbols) - bad_performers)

            if testing_limit:
                symbols_to_process = symbols_to_process[:testing_limit]

            logging.info(f"Starting to process {len(symbols_to_process)} symbols")

            # Process symbols in parallel with proper error handling
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(self.process_symbol, symbols_to_process))

            logging.info(f"Pipeline completed. Processed {len(symbols_to_process)} symbols.")

        except Exception as e:
            logging.error(f"Pipeline error: {e}")
            raise


if __name__ == "__main__":
    try:
        # Load environment variables
        load_dotenv()
        mongo_conn = os.getenv('MONGO_DB_CONN_STRING')

        if not mongo_conn:
            raise ValueError("MongoDB connection string not found in environment variables")

        # Initialize and run pipeline
        pipeline = OptimizedETLPipeline(mongo_conn)
        pipeline.run_pipeline(testing_limit=None)  # Set testing_limit for testing

    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        raise