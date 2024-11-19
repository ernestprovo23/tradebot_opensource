import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Connect to MongoDB
client = MongoClient(MONGO_DB_CONN_STRING)
db = client['machinelearning']


def get_financial_data(collection_name, start_date, end_date):
    query = {
        "$or": [
            {"cash_flow.annualReports.fiscalDateEnding": {"$gte": start_date.isoformat(),
                                                          "$lte": end_date.isoformat()}},
            {"cash_flow.quarterlyReports.fiscalDateEnding": {"$gte": start_date.isoformat(),
                                                             "$lte": end_date.isoformat()}},
            {"balance_sheet.annualReports.fiscalDateEnding": {"$gte": start_date.isoformat(),
                                                              "$lte": end_date.isoformat()}},
            {"balance_sheet.quarterlyReports.fiscalDateEnding": {"$gte": start_date.isoformat(),
                                                                 "$lte": end_date.isoformat()}}
        ]
    }
    return list(db[collection_name].find(query))


def process_financial_data(data, report_type):
    all_reports = []
    for doc in data:
        financial_data = doc[report_type]
        symbol = financial_data['symbol']

        for report_category in ['annualReports', 'quarterlyReports']:
            for report in financial_data[report_category]:
                report['reportType'] = 'annual' if report_category == 'annualReports' else 'quarterly'
                report['symbol'] = symbol
                all_reports.append(report)

    df = pd.DataFrame(all_reports)
    df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])

    # Convert string values to numeric, replacing 'None' with NaN
    for col in df.columns:
        if col not in ['symbol', 'reportType', 'fiscalDateEnding', 'reportedCurrency']:
            df[col] = pd.to_numeric(df[col].replace('None', np.nan), errors='coerce')

    return df


def calculate_financial_ratios(df):
    df['current_ratio'] = df['totalCurrentAssets'] / df['totalCurrentLiabilities']
    df['quick_ratio'] = (df['totalCurrentAssets'] - df['inventory']) / df['totalCurrentLiabilities']
    df['cash_ratio'] = df['cashAndCashEquivalentsAtCarryingValue'] / df['totalCurrentLiabilities']
    df['debt_to_equity_ratio'] = df['totalLiabilities'] / df['totalShareholderEquity']
    df['return_on_equity'] = df['netIncome'] / df['totalShareholderEquity']
    return df


def main():
    # Set date range for data retrieval (last 10 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3650)

    # Retrieve data from MongoDB
    cash_flow_data = get_financial_data('cash_flow', start_date, end_date)
    balance_sheet_data = get_financial_data('balance_sheets', start_date, end_date)

    # Process the data
    cash_flow_df = process_financial_data(cash_flow_data, 'cash_flow')
    balance_sheet_df = process_financial_data(balance_sheet_data, 'balance_sheet')

    # Merge the dataframes
    merged_df = pd.merge(cash_flow_df, balance_sheet_df,
                         on=['symbol', 'fiscalDateEnding', 'reportType'],
                         suffixes=('_cf', '_bs'))

    # Calculate financial ratios
    merged_df = calculate_financial_ratios(merged_df)

    # Basic analysis
    print("Data shape:", merged_df.shape)
    print("\nColumns:", merged_df.columns.tolist())
    print("\nSample data:")
    print(merged_df.head())

    print("\nFinancial Ratios Summary:")
    ratios = ['current_ratio', 'quick_ratio', 'cash_ratio', 'debt_to_equity_ratio', 'return_on_equity']
    print(merged_df[['symbol', 'fiscalDateEnding'] + ratios].describe())

    # Group by symbol and calculate mean ratios
    mean_ratios = merged_df.groupby('symbol')[ratios].mean()
    print("\nMean Ratios by Symbol:")
    print(mean_ratios)

    # # Save to CSV
    # merged_df.to_csv('financial_data_analysis.csv', index=False)
    # print("\nData saved to 'financial_data_analysis.csv'")


if __name__ == "__main__":
    main()