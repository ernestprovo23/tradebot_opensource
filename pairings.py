import pandas as pd
from azure.storage.blob import BlobServiceClient
from s3connector import azure_connection_string
from io import StringIO

# Connect to Azure Blob Storage
blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)

# Define the container and file name
container_name = "historic"
file_name = "company_overviews.csv"

try:
    # Get the blob client
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(file_name)

    # Download the blob data as text
    blob_data = blob_client.download_blob().readall().decode("utf-8")

    # Load the CSV data into a DataFrame
    df = pd.read_csv(StringIO(blob_data))

    # Convert columns to appropriate data types
    numeric_cols = ["MarketCapitalization", "PERatio", "DividendYield", "RevenuePerShareTTM", "ProfitMargin"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Filter by market capitalization
    min_market_cap = 1_000_000_000  # Minimum market capitalization in dollars
    max_market_cap = 10_000_000_000  # Maximum market capitalization in dollars
    df = df[(df["MarketCapitalization"] >= min_market_cap) & (df["MarketCapitalization"] <= max_market_cap)]

    # Calculate relevant ratios
    df["PriceEarningsRatio"] = df["PERatio"]
    df["DividendYieldPercentage"] = df["DividendYield"]
    df["RevenuePerShare"] = df["RevenuePerShareTTM"]
    df["ProfitMarginPercentage"] = df["ProfitMargin"]

    # Apply additional filters and criteria
    df = df[df["ProfitMarginPercentage"] > 0]  # Filter out assets with negative profit margin
    df = df[df["PriceEarningsRatio"] < 20]  # Filter out assets with high P/E ratio

    # Sort and rank the DataFrame based on selected criteria
    df = df.sort_values("MarketCapitalization", ascending=False)

    # Output the selected pairs
    selected_pairs = df[["Symbol", "Sector", "Industry", "MarketCapitalization", "PriceEarningsRatio",
                         "DividendYieldPercentage", "RevenuePerShare", "ProfitMarginPercentage"]]
    selected_pairs.to_csv("selected_pairs.csv", index=False)
    print("Selected pairs saved to 'selected_pairs.csv'")

except Exception as e:
    print(f"An error occurred: {e}")
