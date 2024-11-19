import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import os
import numpy as np
from datetime import datetime

# Load environment variables
load_dotenv()
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Setup MongoDB connection
client = MongoClient(MONGO_DB_CONN_STRING)
db_silvertables = client['machinelearning']
db_ml_ready = client['cleaneddata_ml']


# Function to clean and scale data
def clean_and_scale_data(df):
    # Remove the '_id' column or any other non-numeric columns
    df = df.drop(columns=[column for column in ['_id'] if column in df.columns])

    # Drop columns where all values are NaN
    df.dropna(axis=1, how='all', inplace=True)

    # Convert all columns that should be numeric (avoid setting copy warning)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    df.loc[:, numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Fill NaNs for numeric columns only, handling properly to avoid SettingWithCopyWarning
    for column in numeric_columns:
        df.loc[:, column] = df[column].fillna(df[column].mean())

    # Standardizing numeric columns, ensuring no NaN or infinite values are present
    if numeric_columns:
        scaler = StandardScaler()
        df.loc[:, numeric_columns] = scaler.fit_transform(df[numeric_columns].replace([np.inf, -np.inf], np.nan).dropna())

    return df



def main():
    print("Starting data cleaning and scaling...")
    try:
        collection_names = db_silvertables.list_collection_names()
        print(f"Collections in silvertables database: {collection_names}")

        for index, collection_name in enumerate(collection_names):
            if "processed_data" in collection_name:
                print(f"Processing collection: {collection_name}")
                collection = db_silvertables[collection_name]
                data_count = collection.count_documents({})
                print(f"Number of documents in {collection_name}: {data_count}")

                if data_count > 0:
                    data = pd.DataFrame(list(collection.find()))
                    print(f"Loaded {len(data)} documents into a Pandas DataFrame.")

                    # Clean and scale data
                    cleaned_data = clean_and_scale_data(data)
                    cleaned_data['datetime_imported'] = datetime.now()  # Adding datetime stamp

                    # Handling existing data: Drop the existing collection and insert anew
                    db_ml_ready[collection_name].drop()  # Drop existing collection if it exists
                    if not cleaned_data.empty:
                        cleaned_collection = db_ml_ready[collection_name]
                        cleaned_collection.insert_many(cleaned_data.to_dict('records'))
                        print(f"Inserted cleaned data into {collection_name} in cleaneddata_ml database.")
                    else:
                        print(f"No data to insert after cleaning for {collection_name}.")
                else:
                    print(f"{collection_name} is empty.")

                # Uncomment the following line to limit processing to the first matching collection during development
                # if index == 0: break

    except Exception as e:
        print(f"An error occurred: {e}")

    print("Data cleaning and scaling completed.")


if __name__ == "__main__":
    print("Executing data cleaning and scaling script...")
    main()
