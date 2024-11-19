from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') #asdf

# Load environment variables
load_dotenv()
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Establish MongoDB connection
client = MongoClient(MONGO_DB_CONN_STRING)
db_source = client['cleaneddata_ml']
db_target = client['training_data']  # New database for storing training data

def feature_engineering(data):
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)

    data['MA_diff'] = data['MA50'] - data['MA200']
    data['EMA_diff'] = data['EMA50'] - data['EMA200']
    data['RSI_scaled'] = data['RSI'] / 100  # Scale RSI to be between 0 and 1
    data['MA50_rate_change'] = data['MA50'].diff() / (data['MA50'].shift(1) + 1e-8)
    data['MA200_rate_change'] = data['MA200'].diff() / (data['MA200'].shift(1) + 1e-8)

    return data

def normalize_features(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features.fillna(0))
    return pd.DataFrame(scaled_features, columns=features.columns)

def clean_and_store_data(collection_name):
    input_collection = db_source[collection_name]
    output_collection_name = f"{collection_name.split('_')[0]}_cleaned_for_training"
    output_collection = db_target[output_collection_name]

    try:
        data = pd.DataFrame(list(input_collection.find()))

        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data = feature_engineering(data)

        features = data[['MA_diff', 'EMA_diff', 'RSI_scaled', 'MA50_rate_change', 'MA200_rate_change']]
        target = data['close_scaled']

        features = normalize_features(features)

        cleaned_data = pd.concat([features, target], axis=1)

        if output_collection_name in db_target.list_collection_names():
            db_target.drop_collection(output_collection_name)
            logging.info(f"Dropped existing collection: {output_collection_name}")

        output_collection.insert_many(cleaned_data.to_dict('records'))
        logging.info(f"Data stored in {output_collection_name}")

    except Exception as e:
        logging.error(f"Error processing {collection_name}: {e}")

if __name__ == "__main__":
    collection_names = [name for name in db_source.list_collection_names() if 'processed_data' in name]
    for collection_name in collection_names:
        logging.info(f"Processing {collection_name}...")
        clean_and_store_data(collection_name)
