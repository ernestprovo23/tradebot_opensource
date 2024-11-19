import sys
import os
import joblib
import logging
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

# Adjust system path to include the directory where the teams_communicator module is located
script_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(script_dir, '..'))

from teams_communicator import TeamsCommunicator

# Load environment variables
load_dotenv()
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Setup MongoDB connection
client = MongoClient(MONGO_DB_CONN_STRING)
db_ml = client['machinelearning']
db_predictions = client['predictions']

# Setup logging
logging.basicConfig(filename='prediction_logfile.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def get_latest_file(directory, prefix, symbol):
    """Get the latest file with a specific prefix based on timestamp."""
    print(f"Looking for latest file with prefix '{prefix}' in directory '{directory}'")
    logging.info(f"Looking for latest file with prefix '{prefix}' in directory '{directory}'")
    try:
        symbol_dir = os.path.join(directory, symbol)
        if os.path.exists(symbol_dir):
            directory = symbol_dir
        files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".pkl")]
        if not files:
            logging.warning(f"No files with prefix '{prefix}' found in directory: {directory}")
            print(f"No files with prefix '{prefix}' found in directory: {directory}")
            return None
        latest_file = sorted(files, key=lambda x: x.split('_')[-1].split('.')[0], reverse=True)[0]
        print(f"Found latest file: {latest_file}")
        logging.info(f"Found latest file: {latest_file}")
        return os.path.join(directory, latest_file)
    except Exception as e:
        logging.error(f"Failed to find the latest file in {directory}: {str(e)}")
        print(f"Failed to find the latest file in {directory}: {str(e)}")
        return None

def load_and_verify_scalers(feature_scaler_path, target_scaler_path, feature_data, target_data):
    """Load and verify the feature and target scalers, fitting them if necessary."""
    print(f"Loading scalers from {feature_scaler_path} and {target_scaler_path}")
    logging.info(f"Loading scalers from {feature_scaler_path} and {target_scaler_path}")
    try:
        feature_scaler = joblib.load(feature_scaler_path)
        target_scaler = joblib.load(target_scaler_path)

        try:
            feature_scaler.transform(feature_data)
        except NotFittedError:
            print("Feature scaler not fitted, fitting now...")
            logging.info("Feature scaler not fitted, fitting now...")
            feature_scaler.fit(feature_data)

        try:
            target_scaler.inverse_transform(target_data)
        except NotFittedError:
            print("Target scaler not fitted, fitting now...")
            logging.info("Target scaler not fitted, fitting now...")
            target_scaler.fit(target_data)

        return feature_scaler, target_scaler
    except Exception as e:
        logging.error(f"Error loading scaler files: {str(e)}")
        print(f"Error loading scaler files: {str(e)}")
        return None, None

def predict_new_prices(symbol, model_dir, new_data):
    """Predict prices using the trained model and new input data."""
    print(f"Predicting new prices for {symbol}")
    logging.info(f"Predicting new prices for {symbol}")

    model_path = get_latest_file(model_dir, f'model_{symbol}', symbol)
    feature_scaler_path = get_latest_file(model_dir, f'feature_scaler_{symbol}', symbol)
    target_scaler_path = get_latest_file(model_dir, f'target_scaler_{symbol}', symbol)

    if not model_path or not feature_scaler_path or not target_scaler_path:
        logging.error(f"Required files are missing for symbol {symbol} in directory {model_dir}")
        print(f"Required files are missing for symbol {symbol} in directory {model_dir}")
        return

    print(f"Loading model from {model_path}")
    logging.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    # Define the exact features used in the model
    feature_columns = [
        'balance_sheet_current_ratio',
        'balance_sheet_debt_to_equity_ratio',
        'balance_sheet_long_term_debt_to_equity_ratio',
        'balance_sheet_quick_ratio',
        'cash_flow_capitalExpenditures',
        'cash_flow_free_cash_flow',
        'cash_flow_operatingCashflow',
        'inflation_value',
        'real_gdp_value',
        'retail_sales_value',
        'unemployment_rate'
    ]

    # Select only the features used in the model
    features = new_data[feature_columns]
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)

    print(f"Selected features for prediction: {features.columns}")
    logging.info(f"Selected features for prediction: {features.columns}")
    logging.info(f"Number of features: {features.shape[1]}")
    print(f"Number of features: {features.shape[1]}")

    target_data = new_data[['close']].apply(pd.to_numeric, errors='coerce').fillna(0)

    feature_scaler, target_scaler = load_and_verify_scalers(feature_scaler_path, target_scaler_path, features, target_data)

    if feature_scaler is None or target_scaler is None:
        logging.error(f"Failed to load or fit scalers for {symbol}")
        print(f"Failed to load or fit scalers for {symbol}")
        return

    try:
        features_scaled = feature_scaler.transform(features)
    except ValueError as e:
        logging.error(f"Error scaling features: {str(e)}")
        print(f"Error scaling features: {str(e)}")
        logging.error(f"Feature scaler n_features_in_: {feature_scaler.n_features_in_}")
        print(f"Feature scaler n_features_in_: {feature_scaler.n_features_in_}")
        return

    print("Making predictions...")
    logging.info("Making predictions...")
    predicted_scaled = model.predict(features_scaled)
    predicted_price = target_scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()[-1]

    max_date = new_data['date'].max()
    prediction_date = pd.to_datetime(max_date) + timedelta(days=90)

    print(f"Max Date from Data: {max_date}, Prediction Date: {prediction_date}")
    logging.info(f"Max Date from Data: {max_date}, Prediction Date: {prediction_date}")

    save_prediction(symbol, prediction_date, predicted_price)

def save_prediction(symbol, prediction_date, predicted_price):
    """Save prediction results to MongoDB with metadata if it doesn't already exist."""
    print(f"Saving prediction for {symbol} on {prediction_date}")
    logging.info(f"Saving prediction for {symbol} on {prediction_date}")
    current_month_year = datetime.now().strftime("%B_%Y")
    collection = db_predictions[current_month_year]

    predicted_price = float(predicted_price)

    existing_record = collection.find_one({
        'symbol': symbol,
        'prediction_date': prediction_date,
        'predicted_price': predicted_price
    })

    if existing_record:
        logging.info(f"Prediction for {symbol} on {prediction_date.strftime('%Y-%m-%d')} with price {predicted_price} already exists. Skipping insertion.")
        print(f"Prediction for {symbol} on {prediction_date.strftime('%Y-%m-%d')} with price {predicted_price} already exists. Skipping insertion.")
    else:
        data_entry = {
            'symbol': symbol,
            'prediction_date': prediction_date,
            'predicted_price': predicted_price,
            'entry_date': datetime.now()
        }
        collection.insert_one(data_entry)
        logging.info(f"Inserted prediction for {symbol} on {prediction_date.strftime('%Y-%m-%d')}: {predicted_price}")
        print(f"Inserted prediction for {symbol} on {prediction_date.strftime('%Y-%m-%d')}: {predicted_price}")

def process_predictions(model_dir, symbol=None):
    """Process predictions for each symbol found in the model directory."""
    print(f"Processing predictions with model directory '{model_dir}'")
    logging.info(f"Processing predictions with model directory '{model_dir}'")
    if symbol:
        symbols = [symbol]
    else:
        # List directories inside the model_dir to find the symbols
        symbols = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]

    print(f"Found symbols: {symbols}")
    logging.info(f"Found symbols: {symbols}")

    for sym in symbols:
        collection_name = f"{sym}_processed_data"
        if collection_name in db_ml.list_collection_names():
            print(f"Processing symbol: {sym}")
            logging.info(f"Processing symbol: {sym}")
            collection = db_ml[collection_name]
            documents = collection.find({})
            all_data = pd.DataFrame(list(documents))
            if not all_data.empty:
                print(f"Data loaded for symbol {sym}, proceeding with predictions.")
                logging.info(f"Data loaded for symbol {sym}, proceeding with predictions.")
                all_data['date'] = pd.to_datetime(all_data['date'])
                predict_new_prices(sym, model_dir, all_data)
            else:
                logging.info(f"No data found for {sym}")
                print(f"No data found for {sym}")
        else:
            logging.info(f"No collection found for {sym}")
            print(f"No collection found for {sym}")

if __name__ == "__main__":
    model_dir = "/home/jaguar/DSEalgo_v2/modelfiles/"
    if not os.path.exists(model_dir):
        logging.error(f"Model directory {model_dir} does not exist.")
        print(f"Model directory {model_dir} does not exist.")
    else:
        communicator = TeamsCommunicator("ml_database", "prediction_logs")
        logging.info("Processing symbols")
        print("Processing symbols")


        # Check for command-line arguments
        if len(sys.argv) > 1:
            symbol = sys.argv[1]
            print(f"Symbol provided via command line: {symbol}")
            logging.info(f"Symbol provided via command line: {symbol}")
            process_predictions(model_dir, symbol)
        else:
            print("No symbol provided, processing all symbols")
            logging.info("No symbol provided, processing all symbols")
            process_predictions(model_dir)
