import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras import optimizers
from tensorflow.keras.losses import Huber
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras import regularizers
from scikeras.wrappers import KerasRegressor
import joblib
from joblib import dump
import logging
from sys import argv
import glob
import os
import tensorflow as tf
import gc
from memory_profiler import profile
from tensorflow.keras.layers import BatchNormalization
import requests
import json
from dotenv import load_dotenv
from contextlib import contextmanager
from sklearn.preprocessing import MinMaxScaler

# Load environment variables
load_dotenv()

# Teams webhook URL
TEAMS_URL = os.getenv("TEAMS_WEBHOOK_URL")


# Configure TensorFlow for memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


@contextmanager
def tf_session():
    session = tf.compat.v1.Session()
    try:
        yield session
    finally:
        session.close()


def clear_session():
    K.clear_session()
    tf.compat.v1.reset_default_graph()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Is GPU available: ", len(tf.config.list_physical_devices('GPU')) > 0)
print("Is built with CUDA: ", tf.test.is_built_with_cuda())
print("Listing physical devices:", tf.config.list_physical_devices())

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_visible_devices(gpus[0], 'GPU')

# Load environment variables
load_dotenv()
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Setup MongoDB connection
client = MongoClient(MONGO_DB_CONN_STRING)
db = client['machinelearning']


def is_recently_modified(symbol_dir, hours=4):
    """
    Check if the symbol directory has been modified within the last specified hours.

    Args:
    symbol_dir (str): Path to the symbol directory
    hours (int): Number of hours to consider as recent

    Returns:
    bool: True if the directory has been modified within the specified hours, False otherwise
    """
    if not os.path.exists(symbol_dir):
        return False

    mod_time = datetime.fromtimestamp(os.path.getmtime(symbol_dir))
    current_time = datetime.now()
    time_difference = current_time - mod_time

    return time_difference < timedelta(hours=hours)


def estimate_runtime(n_candidates, n_folds, avg_fit_time):
    total_fits = n_candidates * n_folds
    estimated_time = total_fits * avg_fit_time
    return estimated_time


def load_data(collection_name, batch_size=5000):
    collection = db[collection_name]
    total_documents = collection.count_documents({})
    logging.info(f"Total documents in {collection_name}: {total_documents}")

    if total_documents == 0:
        logging.error(f"No documents found in collection {collection_name}")
        return None, 0

    def data_generator():
        while True:
            for i in range(0, total_documents, batch_size):
                try:
                    batch = list(collection.find().skip(i).limit(batch_size))
                    if not batch:
                        logging.warning(f"Empty batch at index {i}")
                        continue

                    batch_df = pd.DataFrame(batch)

                    # Preprocess the data to handle NaN and data types
                    batch_df = preprocess_data(batch_df)

                    # Check for NaNs after preprocessing
                    if batch_df.isnull().values.any():
                        logging.error(f"NaN values detected in batch at index {i}")
                        continue  # Skip this batch

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

                    X = batch_df[feature_columns].values
                    y = batch_df['close'].values.reshape(-1, 1)

                    yield X, y
                except Exception as e:
                    logging.error(f"Error processing batch at index {i}: {str(e)}")
                    continue

    return data_generator(), total_documents


def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    ''' Wrapper function to create a LearningRateScheduler with step decay schedule. '''

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule)


# Force TensorFlow to use the GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Use only the first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Using {len(logical_gpus)} Logical GPU")
    except RuntimeError as e:
        print(e)


def build_model(input_dim, **kwargs):
    # Remove 'model__' prefix from parameter names
    cleaned_kwargs = {k.replace('model__', ''): v for k, v in kwargs.items()}

    optimizer = cleaned_kwargs.get('optimizer', 'adam')
    activation = cleaned_kwargs.get('activation', 'relu')
    learning_rate = cleaned_kwargs.get('learning_rate', 0.001)
    dropout_rate = cleaned_kwargs.get('dropout_rate', 0.2)
    l2_regularization = cleaned_kwargs.get('l2_regularization', 0.01)
    num_layers = cleaned_kwargs.get('num_layers', 5)
    units_per_layer = cleaned_kwargs.get('units_per_layer', 256)

    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for _ in range(num_layers):
        model.add(Dense(units_per_layer, kernel_regularizer=regularizers.l2(l2_regularization)))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    if optimizer == 'adam':
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_absolute_error'])

    # Log model summary instead of individual layer shapes
    model.summary(print_fn=logging.info)

    return model


def train_and_evaluate_model(model, data_gen, total_samples, X_test, y_test, feature_scaler, target_scaler, symbol):
    print("Training on:", tf.config.list_physical_devices())
    logging.info(f"X_test shape in train_and_evaluate_model: {X_test.shape}")
    logging.info(f"y_test shape in train_and_evaluate_model: {y_test.shape}")

    steps_per_epoch = min(total_samples // 128, 200)  # Limit to 100 steps per epoch
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr_scheduler = step_decay_schedule(initial_lr=0.005, decay_factor=0.5, step_size=20)

    with tf.device('/GPU:0'):
        for epoch in range(50):
            for step in range(steps_per_epoch):
                X_batch, y_batch = next(data_gen)

        history = model.fit(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=200,
            validation_data=(X_test, y_test),
            callbacks=[early_stop, lr_scheduler],
            verbose=1
        )

        logging.info(f"Training history: {history.history}")

        test_scores = model.evaluate(X_test, y_test, verbose=0)
        logging.info(f"Test set scores: {test_scores}")

        # Check for NaN values in predictions
        predictions = model.predict(X_test)
        if np.isnan(predictions).any():
            logging.error("NaN values detected in predictions")
            return None

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_model_and_scalers_with_cleanup(model, feature_scaler, target_scaler, symbol, version)

    return model


def hyperparameter_optimization(X_train, y_train, input_dim):
    if np.isnan(X_train).any() or np.isnan(y_train).any():
        logging.error(
            "NaN values found in X_train or y_train. Please clean the data before hyperparameter optimization.")
        X_train, y_train = X_train[~np.isnan(X_train).any(axis=1)], y_train[~np.isnan(y_train)]

    param_grid = {
        'model__optimizer': ['adam', 'rmsprop'],
        'model__activation': ['relu', 'elu'],
        'model__learning_rate': [0.0001, 0.001, 0.01],
        'model__dropout_rate': [0.1, 0.2],
        'model__l2_regularization': [0.0001, 0.001],
        'model__num_layers': [1, 2, 3],
        'model__units_per_layer': [32, 64, 128]
    }

    def create_model(optimizer='adam', activation='relu', learning_rate=0.000115, dropout_rate=0.255,
                     l2_regularization=0.0425, num_layers=1, units_per_layer=37):
        model = Sequential()
        model.add(Input(shape=(input_dim,)))
        for _ in range(num_layers):
            model.add(Dense(units_per_layer, activation=activation,
                            kernel_regularizer=regularizers.l2(l2_regularization)))
            model.add(Dropout(dropout_rate))
        model.add(Dense(1))

        if optimizer == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            opt = optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        model.compile(optimizer=opt, loss=Huber(), metrics=['mean_absolute_error'])
        return model

    model = KerasRegressor(
        model=create_model,
        optimizer='adam',
        activation='relu',
        learning_rate=0.000115,
        dropout_rate=0.255,
        l2_regularization=0.0425,
        num_layers=1,
        units_per_layer=37,
        verbose=0
    )

    try:
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=25,
            cv=6,
            verbose=2,
            scoring='neg_mean_absolute_error',
            n_jobs=1
        )
        random_search_result = random_search.fit(X_train, y_train)
        return random_search_result.best_params_
    except Exception as e:
        logging.error(f"Error during hyperparameter optimization: {str(e)}")
        return {}


def evaluate_and_predict(model, X_test, y_test, target_scaler):
    """Evaluate the trained model on the test set and provide scaled predictions."""
    try:
        # Check for NaNs in test data
        if np.isnan(X_test).any() or np.isnan(y_test).any():
            logging.error("NaN values found in X_test or y_test.")
            X_test = np.nan_to_num(X_test)  # Replace NaNs with 0
            y_test = np.nan_to_num(y_test)  # Replace NaNs with 0

        logging.info("Model trained. Starting evaluation and prediction.")
        metrics = model.evaluate(X_test, y_test, verbose=0)
        logging.info(f"Test loss: {metrics[0]}, Test MAE: {metrics[1]}")

        predictions = model.predict(X_test)

        # Check for NaNs in predictions
        if predictions is None or len(predictions) == 0:
            logging.error("Predictions are empty or None.")
            return None, metrics

        if np.isnan(predictions).any():
            logging.error("NaN values detected in predictions.")
            predictions = np.nan_to_num(predictions)

        if target_scaler is None:
            logging.error("Target scaler is None. Cannot rescale predictions.")
            return None, metrics

        rescaled_predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()


        logging.info(f"Sample of rescaled predictions: {rescaled_predictions[:1]}")

        return rescaled_predictions, metrics
    except Exception as e:
        logging.error(f"Error during model evaluation and preadiction: {e}")
        return None, None


def get_next_version_number(symbol, model_dir):
    """Retrieve the next version number for a given symbol by checking existing files."""
    files = os.listdir(model_dir)
    versions = [
        int(f.split('_v')[-1].split('.')[0]) for f in files
        if f.startswith(f"model_{symbol}_v") and f.endswith(".pkl")
    ]
    return max(versions, default=0) + 1


def save_scalers(feature_scaler, target_scaler, path='scalers/'):
    joblib.dump(feature_scaler, os.path.join(path, 'feature_scaler.gz'))
    joblib.dump(target_scaler, os.path.join(path, 'target_scaler.gz'))


def load_scalers(path='scalers/'):
    feature_scaler = joblib.load(os.path.join(path, 'feature_scaler.gz'))
    target_scaler = joblib.load(os.path.join(path, 'target_scaler.gz'))
    return feature_scaler, target_scaler


def store_predictions(predictions, symbol, collection, target_scaler):
    if predictions is None or len(predictions) == 0:
        logging.error(f"No predictions to store for {symbol}.")
        return

    # Rescale predictions to the original scale
    predictions_rescaled = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    # Convert numpy types to Python types
    predictions_rescaled = convert_numpy_to_python(predictions_rescaled)

    prediction_data = [{
        'symbol': symbol,
        'predicted_price': pred,
        'datetime_calculated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    } for pred in predictions_rescaled]
    collection.insert_many(prediction_data)

    # # Log a sample of stored predictions
    # log_sample_predictions(collection)


def save_model_and_scalers(model, feature_scaler, target_scaler, symbol, model_dir):
    """
    Save the model and scalers with a sequential version number.
    """
    version = get_next_version_number(symbol, model_dir)
    model_path = os.path.join(model_dir, f"model_{symbol}_v{version}.pkl")
    feature_scaler_path = os.path.join(model_dir, f"feature_scaler_{symbol}_v{version}.pkl")
    target_scaler_path = os.path.join(model_dir, f"target_scaler_{symbol}_v{version}.pkl")

    joblib.dump(model, model_path)
    joblib.dump(feature_scaler, feature_scaler_path)
    joblib.dump(target_scaler, target_scaler_path)

    logging.info(f"Model saved to {model_path}")
    logging.info(f"Feature scaler saved to {feature_scaler_path}")
    logging.info(f"Target scaler saved to {target_scaler_path}")


def create_symbol_directory(symbol, parent_dir='modelfiles'):
    """Create a separate directory for each symbol within the parent directory."""
    symbol_dir = os.path.join(parent_dir, symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    return symbol_dir


def cleanup_old_files(symbol_dir, max_files=5):
    """Remove old model files based on modification time, keeping only the most recent ones."""
    files = glob.glob(os.path.join(symbol_dir, '*'))
    files.sort(key=os.path.getmtime, reverse=True)

    if len(files) > max_files:
        old_files = files[max_files:]
        for file in old_files:
            os.remove(file)
            logging.info(f"Removed old file: {file}")


def save_model_and_scalers_with_cleanup(model, feature_scaler, target_scaler, symbol, version):
    """
    Save the model and scalers with unique filenames, versioning, and perform cleanup of old files.

    Args:
        model: The trained Keras model.
        feature_scaler: The fitted feature scaler object.
        target_scaler: The fitted target scaler object.
        symbol: The stock symbol associated with the model.
        version: The version number or timestamp of the model.
    """
    # Ensure the model files are saved in the correct directory
    model_dir = "modelfiles"
    os.makedirs(model_dir, exist_ok=True)

    # Create a subdirectory for the symbol within the correct modelfiles directory
    symbol_dir = os.path.join(model_dir, symbol)
    os.makedirs(symbol_dir, exist_ok=True)

    # Construct file paths within the correct directory
    model_path = os.path.join(symbol_dir, f"model_{symbol}_{version}.pkl")
    feature_scaler_path = os.path.join(symbol_dir, f"feature_scaler_{symbol}_{version}.pkl")
    target_scaler_path = os.path.join(symbol_dir, f"target_scaler_{symbol}_{version}.pkl")

    # Save the model and scalers using joblib
    dump(model, model_path)
    dump(feature_scaler, feature_scaler_path)
    dump(target_scaler, target_scaler_path)

    logging.info(f"Model saved to {model_path}")
    logging.info(f"Feature scaler saved to {feature_scaler_path}")
    logging.info(f"Target scaler saved to {target_scaler_path}")

    # Cleanup old files, keeping only the most recent versions
    cleanup_old_files(symbol_dir)


def preprocess_data(df):
    # Convert '_id' to string
    df['_id'] = df['_id'].astype(str)

    # Convert 'date' to datetime if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # List of numerical columns
    numerical_columns = [
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

    # Convert numerical columns to numeric, handle errors gracefully
    for col in numerical_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Check for NaNs and handle them
    if df[numerical_columns].isnull().values.any():
        logging.warning(f"NaN values found in columns: {df[numerical_columns].columns[df[numerical_columns].isnull().any()].tolist()}")

        # Option 1: Impute NaNs with the median value
        df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

        # Option 2: Drop rows with NaN values
        # df = df.dropna(subset=numerical_columns)

        logging.info("NaN values handled by imputing with median values.")

    # Ensure no NaNs remain
    if df[numerical_columns].isnull().values.any():
        logging.error("NaN values still present after handling. This could lead to issues during model training.")
        df = df.dropna(subset=numerical_columns)

    return df


def convert_numpy_to_python(data):
    """
    Convert numpy types in the data to native Python types for serialization compatibility.
    """
    if isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: convert_numpy_to_python(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_python(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_numpy_to_python(item) for item in data)
    else:
        return data


def prepare_data_for_training(df):
    # Final check for any remaining NaNs
    if df.isnull().values.any():
        logging.warning("NaN values still present after preprocessing. Dropping these rows.")
        df = df.dropna()

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

    X = df[feature_columns].values
    y = df['close'].values.reshape(-1, 1)

    return X, y


def load_test_data(collection_name, test_size=200):
    collection = db[collection_name]
    total_documents = collection.count_documents({})

    # Get the last 'test_size' documents
    test_data = list(collection.find().sort([('_id', -1)]).limit(test_size))
    test_df = pd.DataFrame(test_data)

    # Apply the same preprocessing as in load_data
    if 'date' in test_df.columns:
        test_df['date'] = pd.to_datetime(test_df['date'])
    if 'date_imported' in test_df.columns:
        test_df['date_imported'] = pd.to_datetime(test_df['date_imported'])

    for col in test_df.columns:
        if isinstance(test_df[col].iloc[0], dict):
            nested_df = pd.json_normalize(test_df[col])
            nested_df.columns = [f"{col}_{subkey}" for subkey in nested_df.columns]
            test_df = pd.concat([test_df.drop(columns=[col]), nested_df], axis=1)

    for col in test_df.columns:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

    test_df.fillna(0, inplace=True)

    # Use the same feature columns as in load_data
    # In load_data or during model training
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

    X_test = test_df[feature_columns].values
    y_test = test_df['close'].values.reshape(-1, 1)

    logging.info(f"X_test shape in load_test_data: {X_test.shape}")
    logging.info(f"y_test shape in load_test_data: {y_test.shape}")

    return X_test, y_test


def send_teams_notification(message, webhook_url):
    """Send a notification to a Microsoft Teams channel."""
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "text": message
    }
    try:
        response = requests.post(webhook_url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            logging.info("Successfully sent notification to Teams.")
        else:
            logging.error(f"Failed to send notification to Teams. Status code: {response.status_code}, Response: {response.text}")
    except Exception as e:
        logging.error(f"Exception occurred while sending notification to Teams: {str(e)}")


def get_processed_symbols(model_dir='modelfiles'):
    """
    Check the modelfiles directory to identify which symbols have already been processed.

    Args:
    model_dir (str): The directory where model files are stored.

    Returns:
    set: A set of symbols that have already been processed.
    """
    processed_symbols = set()
    if os.path.exists(model_dir):
        for symbol_dir in os.listdir(model_dir):
            symbol_path = os.path.join(model_dir, symbol_dir)
            if os.path.isdir(symbol_path) and any(file.startswith('model_') for file in os.listdir(symbol_path)):
                processed_symbols.add(symbol_dir)
    return processed_symbols


def get_symbols_to_process(all_symbols, processed_symbols, batch_size=25):
    """
    Get the next batch of symbols to process, excluding those already processed.

    Args:
    all_symbols (list): List of all available symbols.
    processed_symbols (set): Set of symbols already processed.
    batch_size (int): Number of symbols to process in this batch.

    Returns:
    list: List of symbols to process in this batch.
    """
    symbols_to_process = [symbol for symbol in all_symbols if symbol not in processed_symbols]
    return symbols_to_process[:batch_size]

## main execution block
@profile
def main(symbols=None):
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        logging.info(f"GPU is available and will be used for training: {physical_devices}")
    else:
        logging.warning("No GPU available. Training will proceed on CPU.")

    try:
        model_outputs = db['model_outputs']

        # Get all available collections if no symbols are provided
        if not symbols:
            collection_names = [name for name in db.list_collection_names() if 'processed_data' in name]
            symbols = [name.split('_')[0] for name in collection_names]

        for symbol in symbols:
            collection_name = f"{symbol}_processed_data"
            logging.info(f"Processing {collection_name}...")

            symbol_dir = os.path.join("modelfiles", symbol)

            with tf_session():
                try:
                    model_dir = create_symbol_directory(symbol)
                    feature_scaler = StandardScaler()
                    target_scaler = MinMaxScaler()

                    data_gen_result = load_data(collection_name)
                    if data_gen_result is None:
                        logging.error(f"Failed to load data for {collection_name}")
                        continue

                    X_gen, total_samples = data_gen_result
                    if total_samples == 0:
                        logging.error(f"No samples found in {collection_name}")
                        continue

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

                    try:
                        X_sample, y_sample = next(X_gen)
                        logging.info(f"X_sample shape in main: {X_sample.shape}")
                        logging.info(f"y_sample shape in main: {y_sample.shape}")

                        logging.info(f"Feature columns in context: {feature_columns}")

                        if X_sample.shape[0] == 0 or y_sample.shape[0] == 0:
                            logging.error(f"Empty sample batch for {collection_name}")
                            continue

                    except StopIteration:
                        logging.error(f"No data available in generator for {collection_name}")
                        continue

                    # Get a sample batch from the data generator
                    X_sample, y_sample = next(X_gen)

                    # Fit scalers on training data
                    feature_scaler.fit(X_sample)
                    target_scaler.fit(y_sample)

                    # Transform the test data using the scalers fitted on training data
                    try:
                        X_test, y_test = load_test_data(collection_name)
                        X_test_scaled = feature_scaler.transform(X_test)
                        y_test_scaled = target_scaler.transform(y_test)
                    except Exception as e:
                        logging.error(f"Error loading or transforming test data for {collection_name}: {str(e)}")
                        continue

                    X_sample, y_sample = next(X_gen)
                    best_params = hyperparameter_optimization(X_sample, y_sample, X_sample.shape[1])

                    try:
                        model = build_model(X_sample.shape[1], **best_params)
                        logging.info(f"Input data shape: {X_sample.shape}")
                    except Exception as e:
                        logging.error(f"Error building model: {str(e)}")
                        continue

                    try:
                        model = train_and_evaluate_model(
                            model, X_gen, total_samples, X_test_scaled, y_test_scaled,
                            feature_scaler, target_scaler, symbol
                        )
                    except Exception as e:
                        logging.error(f"Error training and evaluating model: {str(e)}")
                        continue

                    logging.info("Model trained. Starting evaluation and prediction.")
                    predictions, metrics = evaluate_and_predict(model, X_test_scaled, y_test_scaled, target_scaler)

                    if predictions is not None:
                        store_predictions(predictions, symbol, model_outputs, target_scaler)

                    save_model_and_scalers(model, feature_scaler, target_scaler, symbol, model_dir)

                except Exception as e:
                    logging.error(f"Error processing {collection_name}: {str(e)}")
                finally:
                    # Cleanup after each symbol
                    del X_gen, X_test, y_test, model, predictions, metrics
                    gc.collect()
                    clear_session()

            # Force garbage collection after each symbol
            gc.collect()

            # Notify via Teams webhook after each symbol's model processing is complete
            send_teams_notification(f"Model training and evaluation completed for symbol {symbol}.", TEAMS_URL)

    except Exception as e:
        logging.error(f"An error occurred in main execution: {str(e)}")
    finally:
        # Final cleanup
        gc.collect()
        clear_session()



if __name__ == "__main__":
    log_file = 'model_building.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info("Starting model training script")

    try:
        # Get all available symbols from the database
        all_symbols = [name.split('_')[0] for name in db.list_collection_names() if 'processed_data' in name]
        logging.info(f"All available symbols in database: {all_symbols}")

        # Get the set of already processed symbols
        processed_symbols = get_processed_symbols()
        logging.info(f"Already processed symbols: {processed_symbols}")

        if len(argv) > 10:
            # If symbols are provided via command line, use them (up to 25)
            symbols_to_process = argv[10:35]
            logging.info(f"Processing symbols from command line arguments: {symbols_to_process}")
        else:
            # If no symbols provided, use all symbols that haven't been processed yet
            symbols_to_process = [sym for sym in all_symbols if sym not in processed_symbols][:25]
            logging.info(f"Processing unprocessed symbols from database: {symbols_to_process}")

        if not symbols_to_process:
            logging.info("No symbols to process. Please provide symbols or ensure the database contains unprocessed symbols.")
        else:
            main(symbols_to_process)

    except Exception as e:
        logging.exception(f"An error occurred in main execution: {e}")

    logging.info("Model training script completed")