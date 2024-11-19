import os
import joblib
import logging
import json
from sklearn.metrics import mean_squared_error
from datetime import datetime
import warnings
import stat

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

logging.basicConfig(filename='model_training_logfile.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def set_permissions(directory):
    try:
        os.chmod(directory, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
        logging.info(f"Permissions set for directory: {directory}")
    except Exception as e:
        logging.error(f"Failed to set permissions for directory {directory}: {str(e)}")
        print(f"Failed to set permissions for directory {directory}: {str(e)}")

def get_latest_file(directory, file_prefix):
    print(f"Checking for latest file with prefix '{file_prefix}' in directory '{directory}'")
    files = os.listdir(directory)
    relevant_files = [f for f in files if f.startswith(file_prefix) and f.endswith(".pkl")]
    if relevant_files:
        latest_file = sorted(relevant_files, key=lambda x: x.split('_')[-1].split('.')[0], reverse=True)[0]
        print(f"Latest file found: {latest_file}")
        return latest_file
    print(f"No relevant files found for prefix '{file_prefix}'")
    return None

def load_and_verify_scalers(feature_scaler_path, target_scaler_path):
    try:
        print(f"Loading feature scaler from '{feature_scaler_path}'")
        feature_scaler = joblib.load(feature_scaler_path)
        print(f"Loading target scaler from '{target_scaler_path}'")
        target_scaler = joblib.load(target_scaler_path)
        return feature_scaler, target_scaler
    except Exception as e:
        logging.error(f"Error loading scaler files: {str(e)}")
        print(f"Error loading scaler files: {str(e)}")
        return None, None

def train_model(symbol, model_dir):
    print(f"Starting training for symbol '{symbol}'")
    set_permissions(model_dir)
    latest_model_file = get_latest_file(model_dir, f"model_{symbol}")
    latest_feature_scaler_file = get_latest_file(model_dir, f"feature_scaler_{symbol}")
    latest_target_scaler_file = get_latest_file(model_dir, f"target_scaler_{symbol}")

    if not (latest_model_file and latest_feature_scaler_file and latest_target_scaler_file):
        logging.warning(f"Missing files for {symbol}, skipping...")
        print(f"Missing files for {symbol}, skipping...")
        return

    model_path = os.path.join(model_dir, latest_model_file)
    feature_scaler_path = os.path.join(model_dir, latest_feature_scaler_file)
    target_scaler_path = os.path.join(model_dir, latest_target_scaler_file)

    try:
        print(f"Loading model from '{model_path}'")
        model = joblib.load(model_path)
        feature_scaler, target_scaler = load_and_verify_scalers(feature_scaler_path, target_scaler_path)
        if feature_scaler is None or target_scaler is None:
            print(f"Failed to load scalers for '{symbol}', skipping...")
            return
    except Exception as e:
        logging.error(f"Error loading files for {symbol}: {str(e)}")
        print(f"Error loading files for {symbol}: {str(e)}")
        return

    logging.info(f"Model trained for {symbol}")
    print(f"Model successfully trained for '{symbol}'")

    new_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_model_path = os.path.join(model_dir, f"model_{symbol}_{new_version}.pkl")
    print(f"Saving new model version to '{new_model_path}'")
    joblib.dump(model, new_model_path)

    metadata = {
        "model_version": new_version,
        "timestamp": logging.Formatter('%(asctime)s').format(logging.makeLogRecord({'levelname': 'INFO'}))
    }
    metadata_path = os.path.join(model_dir, f"training_metadata_{symbol}_{new_version}.json")
    print(f"Saving metadata to '{metadata_path}'")
    with open(metadata_path, 'w') as metafile:
        json.dump(metadata, metafile)

def process_models(base_dir):
    print(f"Processing models in base directory '{base_dir}'")
    for symbol_dir in os.listdir(base_dir):
        full_symbol_path = os.path.join(base_dir, symbol_dir)
        if os.path.isdir(full_symbol_path):
            print(f"Processing model for '{symbol_dir}'")
            train_model(symbol_dir, full_symbol_path)

if __name__ == "__main__":
    model_dir = "/home/jaguar/DSEalgo_v2/modelfiles/"
    set_permissions(model_dir)
    print(f"Starting processing with model directory '{model_dir}'")
    process_models(model_dir)
    print("Model processing completed.")
