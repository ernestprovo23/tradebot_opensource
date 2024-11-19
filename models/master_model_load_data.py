import subprocess
import sys
import time
import os
import logging

# Configure the logger
logging.basicConfig(filename='logfile.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Append the directory containing 'teams_communicator.py' to sys.path
parent_dir = os.path.dirname(script_dir)  # This is the 'DSEalgo_v2' directory
sys.path.append(parent_dir)  # This should now correctly include the path to the 'teams_communicator' module

from teams_communicator import TeamsCommunicator  # Now we can import TeamsCommunicator

# List your scripts in the desired order of execution
scripts = [
    "balance_sheets.py",
    "cash_flow.py",
    "earnings_calendar.py",
    "income_statements.py",
    "news_sentiment.py",
    "selected_pairs_ts_data.py",
    "copper_historical.py",
    "crude_oil.py",
    "inflation.py",
    "real_gdp.py",
    "retail_sales.py",
    "sugar_historical.py",
    "treasury_yields.py",
    "unemployment.py",
    "balance_sheet_cleaning.py",
    "cash_flow_cleaning.py",
    "ML_data_prep.py"
]

def check_script_exists(script_path):
    if not os.path.exists(script_path):
        logging.error(f"Script {script_path} does not exist.")
        return False
    return True

def run_script_with_retries(script, max_retries=2, wait_time=20):
    retries = 0
    while retries <= max_retries:
        logging.info(f"Attempting to run {script}")

        script_path = os.path.join(script_dir, script)
        if not check_script_exists(script_path):
            logging.error(f"Script file missing: {script}")
            break

        try:
            subprocess.check_call([sys.executable, script_path])
            logging.info(f"Successfully ran {script}")
            break
        except subprocess.CalledProcessError as e:
            logging.error(f"Error occurred while running {script}, exit code: {e.returncode}")
            retries += 1
            if retries <= max_retries:
                logging.info(f"Retrying {script} after waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"Script {script} failed after {max_retries} retries. Skipping this script and moving to the next one.")
                break

if __name__ == "__main__":
    communicator = TeamsCommunicator("ml_database", "prediction_logs")
    for script in scripts:
        run_script_with_retries(script)
        success_message = f"The {script} ran successfully."
        communicator.send_teams_message(success_message)


