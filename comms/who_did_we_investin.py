import os
from pymongo import MongoClient
import requests
import json
from dotenv import load_dotenv
import logging

logging.basicConfig(filename='openai_analysis.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGO_DB_CONN_STRING = os.getenv("MONGO_DB_CONN_STRING")

# Teams webhook URL
TEAMS_URL = os.getenv("TEAMS_WEBHOOK_URL")

# Initialize MongoDB client
mongo_client = MongoClient(MONGO_DB_CONN_STRING)

# Select the database and collection
db = mongo_client["stock_data"]
selected_pairs_collection = db["selected_pairs"]


def fetch_data_with_join():
    """Performs an aggregation to join selected_pairs with company_overviews."""
    pipeline = [
        {
            '$lookup': {
                'from': 'company_overviews',
                'localField': 'Symbol',  # Adjust if the reference field in selected_pairs is different
                'foreignField': 'Symbol',
                'as': 'company_details'
            }
        },
        {'$unwind': '$company_details'},
        # Deconstructs the array field from the joined documents to output a document for each element
        {'$match': {'company_details': {'$exists': True}}}
        # Ensures that only documents with valid company details are included
    ]
    return list(selected_pairs_collection.aggregate(pipeline))


def build_message(data):
    """Builds a formatted message for Teams."""
    message = "### Selected Pairs Overview\n\n"
    message += "| Symbol | Company Name | Industry | Sector |\n"
    message += "| ------ | ------------ | -------- | ------ |\n"

    for item in data:
        company = item.get('company_details', {})
        message += f"| {item.get('Symbol', 'N/A')} | {company.get('Name', 'N/A')} | {company.get('Industry', 'N/A')} | {company.get('Sector', 'N/A')} |\n"

    return message


def send_teams_message(webhook_url, message):
    """Sends a message to a Microsoft Teams channel via webhook."""
    headers = {"Content-Type": "application/json"}
    payload = {"text": message}

    try:
        response = requests.post(webhook_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        logging.info("Message sent to Teams successfully.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending message to Teams: {e}")


if __name__ == "__main__":
    data = fetch_data_with_join()
    if data:
        message = build_message(data)
        send_teams_message(TEAMS_URL, message)
    else:
        logging.info("No data found after join operation.")
