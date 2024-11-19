import os
import json
import requests
from pymongo import MongoClient
from dotenv import load_dotenv
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TeamsCommunicator:
    def __init__(self, db_name, collection_name):
        """Initialize connection to MongoDB with specified database and collection."""
        load_dotenv()
        mongo_db_conn_string = os.getenv("MONGO_DB_CONN_STRING")
        self.teams_url = os.getenv("TEAMS_WEBHOOK_URL")
        self.mongo_client = MongoClient(mongo_db_conn_string)
        self.db = self.mongo_client[db_name]
        self.collection = self.db[collection_name]

    def send_teams_message(self, message):
        """Sends a message to a Microsoft Teams channel via webhook."""
        headers = {"Content-Type": "application/json"}
        payload = {"text": message}
        try:
            response = requests.post(self.teams_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            logging.info("Message sent to Teams successfully.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error sending message to Teams: {e}")

# Example usage (to be adapted by the script calling this class):
if __name__ == "__main__":
    # You would normally specify the database and collection here when creating the instance
    communicator = TeamsCommunicator("stock_data", "selected_pairs")
    # Example message to send; replace with actual message building logic
    sample_message = "Hello, Teams! This is a test message."
    communicator.send_teams_message(sample_message)
