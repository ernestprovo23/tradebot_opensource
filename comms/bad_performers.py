import os
import json
import requests
from pymongo import MongoClient
from dotenv import load_dotenv
import logging
from datetime import datetime

# Logging configuration
logging.basicConfig(filename='teams_sender.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TeamsCommunicator:
    def __init__(self, db_name, collection_name):
        """Initialize connection to MongoDB with specified database and collection."""
        load_dotenv()
        mongo_db_conn_string = os.getenv("MONGO_DB_CONN_STRING")
        self.teams_url = os.getenv("TEAMS_WEBHOOK_URL")
        self.mongo_client = MongoClient(mongo_db_conn_string)
        self.db = self.mongo_client[db_name]
        self.collection = self.db[collection_name]

    def send_teams_message(self, message, card_content):
        """Sends a message and a card to a Microsoft Teams channel via webhook."""
        headers = {"Content-Type": "application/json"}
        payload = {
            "text": message,
            "sections": [
                {
                    "activityTitle": "Risk Parameters Violation",
                    "activitySubtitle": "Sold due to price drop",
                    "facts": card_content,
                    "markdown": True
                }
            ]
        }
        try:
            response = requests.post(self.teams_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            logging.info("Message sent to Teams successfully.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error sending message to Teams: {e}")

    def fetch_and_send_messages(self):
        """Fetch documents from MongoDB collection and send formatted messages to Teams."""
        try:
            documents = list(self.collection.find())
            if documents:
                card_content = []
                for doc in documents:
                    symbol = doc.get("symbol")
                    date_added = doc.get("date_added", datetime.now())
                    avoid_until = doc.get("avoid_until", datetime.now())
                    card_content.append({
                        "name": "Symbol",
                        "value": symbol
                    })
                    card_content.append({
                        "name": "Date Added",
                        "value": date_added.strftime('%Y-%m-%d %H:%M:%S')
                    })
                    card_content.append({
                        "name": "Avoid Until",
                        "value": avoid_until.strftime('%Y-%m-%d %H:%M:%S')
                    })
                    card_content.append({"name": "", "value": ""})  # Adding an empty line for spacing

                message = "The following have been sold because of risk parameters violation due to price drop:"
                self.send_teams_message(message, card_content)
            else:
                logging.info("No documents found in the collection.")
        except Exception as e:
            logging.error(f"An error occurred while fetching documents from MongoDB: {e}")

# Main script execution
if __name__ == "__main__":
    # Initialize the communicator with the database and collection names
    communicator = TeamsCommunicator("trading_db", "bad_performers")
    # Fetch documents and send messages to Teams
    communicator.fetch_and_send_messages()
