import os
import json
import requests
from pymongo import MongoClient
from dotenv import load_dotenv
import logging

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

    def clean_content(self, content):
        """Clean and format the content for better readability."""
        # Removing the placeholder tags and improving readability
        content = content.replace('<', '**').replace('>', '**').replace('**:', '**')
        return content

    def fetch_and_send_messages(self):
        """Fetch documents from MongoDB collection and send formatted messages to Teams."""
        try:
            documents = list(self.collection.find())
            for doc in documents:
                symbol = doc.get("symbol")
                content = doc.get("content")
                if symbol and content:
                    cleaned_content = self.clean_content(content)
                    message = f"**Symbol:** {symbol}\n**Content:**\n{cleaned_content}"
                    self.send_teams_message(message)
        except Exception as e:
            logging.error(f"An error occurred while fetching documents from MongoDB: {e}")

# Main script execution
if __name__ == "__main__":
    # Initialize the communicator with the database and collection names
    communicator = TeamsCommunicator("stock_data", "openai_analysis")
    # Fetch documents and send messages to Teams
    communicator.fetch_and_send_messages()
