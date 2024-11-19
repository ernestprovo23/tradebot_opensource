import os
from dotenv import load_dotenv
from pymongo import MongoClient
import requests
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGO_DB_CONN_STRING = os.getenv("MONGO_DB_CONN_STRING")
DB_NAME = "stock_data"

# Teams webhook URL
TEAMS_WEBHOOK_URL = os.getenv("TEAMS_WEBHOOK_URL")

# MongoDB connection
mongo_client = MongoClient(MONGO_DB_CONN_STRING)
db = mongo_client[DB_NAME]

def fetch_recent_analyses():
    twenty_four_hours_ago = datetime.now() - timedelta(hours=24)
    return list(db.openai_analysis.find({
        "datetime": {"$gte": twenty_four_hours_ago.isoformat()}
    }))

def extract_section(content, section):
    start = content.find(f'"{section}":')
    if start != -1:
        end = content.find('"}', start)
        if end == -1:
            end = content.find('},', start)
        if end == -1:
            end = len(content)
        return content[start:end].strip()
    return ""

def send_to_teams(analysis):
    try:
        content = analysis['content']
        card = {
            "type": "AdaptiveCard",
            "body": [
                {
                    "type": "TextBlock",
                    "size": "Medium",
                    "weight": "Bolder",
                    "text": f"AI Analysis/Feedback: {analysis['symbol']}"
                },
                {
                    "type": "TextBlock",
                    "text": "Analysis:",
                    "weight": "Bolder"
                },
                {
                    "type": "TextBlock",
                    "text": extract_section(content, 'Analysis'),
                    "wrap": True
                },
                {
                    "type": "TextBlock",
                    "text": "Numbers:",
                    "weight": "Bolder"
                },
                {
                    "type": "TextBlock",
                    "text": extract_section(content, 'Numbers'),
                    "wrap": True
                },
                {
                    "type": "TextBlock",
                    "text": "Justification:",
                    "weight": "Bolder"
                },
                {
                    "type": "TextBlock",
                    "text": extract_section(content, 'Justification'),
                    "wrap": True
                }
            ],
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "version": "1.2"
        }

        payload = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": card
                }
            ]
        }

        response = requests.post(TEAMS_WEBHOOK_URL, json=payload)
        response.raise_for_status()
        print(f"Card for {analysis['symbol']} sent successfully")
    except Exception as e:
        print(f"Error sending card for {analysis['symbol']}: {e}")

def main():
    recent_analyses = fetch_recent_analyses()
    for analysis in recent_analyses:
        send_to_teams(analysis)

if __name__ == "__main__":
    main()