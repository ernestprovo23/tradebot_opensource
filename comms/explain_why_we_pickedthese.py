import os
import pymongo
import requests
import json
import logging
from dotenv import load_dotenv
import anthropic

# Configure the logger
logging.basicConfig(filename='ai_analysis.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')
load_dotenv()

MONGO_DB_CONN_STRING = os.getenv("MONGO_DB_CONN_STRING")
teams_webhook_url = os.getenv("TEAMS_WEBHOOK_URL")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

mongo_client = pymongo.MongoClient(MONGO_DB_CONN_STRING)
db = mongo_client["stock_data"]
collection = db["openai_analysis"]

def get_latest_records():
    pipeline = [
        {"$sort": {"symbol": 1, "datetime": -1}},
        {"$group": {
            "_id": "$symbol",
            "latest_record": {"$first": "$$ROOT"}
        }}
    ]
    return list(collection.aggregate(pipeline))

def format_content(record):
    try:
        data = json.loads(record["content"])
    except json.JSONDecodeError:
        return None

    justification = data.get('Justification', {})
    if isinstance(justification, str):
        try:
            justification = json.loads(justification)
        except json.JSONDecodeError:
            justification = {}

    risk_management = justification.get('RiskManagement', {})
    fundamental_analysis = justification.get('FundamentalAnalysis', {})

    return {
        "symbol": record['symbol'],
        "analysis": data.get('Analysis', 'N/A'),
        "beta": str(risk_management.get('Beta', 'N/A')),
        "diversification": risk_management.get('Diversification', 'N/A'),
        "positive_indicators": ", ".join(fundamental_analysis.get('PositiveIndicators', ['N/A'])),
        "negative_indicators": ", ".join(fundamental_analysis.get('NegativeIndicators', ['N/A']))
    }


def refine_content_with_claude(content):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system="You are a financial analyst expert. Format the given stock analysis information into a concise summary suitable for a Microsoft Teams Adaptive Card. Provide a title, a brief summary, and up to 5 key points in a bullet-point format.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Format this stock analysis for a Teams Adaptive Card: {json.dumps(content)}"
                    }
                ]
            }
        ]
    )
    logging.info(f"Claude response: {message.content}")

    # Parse the response into a structured format
    parsed_response = json.loads(message.content[0].text)
    return {"symbol": content["symbol"], "card_content": parsed_response}


def create_teams_card(refined_content):
    logging.info(f"Creating Teams card with content: {refined_content}")
    card_content = refined_content['card_content']

    facts = [{"title": point["title"], "value": point["value"]} for point in card_content["key_points"]]

    return {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "type": "AdaptiveCard",
                "body": [
                    {
                        "type": "TextBlock",
                        "size": "Medium",
                        "weight": "Bolder",
                        "text": card_content["title"]
                    },
                    {
                        "type": "TextBlock",
                        "text": card_content["summary"],
                        "wrap": True
                    },
                    {
                        "type": "FactSet",
                        "facts": facts
                    }
                ],
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "version": "1.2"
            }
        }]
    }


def send_message_to_teams(content):
    headers = {"Content-Type": "application/json"}
    response = requests.post(teams_webhook_url, headers=headers, json=content)
    if response.status_code != 200:
        logging.error(f"Request to Teams returned an error {response.status_code}, the response is: {response.text}")

def main():
    latest_records = get_latest_records()
    for record in latest_records[:2]:  # Only process the first 5 records
        content = format_content(record['latest_record'])
        if content:
            logging.info(f"Formatted content: {content}")
            refined_content = refine_content_with_claude(content)
            logging.info(f"Refined content: {refined_content}")
            teams_card = create_teams_card(refined_content)
            send_message_to_teams(teams_card)
            logging.info(f"Refined message sent for symbol: {record['_id']}")
        else:
            logging.warning(f"Failed to format content for symbol: {record['_id']}")

if __name__ == "__main__":
    main()