import openai
import os
from dotenv import load_dotenv
import logging
from pymongo import MongoClient
import json
from openai import OpenAI
from datetime import datetime, timedelta
import requests
import time
from pydantic import BaseModel, Field
from typing import Optional


# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(filename='AI_logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MongoDB configuration
MONGO_DB_CONN_STRING = os.getenv("MONGO_DB_CONN_STRING")
DB_NAME = "stock_data"

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Teams webhook configuration
TEAMS_WEBHOOK_URL = os.getenv("TEAMS_WEBHOOK_URL")

# Ensure the API key is loaded correctly
if not OPENAI_API_KEY:
    logging.error("Failed to load OPENAI_API_KEY from environment variables.")
    exit(1)

# Initialize OpenAI client with the API key
openai.api_key = OPENAI_API_KEY

client = OpenAI()

# MongoDB connection
mongo_client = MongoClient(MONGO_DB_CONN_STRING)
db = mongo_client[DB_NAME]

trade_stats = {
    """
    # Profit Margin:
    # - Very low or Negative: A profit margin of 1% or lower is considered very low,
    #   and if it's negative (e.g., -5%), the company is losing money.
    #   Example: Amazon, in its early years, often reported negative profit margins.
    # - High: A profit margin above 15% is typically considered high.
    #   Example: Companies like Microsoft often have profit margins exceeding 30%.

    # Price-Earnings Ratio (P/E):
    # - Very low or Negative: A P/E ratio below 5 is considered low, suggesting
    #   the market has low expectations for the company's future. Companies with negative earnings have a negative P/E ratio.
    #   Example: In 2020, many airlines had negative P/E ratios due to substantial losses caused by the COVID-19 pandemic.
    # - High: A P/E ratio above 20 is typically considered high, indicating that
    #   the market expects high earnings growth.
    #   Example: Amazon has had a high P/E ratio for many years, often exceeding 100.

    # Return on Equity (ROE):
    # - Very low or Negative: An ROE below 5% is considered low, suggesting the company
    #   isn't generating much profit from its equity. Negative ROE (e.g., -10%) means the company is losing money.
    #   Example: In 2008 during the financial crisis, many banks reported negative ROE.
    # - High: An ROE above 20% is generally considered high.
    #   Example: Companies like Apple have consistently reported ROE above 30%.

    # EV to EBITDA:
    # - Very low or Negative: An EV/EBITDA below 5 is generally considered low, suggesting
    #   the company might be undervalued, assuming it's a profitable business. Negative values can occur if EBITDA is negative,
    #   indicating operating losses. Example: In 2008, during the financial crisis, some banks had low EV/EBITDA ratios.
    # - High: An EV/EBITDA above 15 is usually considered high, suggesting the company may be overvalued.
    #   High-growth tech companies often have high EV/EBITDA ratios. Example: Zoom Video Communications had an EV/EBITDA ratio over 200 in 2020.

    # Quarterly Earnings Growth YoY:
    # - Very low or Negative: Negative quarterly earnings growth means the company's earnings have shrunk compared to the same quarter in the previous year.
    #   Example: During the COVID-19 pandemic in 2020, many companies in the travel and hospitality industry faced negative quarterly earnings growth.
    # - High: A high number (e.g., 50% or higher) would indicate a significant increase in earnings compared to the same quarter in the previous year.
    #   Example: Many tech companies like Apple and Amazon reported high quarterly earnings growth in 2020 due to the increased demand for digital services amidst the pandemic.
    """
}

# Read trading strategy from file
def read_trading_strategy(file_path):
    try:
        with open(file_path, 'r') as file:
            strategy = file.read()
        return strategy
    except Exception as e:
        logging.error(f"An error occurred while reading the trading strategy file: {e}")
        return ""

trading_strategy = read_trading_strategy('trading_strategy.md')

def fetch_selected_pairs():
    try:
        selected_pairs_collection = db['selected_pairs']
        selected_pairs = list(selected_pairs_collection.find())
        bad_performers = fetch_bad_performers()
        return selected_pairs, bad_performers
    except Exception as e:
        logging.error(f"An error occurred while fetching selected pairs from MongoDB: {e}")
        return [], []

def fetch_bad_performers():
    try:
        bad_performers_collection = mongo_client['trading_db']['bad_performers']
        bad_performers = list(bad_performers_collection.find({}, {'symbol': 1, '_id': 0}))
        return [doc['symbol'] for doc in bad_performers]
    except Exception as e:
        logging.error(f"An error occurred while fetching bad performers from MongoDB: {e}")
        return []

def insert_analysis_result(db, analysis_result):
    try:
        fifteen_days_ago = datetime.now() - timedelta(days=15)
        existing_document = db.openai_analysis.find_one({
            "symbol": analysis_result["symbol"],
            "datetime": {"$gte": fifteen_days_ago.isoformat()}
        })

        if existing_document:
            logging.info(f"Skipping insert. Analysis for {analysis_result['symbol']} already exists within the last 15 days.")
            print(f"Skipping insert. Analysis for {analysis_result['symbol']} already exists within the last 15 days.")
            return

        db.openai_analysis.insert_one(analysis_result)
        logging.info(f"Successfully inserted analysis result for {analysis_result['symbol']}")
    except Exception as e:
        logging.error(f"An error occurred while inserting the analysis result: {e}")


def send_to_teams(webhook_url, analysis_result):
    try:
        headers = {
            "Content-Type": "application/json"
        }

        # Extract data from the analysis content
        content = json.loads(analysis_result['content'])

        # Prepare a more structured, detailed payload for Teams card
        payload = {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "summary": f"Stock Analysis for {analysis_result['symbol']}",
            "themeColor": "0072C6",
            "title": f"Stock Analysis Report: {analysis_result['symbol']}",
            "sections": [
                {
                    "activityTitle": f"Analysis for {analysis_result['symbol']} as of {datetime.fromisoformat(analysis_result['datetime']).strftime('%m/%d/%Y %H:%M:%S')}",
                    "facts": [
                        {"name": "Profit Margin", "value": f"{content['Numbers'].get('ProfitMargin', 'N/A'):.2f}%"},
                        {"name": "P/E Ratio", "value": content['Numbers'].get('PERatio', 'N/A')},
                        {"name": "EV/EBITDA", "value": content['Numbers'].get('EVToEBITDA', 'N/A')},
                        {"name": "ROE", "value": content['Numbers'].get('ROE', 'N/A')},
                        {"name": "Dividend Yield", "value": f"{content['Numbers'].get('DividendYield', 'N/A'):.2f}%"},
                        {"name": "Quarterly Earnings Growth YoY", "value": f"{content['Numbers'].get('QuarterlyEarningsGrowthYOY', 'N/A'):.2f}%"}
                    ],
                    "text": content.get('Analysis', 'No analysis provided.')
                },
                {
                    "activityTitle": "Justification:",
                    "text": content.get('Justification', 'No justification provided.')
                }
            ]
        }

        response = requests.post(webhook_url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            logging.info(f"Successfully sent analysis result for {analysis_result['symbol']} to Teams")
        else:
            logging.error(f"Failed to send analysis result for {analysis_result['symbol']} to Teams: {response.status_code} {response.text}")

    except Exception as e:
        logging.error(f"An error occurred while sending analysis result to Teams: {e}")


def fetch_prediction_data():
    try:
        predictions_collection = mongo_client['predictions']['September_2024']
        prediction_data = list(predictions_collection.find({}, {'symbol': 1, 'predicted_price': 1, '_id': 0}))
        return {doc['symbol']: doc for doc in prediction_data}
    except Exception as e:
        logging.error(f"An error occurred while fetching prediction data from MongoDB: {e}")
        return {}


def fetch_symbol_data(symbol):
    """
    Fetches the entire record for the given symbol from the 'selected_pairs' collection
    and merges it into the stock_data for further analysis.
    """
    try:
        symbol_record = db['selected_pairs'].find_one({'symbol': symbol})
        if not symbol_record:
            logging.warning(f"No record found for symbol: {symbol}")
            return None
        return symbol_record
    except Exception as e:
        logging.error(f"An error occurred while fetching data for symbol {symbol}: {e}")
        return None


class AnalysisNumbers(BaseModel):
    ProfitMargin: Optional[float] = Field(None, description="Profit margin percentage")
    PERatio: Optional[float] = Field(None, description="Price-Earnings ratio")
    EVToEBITDA: Optional[float] = Field(None, description="EV to EBITDA ratio")
    ROE: Optional[float] = Field(None, description="Return on Equity")
    QuarterlyEarningsGrowthYOY: Optional[float] = Field(None, description="Quarterly earnings growth year-over-year percentage")
    DividendYield: Optional[float] = Field(None, description="Dividend yield percentage")

class AnalysisResult(BaseModel):
    Analysis: str
    Numbers: AnalysisNumbers
    Justification: str


def fetch_latest_close_prices():
    try:
        # Aggregating to get the latest close price for each symbol
        aggregation_pipeline = [
            {"$match": {"function": "TIME_SERIES_DAILY_data"}},
            {"$sort": {"timestamp": -1}},  # Sort by latest timestamp first
            {"$group": {
                "_id": "$symbol",
                "latest_close_price": {"$first": "$close_price"}
            }}
        ]

        aggregate_collection = mongo_client['stock_data']['aggregated_stock_data']
        results = list(aggregate_collection.aggregate(aggregation_pipeline))

        # Convert aggregation result to a dictionary with symbol as key
        return {result['_id']: result['latest_close_price'] for result in results}

    except Exception as e:
        logging.error(f"An error occurred while fetching latest close prices: {e}")
        return {}


def analyze_stock_data(stock_data, bad_performers, predictions_data, latest_close_prices):
    try:
        if stock_data["symbol"] in predictions_data and stock_data["symbol"] in latest_close_prices:
            predicted_price = predictions_data[stock_data["symbol"]].get("predicted_price")
            stock_close_price = latest_close_prices[stock_data["symbol"]]

            price_change = "rise" if predicted_price > stock_close_price else "drop"

            # Setup context for analysis
            bad_performers_info = "Yes" if stock_data["symbol"] in bad_performers else "No"

            messages = [
                {"role": "system",
                 "content": "Your role is to act as a Quantitative Trading Engineer, providing expert advice and solutions in the field of quantitative trading. "
                            "Your goal is to deliver the most accurate and logical information in JSON format. "
                            "Your report will have 3 sections: 1) Analysis, 2) Numbers, 3) Justification. "
                            "You have access to different datasets including balance sheets, cash flows, earnings calendars, income statements, "
                            "news sentiment, timeseries data for selected pairs, copper and sugar historical prices, crude oil data, "
                            "inflation metrics, GDP figures, retail sales data, treasury yields, and unemployment metrics. "
                            "Use these datasets to inform your analysis and justification."},
                {"role": "user",
                 "content": f"Analyze the stock data: {stock_data} while excluding symbols from the {bad_performers_info} category. "
                            f"Determine the predicted price movement to {price_change}. If the most recent close price is greater than the predicted price, "
                            f"provide an analysis explaining high-level market conditions, using data types mentioned previously, such as earnings or sentiment, "
                            f"that might justify a price decrease. Conversely, if the predicted price is greater, justify the expectation for an increase. "
                            f"For instance, consider market conditions in the balance sheet, income statement, and economic indicators data sets that "
                            f"support the analysis. Analyze only symbols from both prediction and actual lists "
                            f"and produce a concise summary of your trading justification."}
            ]

            logging.info("Sending request to OpenAI API")

            completion = client.beta.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format=AnalysisResult
            )

            logging.info("Received response from OpenAI API")

            if 'refusal' in completion.choices[0].message:
                logging.warning(f"Refusal message for {stock_data['symbol']}: {completion.choices[0].message.refusal}")
            else:
                analysis_content = completion.choices[0].message.parsed

            logging.info(f"Usage: {completion.usage}")

            # Inside analyze_stock_data function
            if analysis_content:
                analysis_result = {
                    "symbol": stock_data["symbol"],
                    "datetime": datetime.now().isoformat(),
                    "content": json.dumps(analysis_content.dict(), indent=2),
                    "token_usage": {
                        "completion_tokens": completion.usage.completion_tokens,
                        "prompt_tokens": completion.usage.prompt_tokens,
                        "total_tokens": completion.usage.total_tokens
                    }
                }

                insert_analysis_result(db, analysis_result)
                send_to_teams(TEAMS_WEBHOOK_URL, analysis_result)
                print(json.dumps(analysis_result, default=str, indent=2))
                return analysis_content

            else:
                logging.error(f"Failed to get structured analysis for {stock_data['symbol']}")

        else:
            logging.info(f"Symbol {stock_data['symbol']} is missing from one of the data sources.")

    except Exception as e:
        logging.error(f"An error occurred while analyzing stock data: {e}")
        return None


def format_analysis_content(analysis_content):
    try:
        # Attempt to parse the JSON response
        start_index = analysis_content.find('{')
        end_index = analysis_content.rfind('}') + 1
        if start_index == -1 or end_index == 0:
            raise ValueError("Invalid JSON format in the response")

        json_content = analysis_content[start_index:end_index]
        analysis_data = json.loads(json_content)

        # Structure the analysis content
        uniform_analysis = {
            "Analysis": analysis_data.get("Analysis", "No analysis provided."),
            "Numbers": {
                "ProfitMargin": analysis_data.get("Numbers", {}).get("ProfitMargin", "N/A"),
                "PERatio": analysis_data.get("Numbers", {}).get("PERatio", "N/A"),
                "EVToEBITDA": analysis_data.get("Numbers", {}).get("EVToEBITDA", "N/A"),
                "ROE": analysis_data.get("Numbers", {}).get("ROE", "N/A"),
                "QuarterlyEarningsGrowthYOY": analysis_data.get("Numbers", {}).get("QuarterlyEarningsGrowthYOY", "N/A"),
                "DividendYield": analysis_data.get("Numbers", {}).get("DividendYield", "N/A")
            },
            "Justification": analysis_data.get("Justification", "No justification provided.")
        }

        formatted_content = json.dumps(uniform_analysis, indent=2)
        return f"```json\n{formatted_content}\n```"
    except (ValueError, json.JSONDecodeError) as e:
        logging.error(f"An error occurred while formatting analysis content: {e}")
        return None


if __name__ == "__main__":
    # Fetch necessary data
    prediction_data = fetch_prediction_data()
    selected_pairs, bad_performers = fetch_selected_pairs()
    latest_close_prices = fetch_latest_close_prices()

    # Filter selected pairs based on available prediction data
    filtered_pairs = [pair for pair in selected_pairs if pair['symbol'] in prediction_data]

    # # Limit to 3 symbols for testing
    # test_pairs = filtered_pairs[:9]

    for pair in filtered_pairs:
        analyze_stock_data(pair, bad_performers, prediction_data, latest_close_prices)
        print(f"Finished analysis for {pair['symbol']}")
        time.sleep(1)