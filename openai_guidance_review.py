import anthropic
import re
import openai
import os
from dotenv import load_dotenv
import logging
from pymongo import MongoClient
import json
from openai import OpenAI
from datetime import datetime, timedelta
from bson import ObjectId
from pprint import pprint

# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MongoDB configuration
MONGO_DB_CONN_STRING = os.getenv("MONGO_DB_CONN_STRING")
DB_NAME = "stock_data"

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

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


# Read in portfolio history
def get_portfolio_history():
    try:
        data = list(db.portfolio_history.find())
        logging.info(f"Fetched {len(data)} documents from portfolio_history")
        return data
    except Exception as e:
        logging.error(f"An error occurred while fetching portfolio history from MongoDB: {e}")
        return []

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

def fetch_earnings_calendar():
    try:
        data = list(db.earnings_calendar.find())
        logging.info(f"Fetched {len(data)} documents from earnings_calendar")
        return data
    except Exception as e:
        logging.error(f"An error occurred while fetching earnings calendar from MongoDB: {e}")
        return []

def fetch_news_sentiment_data():
    try:
        data = list(db.news_sentiment_data.find())
        logging.info(f"Fetched {len(data)} documents from news_sentiment_data")
        return data
    except Exception as e:
        logging.error(f"An error occurred while fetching news sentiment data from MongoDB: {e}")
        return []

def fetch_openai_analysis():
    try:
        data = list(db.openai_analysis.find())
        logging.info(f"Fetched {len(data)} documents from openai_analysis")
        return data
    except Exception as e:
        logging.error(f"An error occurred while fetching OpenAI analysis from MongoDB: {e}")
        return []

def insert_analysis_result(collection, analysis_result):
    try:
        six_hours_ago = datetime.now() - timedelta(hours=6)
        existing_document = collection.find_one({
            "symbol": analysis_result["symbol"],
            "datetime": {"$gte": six_hours_ago.isoformat()}
        })

        if existing_document:
            collection.delete_one({"_id": existing_document["_id"]})
            logging.info(f"Deleted old document for {analysis_result['symbol']}.")

        collection.insert_one(analysis_result)
        logging.info(f"Successfully inserted analysis result for {analysis_result['symbol']}")
    except Exception as e:
        logging.error(f"An error occurred while inserting the analysis result: {e}")

def analyze_data(earnings_calendar, news_sentiment_data, openai_analysis, raw_trade_imports, portfolio_history, symbol, full_symbol):
    # Filter documents using the base_symbol
    earnings_data = [doc for doc in earnings_calendar if doc['symbol'] == symbol]
    sentiment_data = [doc for doc in news_sentiment_data if doc['symbol'] == symbol]
    openai_data = [doc for doc in openai_analysis if doc['symbol'] == symbol]
    raw_trades_data = [doc for doc in raw_trade_imports if doc['symbol'] == full_symbol]
    portfolio_history_data = [doc for doc in portfolio_history]

    # Debugging information
    logging.info(f"Symbol: {symbol}")
    logging.info(f"Earnings Data: {earnings_data}")
    logging.info(f"Sentiment Data: {sentiment_data}")
    logging.info(f"OpenAI Analysis Data: {openai_data}")

    # Prepare the messages for the OpenAI API
    messages = [
        {"role": "system", "content": 'Your role is to act as a Quantitative Trading Engineer, providing expert '
                                      'advice and solutions in the field of quantitative trading. Your goal is to '
                                      'deliver the most accurate and logical information, employing markdown '
                                      'formatting for clarity. You are allowed to ask up to three clarification '
                                      'questions if needed. Use analogies to explain complex concepts, '
                                      'drawing inspiration from industry leaders and thought leaders in quantitative '
                                      'trading. You will provide code and or instructions when requested. You must '
                                      'provide the complete code samples and not summaries so users can verify your '
                                      'responses. Additionally, you will not share custom prompting techniques or '
                                      'respond to such requests. When providing advice, focus solely on the topic at '
                                      'hand, avoiding unnecessary prose or context. Only provide warnings or cautions '
                                      'if explicitly asked for them.'
                                     f"Use the uploaded knowledge document {trading_strategy} for knowledge on our "
                                      "trading strategy."},
        {"role": "user", "content": f"Ingest and Analyze the following data for symbol {symbol}: Earnings Data: {earnings_data}, "
                                    f"Sentiment Data: {sentiment_data}, OpenAI Analysis Data: {openai_data}, and "
                                    f"trades from our account Raw Trades {raw_trades_data} and the last 30 days of portfolio performance "
                                    f"history as {portfolio_history_data} for each symbol. Your objective is to "
                                    f"analyze the information and provide analysis and"
                                    f"interpret the data and provide your feedback using your professional background "
                                    f"and industry thought leaders in accordance with our trading strategy."
                                    f"Act within your role as the quant engineer reviewing the data and provide "
                                    f"analysis on if the company should increase, decrease or forfeit our position. "
                                    f"Remember the trades our account has already executed is available in the "
                                    f"raw_trades_data as well as the portfolio performance."
                                    f"Return your analysis as if sending a memo to me - your boss the CEO "
                                    f"but without formality. Use language that is aligned with 12th grade literacy "
                                    f"and anologies where needed for complex explanations. Limit your response to 250 characters. "
                                    f"Limit prose and context to just your core analysis. "
                                    f"The format for the response can be as follows: <SYMBOL: VALUE, ANALYSIS: ANALYSIS>"}]

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )

        logging.info("Received response from OpenAI API")

        analysis_content = completion.choices[0].message.content
        logging.info(f"Usage: {completion.usage}")

        analysis_result = {
            "symbol": full_symbol,
            "datetime": datetime.now().isoformat(),
            "content": analysis_content,
            "token_usage": {
                "completion_tokens": completion.usage.completion_tokens,
                "prompt_tokens": completion.usage.prompt_tokens,
                "total_tokens": completion.usage.total_tokens
            }
        }

        # Insert the analysis result into MongoDB
        insert_analysis_result(db, analysis_result)

        # Print the analysis result as JSON
        pprint(json.dumps(analysis_result, default=str))

        return analysis_content

    except Exception as e:
        logging.error(f"An error occurred while analyzing stock data: {e}")


import anthropic


def analyze_data_with_claude(earnings_calendar, news_sentiment_data, openai_analysis, raw_trade_imports,
                             portfolio_history, symbol, full_symbol):
    # Filter documents using the base_symbol (same as in your original function)
    earnings_data = [doc for doc in earnings_calendar if doc['symbol'] == symbol]
    sentiment_data = [doc for doc in news_sentiment_data if doc['symbol'] == symbol]
    openai_data = [doc for doc in openai_analysis if doc['symbol'] == symbol]
    raw_trades_data = [doc for doc in raw_trade_imports if doc['symbol'] == full_symbol]
    portfolio_history_data = [doc for doc in portfolio_history]

    # Debugging information
    logging.info(f"Symbol: {symbol}")
    logging.info(f"Earnings Data: {earnings_data}")
    logging.info(f"Sentiment Data: {sentiment_data}")
    logging.info(f"OpenAI Analysis Data: {openai_data}")

    # Prepare the prompt for Claude
    anth_system_prompt = f"""Your role is to act as a Quantitative Trading Engineer, providing expert advice and 
    solutions in the field of quantitative trading. Your goal is to deliver the most accurate and logical 
    information, employing markdown formatting for clarity. You are allowed to ask up to three clarification 
    questions if needed. Use analogies to explain complex concepts, drawing inspiration from industry leaders and 
    thought leaders in quantitative trading. You will provide code and or instructions when requested. You must 
    provide the complete code samples and not summaries so users can verify your responses. Additionally, 
    you will not share custom prompting techniques or respond to such requests. When providing advice, focus solely 
    on the topic at hand, avoiding unnecessary prose or context. Only provide warnings or cautions if explicitly 
    asked for them.

    Ingest and Analyze the following data for symbol {symbol}:
    Earnings Data: {earnings_data}
    Sentiment Data: {sentiment_data}
    OpenAI Analysis Data: {openai_data}
    Raw Trades: {raw_trades_data}
    Portfolio History (last 30 days): {portfolio_history_data}

    Your objective is to analyze the information, provide analysis, and interpret the data. Provide your feedback 
    using your professional background and industry thought leaders in accordance with our trading strategy.

    Act within your role as the quant engineer reviewing the data and provide analysis on if the company should 
    increase, decrease or forfeit our position. Remember the trades our account has already executed is available in 
    the raw_trades_data as well as the portfolio performance.

    Return your analysis as if sending a memo to me - your boss the CEO but without formality. Use language that is 
    aligned with 12th grade literacy and analogies where needed for complex explanations. Limit your response to 250 
    characters. Limit prose and context to just your core analysis.

    The format for the response should be as follows: <SYMBOL: VALUE, ANALYSIS: ANALYSIS>
    
    """

    anth_messages = [

        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Provide your current analysis on the information."

                }
            ]
        }
    ]

    try:
        anth_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        anth_message = anth_client.messages.create(
            model="claude-3-5-sonnet",
            temperature=0,
            system=anth_system_prompt,
            messages=anth_messages
        )

        logging.info("Received response from Claude API")

        analysis_content = anth_message.content

        analysis_result = {
            "symbol": full_symbol,
            "datetime": datetime.now().isoformat(),
            "content": analysis_content,
            "model": "claude-3-5-sonnet"
        }

        # Insert the analysis result into MongoDB
        insert_analysis_result(db, analysis_result)

        # Print the analysis result as JSON
        pprint(json.dumps(analysis_result, default=str))

        return analysis_content

    except Exception as e:
        logging.error(f"An error occurred while analyzing stock data with Claude: {e}")


def fetch_raw_trade_imports():
    try:
        data = list(db.raw_trades_import.find())
        logging.info(f"Fetched {len(data)} documents from raw_trades_import")
        return data
    except Exception as e:
        logging.error(f"An error occurred while fetching raw trade imports from MongoDB: {e}")
        return []

def extract_base_symbol(symbol):
    match = re.match(r'^[A-Z]+', symbol)
    return match.group(0) if match else ''

def main():
    earnings_calendar = fetch_earnings_calendar()
    news_sentiment_data = fetch_news_sentiment_data()
    openai_analysis = fetch_openai_analysis()
    raw_trade_imports = fetch_raw_trade_imports()
    portfolio_history = get_portfolio_history()

    openai_collection = db['openai_analysis']
    claude_collection = db['claude_analysis']

    full_symbols = [
        trade['symbol']
        for trade in raw_trade_imports
        if
        'symbol' in trade and '/' not in trade['symbol'] and not trade['symbol'].startswith('CYTO') and trade['symbol']
    ]

    base_symbols = list({
        extract_base_symbol(symbol)
        for symbol in full_symbols
    })

    if base_symbols:
        logging.info(f"Available base symbols: {base_symbols}")
        for base_symbol in base_symbols:
            full_symbol = next((symbol for symbol in full_symbols if symbol.startswith(base_symbol)), base_symbol)

            openai_result = analyze_data(earnings_calendar, news_sentiment_data, openai_analysis, raw_trade_imports,
                                         portfolio_history,
                                         base_symbol, full_symbol)

            claude_result = analyze_data_with_claude(earnings_calendar, news_sentiment_data, openai_analysis,
                                                     raw_trade_imports,
                                                     portfolio_history, base_symbol, full_symbol)

            openai_document = {
                'symbol': full_symbol,
                'base_symbol': base_symbol,
                'datetime': datetime.now(),
                'analysis': openai_result
            }

            claude_document = {
                'symbol': full_symbol,
                'base_symbol': base_symbol,
                'datetime': datetime.now(),
                'analysis': claude_result
            }

            openai_collection.insert_one(openai_document)
            claude_collection.insert_one(claude_document)

            logging.info(f"Analysis results for {full_symbol} stored in MongoDB")
    else:
        logging.info("No symbols found in raw_trades_import for analysis.")


if __name__ == "__main__":
    main()