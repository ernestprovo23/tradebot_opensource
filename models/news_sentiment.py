import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import pymongo
import hashlib

load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API')
MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')


def fetch_unique_symbols():
    client = pymongo.MongoClient(MONGO_CONN_STRING)
    try:
        db = client['stock_data']
        collection = db['selected_pairs']
        unique_symbols = collection.distinct('symbol')
        return unique_symbols
    finally:
        client.close()


def update_ticker_sentiment(filtered_data, company_ticker, ticker_data, ticker_cumulative_score):
    ticker_sentiment = filtered_data['ticker_sentiment'].get(company_ticker, {
        'cumulative_score': 0,
        'article_count': 0,
        'label_counts': {}
    })
    ticker_sentiment['cumulative_score'] += ticker_cumulative_score
    ticker_sentiment['article_count'] += 1
    ticker_sentiment_label = ticker_data.get('ticker_sentiment_label', 'Neutral')
    ticker_sentiment['label_counts'][ticker_sentiment_label] = ticker_sentiment['label_counts'].get(
        ticker_sentiment_label, 0) + 1
    filtered_data['ticker_sentiment'][company_ticker] = ticker_sentiment


def process_source_and_topic_sentiment(filtered_data, article):
    source = article.get('source', 'Unknown')
    sentiment_score = article.get('overall_sentiment_score', 0)

    source_sentiment = filtered_data['sentiment_by_source'].get(source, {
        'cumulative_score': 0,
        'article_count': 0
    })
    source_sentiment['cumulative_score'] += sentiment_score
    source_sentiment['article_count'] += 1
    filtered_data['sentiment_by_source'][source] = source_sentiment

    for topic_data in article.get('topics', []):
        topic = topic_data.get('topic', 'Unknown')
        relevance_score = float(topic_data.get('relevance_score', 0))
        topic_sentiment = filtered_data['sentiment_by_topic'].get(topic, {
            'cumulative_score': 0,
            'topic_relevance': 0
        })
        topic_sentiment['cumulative_score'] += sentiment_score * relevance_score
        topic_sentiment['topic_relevance'] += relevance_score
        filtered_data['sentiment_by_topic'][topic] = topic_sentiment


def finalize_filtered_data(filtered_data, total_articles, cumulative_sentiment_score, sentiment_label_counts):
    if total_articles > 0:
        filtered_data['overall_sentiment']['average_score'] = cumulative_sentiment_score / total_articles
        filtered_data['overall_sentiment']['label_percentages'] = {label: count / total_articles for label, count in
                                                                   sentiment_label_counts.items()}

        for ticker, ticker_data in filtered_data['ticker_sentiment'].items():
            ticker_data['average_score'] = ticker_data['cumulative_score'] / ticker_data['article_count']
            ticker_data['label_percentages'] = {label: count / ticker_data['article_count'] for label, count in
                                                ticker_data['label_counts'].items()}

        for source_data in filtered_data['sentiment_by_source'].values():
            source_data['average_score'] = source_data['cumulative_score'] / source_data['article_count']
        for topic_data in filtered_data['sentiment_by_topic'].values():
            topic_data['weighted_average_score'] = topic_data['cumulative_score'] / topic_data['topic_relevance']


def fetch_news_sentiment_data(ticker, api_key):
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}'
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None


def generate_data_signature(sentiment_data):
    overall_average_score = sentiment_data.get('overall_sentiment', {}).get('average_score', 0)
    symbol = sentiment_data.get('symbol', '')
    ticker_cumulative_score = sentiment_data.get('ticker_sentiment', {}).get(symbol, {}).get('cumulative_score', 0)
    signature_str = f"{overall_average_score}_{ticker_cumulative_score}_{len(sentiment_data.get('sentiment_by_source', {}))}_{len(sentiment_data.get('sentiment_by_topic', {}))}"
    return hashlib.sha256(signature_str.encode()).hexdigest()


def store_sentiment_data_in_mongo(filtered_sentiment, company_ticker):
    client = pymongo.MongoClient(MONGO_CONN_STRING)
    try:
        db = client['stock_data']
        collection = db['news_sentiment_data']

        if collection.count_documents({}) > 0:
            collection.delete_many({})
            print("Cleared `news_sentiment_data` collection before storing new data.")

        current_signature = generate_data_signature(filtered_sentiment)
        existing_record = collection.find_one({'symbol': company_ticker},
                                              sort=[('datetime_imported', pymongo.DESCENDING)])

        if existing_record:
            existing_signature = generate_data_signature(existing_record)
            if current_signature != existing_signature:
                collection.insert_one(filtered_sentiment)
                print(f"New sentiment data for {company_ticker} stored in MongoDB.")
            else:
                print(f"No significant change in sentiment data for {company_ticker}. No new record inserted.")
        else:
            collection.insert_one(filtered_sentiment)
            print(f"Data for {company_ticker} stored in MongoDB.")
    finally:
        client.close()


def filter_sentiment_data(sentiment_data, company_ticker, start_date, end_date):
    start_date_dt = datetime.strptime(start_date, "%Y%m%d")
    end_date_dt = datetime.strptime(end_date, "%Y%m%d")

    filtered_data = {
        'overall_sentiment': {'average_score': 0, 'label_percentages': {}},
        'ticker_sentiment': {},
        'sentiment_by_source': {},
        'sentiment_by_topic': {},
        'datetime_imported': datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        'symbol': company_ticker
    }

    if 'feed' not in sentiment_data or not sentiment_data['feed']:
        print(f"No news feed data found for {company_ticker} within the specified date range.")
        return filtered_data

    total_articles, cumulative_sentiment_score = 0, 0
    sentiment_label_counts = {}

    for article in sentiment_data['feed']:
        article_date_dt = datetime.strptime(article.get('time_published', '')[:8], "%Y%m%d")
        if start_date_dt <= article_date_dt <= end_date_dt:
            total_articles += 1
            sentiment_score = article.get('overall_sentiment_score', 0)
            sentiment_label = article.get('overall_sentiment_label', 'Neutral')
            cumulative_sentiment_score += sentiment_score
            sentiment_label_counts[sentiment_label] = sentiment_label_counts.get(sentiment_label, 0) + 1

            for ticker_data in article.get('ticker_sentiment', []):
                if ticker_data.get('ticker') == company_ticker:
                    ticker_sentiment_score = float(ticker_data.get('ticker_sentiment_score', 0))
                    relevance_score = float(ticker_data.get('relevance_score', 1))
                    ticker_cumulative_score = ticker_sentiment_score * relevance_score
                    update_ticker_sentiment(filtered_data, company_ticker, ticker_data, ticker_cumulative_score)

            process_source_and_topic_sentiment(filtered_data, article)

    finalize_filtered_data(filtered_data, total_articles, cumulative_sentiment_score, sentiment_label_counts)
    return filtered_data


def fetch_and_store_sentiment(ticker, start_date, end_date, api_key):
    sentiment_data = fetch_news_sentiment_data(ticker, api_key)
    if sentiment_data:
        filtered_sentiment = filter_sentiment_data(sentiment_data, ticker, start_date, end_date)
        store_sentiment_data_in_mongo(filtered_sentiment, ticker)
        return filtered_sentiment
    else:
        print("Failed to fetch sentiment data.")
        return None


def main():
    start_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y%m%d")
    end_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    unique_symbols = fetch_unique_symbols()

    for ticker in unique_symbols:
        print(f"Processing sentiment data for {ticker}...")
        fetch_and_store_sentiment(ticker, start_date, end_date, ALPHA_VANTAGE_API_KEY)


if __name__ == "__main__":
    main()
