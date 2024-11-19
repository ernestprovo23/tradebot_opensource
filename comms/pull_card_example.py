import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_DB_CONN_STRING = os.getenv("MONGO_DB_CONN_STRING")
DB_NAME = "stock_data"

mongo_client = MongoClient(MONGO_DB_CONN_STRING)
db = mongo_client[DB_NAME]


def inspect_documents():
    documents = list(db.openai_analysis.find().limit(5))
    for doc in documents:
        print(f"Symbol: {doc['symbol']}")
        print("Content:")
        content = doc['content']

        # Extract and print each section
        for section in ['Analysis', 'Numbers', 'Justification']:
            start = content.find(f'"{section}":')
            if start != -1:
                end = content.find('"}', start)
                if end == -1:
                    end = content.find('},', start)
                if end == -1:
                    end = len(content)
                print(f"\n{section}:")
                print(content[start:end].strip())

        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    inspect_documents()