from pymongo import MongoClient, DESCENDING
from datetime import datetime, timedelta
import logging

class MongoManager:
    def __init__(self, conn_string, db_name):
        self.client = MongoClient(conn_string)
        self.db = self.client[db_name]
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_index(self, collection_name, field, index_name):
        collection = self.db[collection_name]
        collection.create_index([(field, DESCENDING)], name=index_name)

    def insert_with_deduplication(self, collection_name, documents, unique_field='Symbol'):
        collection = self.db[collection_name]
        for document in documents:
            ticker = document[unique_field]  # Ensure this matches the key for your ticker in the document
            date_cutoff = datetime.utcnow() - timedelta(days=30)

            # Find the most recent document for the given ticker within the last 30 days
            most_recent_doc = collection.find_one({
                unique_field: ticker,
                'date_added': {'$gte': date_cutoff}
            }, sort=[('date_added', DESCENDING)])

            if most_recent_doc:
                # Calculate the time difference between the new document and the most recent document
                time_difference = document['date_added'] - most_recent_doc['date_added']

                # Replace the document only if the new one is at least 24 hours newer
                if time_difference.total_seconds() >= 86400:  # 86400 seconds in 24 hours
                    collection.delete_one({'_id': most_recent_doc['_id']})
                    try:
                        result = collection.insert_one(document)
                        self.logger.info(f"Replaced document for {unique_field} {ticker} with a newer one.")
                    except Exception as e:
                        self.logger.error(f"Error replacing document for {unique_field} {ticker}: {e}")
                else:
                    self.logger.info(
                        f"Found a recent document for {unique_field} {ticker} that is less than 24 hours old. Skipping replacement.")
            else:
                # If there's no recent document for the ticker, insert the new document
                try:
                    result = collection.insert_one(document)
                    self.logger.info(
                        f"Inserted new document for {unique_field} {ticker} into MongoDB collection '{collection_name}'.")
                except Exception as e:
                    self.logger.error(f"Error inserting document for {unique_field} {ticker}: {e}")
