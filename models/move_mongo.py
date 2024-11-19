import os
import logging
from pymongo import MongoClient, errors
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(filename='data_transfer.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables and establish MongoDB connections
load_dotenv()
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Source MongoDB connection string
source_uri = MONGO_DB_CONN_STRING
# Destination MongoDB connection string
destination_uri = "mongodb+srv://ernestprovo:7xTxCRd2CKSL3gte@dseprod1.an2tliw.mongodb.net/?retryWrites=true&w=majority&appName=dseprod1"

# Connect to the source MongoDB
source_client = MongoClient(source_uri)
source_db_names = source_client.list_database_names()
logging.info("Connected to source MongoDB. Databases: %s", source_db_names)

# Connect to the destination MongoDB
destination_client = MongoClient(destination_uri)
logging.info("Connected to destination MongoDB.")

# Drop the databases in the destination server if they exist
for db_name in source_db_names:
    if db_name in ["admin", "local", "config"]:
        continue
    try:
        logging.info("Dropping database %s from destination MongoDB", db_name)
        destination_client.drop_database(db_name)
    except errors.OperationFailure as e:
        logging.error("Failed to drop database %s: %s", db_name, e)

# Copy data from source to destination
for db_name in source_db_names:
    if db_name in ["admin", "local", "config"]:
        continue

    source_db = source_client[db_name]
    dest_db = destination_client[db_name]

    logging.info("Copying database %s", db_name)

    for collection_name in source_db.list_collection_names():
        source_collection = source_db[collection_name]
        dest_collection = dest_db[collection_name]

        logging.info("Copying collection %s from database %s", collection_name, db_name)

        # Copy all documents from the source collection to the destination collection
        documents = list(source_collection.find())
        doc_count = len(documents)
        if doc_count > 0:
            try:
                dest_collection.insert_many(documents)
                logging.info("Copied %d documents to collection %s in database %s", doc_count, collection_name, db_name)
            except errors.BulkWriteError as bwe:
                logging.error("Bulk write error occurred while copying collection %s from database %s: %s", collection_name, db_name, bwe.details)
        else:
            logging.info("No documents to copy in collection %s from database %s", collection_name, db_name)

logging.info("Data transfer completed.")
