from pymongo import MongoClient, UpdateOne
from bson.objectid import ObjectId
from datetime import datetime
import pytz

# Source MongoDB (On-prem)
source_uri = "mongodb://192.168.88.27:27018/"
source_client = MongoClient(source_uri)

# Destination MongoDB (Cloud - Atlas)
dest_uri = "mongodb+srv://ernestprovo:7xTxCRd2CKSL3gte@dseprod1.an2tliw.mongodb.net/?retryWrites=true&w=majority&appName=dseprod1"
dest_client = MongoClient(dest_uri)

# Specify the timezone (adjust as needed)
timezone = pytz.timezone('UTC')

# Function to get the last migration time from a dedicated collection
def get_last_migration_time(db, collection_name):
    migration_collection = db['migration_metadata']
    record = migration_collection.find_one({'collection': collection_name})
    if record:
        last_migration_time = record['last_migration_time']
        if last_migration_time.tzinfo is None:
            # Assume UTC if timezone information is missing
            last_migration_time = last_migration_time.replace(tzinfo=pytz.UTC)
        return last_migration_time
    else:
        # If no record exists, return None
        return None

# Function to update the last migration time
def update_last_migration_time(db, collection_name, time):
    # Ensure time is timezone-aware
    if time.tzinfo is None:
        time = time.replace(tzinfo=pytz.UTC)
    migration_collection = db['migration_metadata']
    migration_collection.update_one(
        {'collection': collection_name},
        {'$set': {'last_migration_time': time}},
        upsert=True
    )

# Batch size for bulk operations
BATCH_SIZE = 1000  # Adjust as needed based on your resources

# Get all databases from source MongoDB
databases = source_client.list_database_names()

# Iterate through databases in the source
for db_name in databases:
    if db_name in ['admin', 'local', 'config']:  # Optionally skip system databases
        continue

    source_db = source_client[db_name]
    dest_db = dest_client[db_name]

    # Get all collections in the current database
    collections = source_db.list_collection_names()
    for coll_name in collections:
        source_coll = source_db[coll_name]
        dest_coll = dest_db[coll_name]

        print(f"Processing collection: {db_name}.{coll_name}")

        # Get the last migration time
        last_migration_time = get_last_migration_time(dest_db, coll_name)
        print(f"Last migration time for {db_name}.{coll_name}: {last_migration_time}")

        # Convert last_migration_time to ObjectId
        if last_migration_time is None:
            # For the initial run, use the minimal ObjectId
            last_migration_oid = ObjectId('000000000000000000000000')
        else:
            try:
                # Ensure the timestamp is within a valid range
                timestamp = int(last_migration_time.timestamp())
                if timestamp < 0 or timestamp > 4294967295:
                    raise ValueError("Timestamp out of valid range for ObjectId")
                last_migration_oid = ObjectId.from_datetime(last_migration_time)
            except Exception as e:
                print(f"Error converting last_migration_time to ObjectId: {e}")
                # Use minimal ObjectId as fallback
                last_migration_oid = ObjectId('000000000000000000000000')

        # Find documents inserted since the last migration time
        query = {'_id': {'$gt': last_migration_oid}}
        cursor = source_coll.find(query)

        operations = []
        count = 0
        latest_migration_time = last_migration_time

        # Ensure latest_migration_time is timezone-aware
        if latest_migration_time is not None and latest_migration_time.tzinfo is None:
            latest_migration_time = latest_migration_time.replace(tzinfo=pytz.UTC)

        for doc in cursor:
            # Add 'date_ingested' field with current datetime
            doc['date_ingested'] = datetime.now(timezone)

            # Prepare the upsert operation
            operation = UpdateOne(
                {'_id': doc['_id']},
                {'$set': doc},
                upsert=True
            )
            operations.append(operation)
            count += 1

            # Convert the doc_generation_time to UTC-aware for comparison
            doc_generation_time = doc['_id'].generation_time.replace(tzinfo=pytz.UTC)

            # Update latest_migration_time
            if latest_migration_time is None or doc_generation_time > latest_migration_time:
                latest_migration_time = doc_generation_time

            # Execute batch when batch size is reached
            if len(operations) >= BATCH_SIZE:
                dest_coll.bulk_write(operations, ordered=False)
                operations = []
                print(f"Processed {count} documents so far...")

        # Execute any remaining operations
        if operations:
            dest_coll.bulk_write(operations, ordered=False)
            print(f"Processed {count} documents in total.")

        if latest_migration_time:
            # Update the last migration time to the latest _id timestamp in the source collection
            update_last_migration_time(dest_db, coll_name, latest_migration_time)
            print(f"Updated last migration time for {db_name}.{coll_name} to {latest_migration_time}")
        else:
            print(f"No new documents to insert/update in {db_name}.{coll_name}")

print("Incremental migration complete.")
