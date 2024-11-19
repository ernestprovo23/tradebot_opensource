import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_DB_CONN_STRING")

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client["machinelearning"]

# Initialize list to store results
results = []

# Date fields to check based on sample structure
date_fields = ["date", "date_imported"]

# Iterate through collections in the database
for collection_name in db.list_collection_names():
    collection = db[collection_name]
    # Get total document count
    total_rows = collection.count_documents({})

    # Initialize dictionary for collection summary
    collection_summary = {
        "Collection Name": collection_name,
        "Total Rows": total_rows,
        "Date Column": None,
        "Min Date": None,
        "Max Date": None
    }

    # Check for date fields if documents exist
    if total_rows > 0:
        # Iterate through each possible date field and find min/max dates
        for date_col in date_fields:
            try:
                # Get min and max dates using aggregation for the current date field
                pipeline = [
                    {"$match": {date_col: {"$exists": True, "$type": "date"}}},
                    {"$group": {
                        "_id": None,
                        "min_date": {"$min": f"${date_col}"},
                        "max_date": {"$max": f"${date_col}"}
                    }}
                ]

                # Run aggregation and extract results if available
                date_stats = list(collection.aggregate(pipeline))
                if date_stats:
                    collection_summary["Date Column"] = date_col
                    collection_summary["Min Date"] = date_stats[0]["min_date"]
                    collection_summary["Max Date"] = date_stats[0]["max_date"]
                    break  # Exit loop once a valid date field is found

            except Exception:
                continue  # Skip if aggregation fails for this date field

    # Append collection summary to results list
    results.append(collection_summary)

# Convert results to a DataFrame for easy viewing and export
df_results = pd.DataFrame(results)

# Save the results as a concise CSV file
df_results.to_csv("collection_summary.csv", index=False)

# Display the result for quick inspection
print(df_results)
