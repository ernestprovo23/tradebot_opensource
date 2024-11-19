from azure.storage.blob import BlobServiceClient
from s3connector import azure_connection_string
from azure.core.exceptions import AzureError
from io import StringIO
import pandas as pd

def download_blob_file(container_name, file_name):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(file_name)
        data = blob_client.download_blob().readall().decode('utf-8')
        return data
    except AzureError as e:
        print(f"Error retrieving file from Azure Blob Storage: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def count_records_in_csv(csv_data):
    try:
        data = StringIO(csv_data)
        df = pd.read_csv(data)
        num_records = len(df)
        return num_records
    except Exception as e:
        print(f"An error occurred while counting records in the CSV file: {e}")
        return None

# Specify the container name and file name
container_name = 'historic'  # Update with your container name
file_name = 'company_overviews.csv'  # Update with your file name

csv_data = download_blob_file(container_name, file_name)
num_records = count_records_in_csv(csv_data)

print(f"The number of records in the CSV file is: {num_records}")
