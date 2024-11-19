from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv
import logging

# Create a logger
logging.basicConfig(filename='logfile.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


# Load environment variables from .env file
load_dotenv()

# Load Azure Storage connection string from environment variables
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = "historic"

def delete_all_blobs_in_container(connection_string, container_name):
    try:
        # Create a BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        # Get a reference to the container
        container_client = blob_service_client.get_container_client(container_name)

        # List all blobs in the container and delete them
        blob_list = container_client.list_blobs()
        for blob in blob_list:
            logging.info(f"Deleting blob: {blob.name}")
            container_client.delete_blob(blob.name)

        logging.info("All blobs in the container have been deleted.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

# Call the function to delete all blobs
delete_all_blobs_in_container(AZURE_CONNECTION_STRING, CONTAINER_NAME)
