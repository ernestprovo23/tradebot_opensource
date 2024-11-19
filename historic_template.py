from azure_storage_operations import connect_to_storage_account, list_containers, upload_blob, download_blob
from s3connector import azure_connection_string

def main(container_name):
    # Connect to the storage account
    blob_service_client = connect_to_storage_account(azure_connection_string)

    # List containers
    list_containers(blob_service_client)

    # Upload a blob
    blob_name = "example_blob.txt"
    file_path = "local_file.txt"
    upload_blob(blob_service_client, container_name, blob_name, file_path)

    # Download a blob
    download_file_path = "downloaded_file.txt"
    download_blob(blob_service_client, container_name, blob_name, download_file_path)

if __name__ == "__main__":
    container_name = "your-container-name"  # Replace with the actual container name

    main(container_name)
