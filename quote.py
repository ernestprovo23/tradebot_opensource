import os
from dotenv import load_dotenv

# Assuming your .env file is in the same directory as your script
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

print("API Key:", os.getenv('ALPHA_VANTAGE_API'))
