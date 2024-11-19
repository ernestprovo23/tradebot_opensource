import anthropic
from dotenv import load_dotenv
from datetime import datetime
import os

load_dotenv()

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=os.getenv('ANTHROPIC_API_KEY'),
)

message = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=1000,
    temperature=0.0,
    system="You are a quantitative analysis engineer working for the world's best finance firm.",
    messages=[
        {"role": "user", "content": "What's the most concerning piece about the current world economy?"}
    ]
)

print(message)
