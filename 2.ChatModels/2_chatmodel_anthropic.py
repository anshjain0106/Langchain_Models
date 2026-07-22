from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

model = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

result = model.invoke("What is capital of France?")

print(result.content)