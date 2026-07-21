from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

model = ChatOpenAI(
    model="gpt-4o",
    base_url="https://models.github.ai/inference",
    api_key=os.getenv("GITHUB_PAT_TOKEN"),
    # temperature=0
    temperature=1.6,
    max_completion_tokens=50
)

result = model.invoke("Suggest me 2 creative ideas for a birthday gift")

print(result.content)