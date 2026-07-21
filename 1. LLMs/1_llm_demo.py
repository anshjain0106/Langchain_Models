import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()  

# Initialize the model with GitHub's endpoint and your PAT
llm = ChatOpenAI(
    model="gpt-4o", 
    base_url="https://models.github.ai/inference",
    api_key=os.getenv("GITHUB_PAT_TOKEN")
)

# Invoke the model
result = llm.invoke("What is the capital of India?")

# ChatOpenAI returns an AIMessage object, so we print .content to get just the text
print(result.content)