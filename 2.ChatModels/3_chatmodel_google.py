from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
result = model.invoke("What is capital of India")

text = result.content[0]["text"] if isinstance(result.content, list) else result.content
print(text)