from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

chat_history = []

while True:
    user_input = input('You: ')
    chat_history.append(user_input)
    if user_input == 'exit' :
        break
    result = model.invoke(chat_history)
    if isinstance(result.content, list):
        # Extract the text from the first block
        text_output = result.content[0].get("text", "")
        chat_history.append(text_output)
        print("AI: ", text_output)
    else:
        chat_history.append(result.content)
        print("AI:", result.content)

print(chat_history)