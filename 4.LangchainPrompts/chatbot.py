from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit' :
        break
    result = model.invoke(chat_history)
    if isinstance(result.content, list):
        # Extract the text from the first block
        text_output = result.content[0].get("text", "")
        chat_history.append(AIMessage(content=text_output))
        print("AI: ", text_output)
    else:
        chat_history.append(AIMessage(content=result.content))
        print("AI:", result.content)

print(chat_history)