# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# llm = HuggingFaceEndpoint(
#                           repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#                           task='text-generation'
# )

# model = ChatHuggingFace(llm = llm)

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# 1st Prompt (Detailed report)
template1 = PromptTemplate(
                           template = 'Write a detailed report on {topic}',
                           input_variables = ['topic']
)


# 2nd Prompt (Summary report)
template2 = PromptTemplate(
                           template = 'Write a 5 line summary report on following text. /n {text}',
                           input_variables = ['text']
)

# Parser
parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic' : 'Black Hole'})

print(result)