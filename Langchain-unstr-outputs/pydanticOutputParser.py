from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age : int = Field(gt = 18, description="Age of the person")
    city : str = Field(description="Name of the city person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = 'Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables = ['place'],
    partial_variables = {'format_instruction': parser.get_format_instructions()} 
)

# prompt = template.invoke({'place':'New York'})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content[0]["text"])

chain = template | model | parser

final_result = chain.invoke({'place':'Indian'})

print(final_result)