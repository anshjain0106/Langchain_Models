from langchain_community.document_loaders import WebBaseLoader

url = 'https://www.w3schools.com/python/python_generators.asp'

loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs))
print(docs[0].page_content)