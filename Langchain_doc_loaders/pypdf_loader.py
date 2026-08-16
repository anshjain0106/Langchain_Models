import warnings

# Suppress the community deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

print(docs)
print(len(docs))
print(docs[0].page_content)