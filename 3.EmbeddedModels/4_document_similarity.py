from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# We no longer strictly need load_dotenv() for an API key if running locally, 
# but you can leave it if you are using other cloud services in your app.

# 1. Initialize the free local embedding model
# The first time you run this, it will take a few seconds to download the model files.
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'Tell me about virat kohli'

# 2. Generate embeddings (This now happens locally on your CPU/GPU)
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# 3. Calculate similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(f"Query: {query}")
print(f"Most similar document: {documents[index]}")
print(f"Similarity score: {score:.4f}")