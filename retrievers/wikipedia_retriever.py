import wikipedia
from langchain_community.retrievers import WikipediaRetriever

# 1. ADD THIS LINE: Set a custom user agent to prevent Wikipedia from blocking the request
# Replace with any dummy app name and your email
wikipedia.set_user_agent("GeopoliticsResearchApp/1.0 (contact: your-email@example.com)")

# Initialize the retriever (optional: set language and top_k)
retriever = WikipediaRetriever(top_k_results=2, lang="en")

# Define your query
query = "the geopolitical history of india and pakistan from the perspective of a chinese"

# Get relevant Wikipedia documents
docs = retriever.invoke(query)

# Print retrieved content
for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")  # truncate for display