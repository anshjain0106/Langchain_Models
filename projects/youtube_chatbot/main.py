from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Step 1a - Indexing (Document Ingestion)
video_id = "Gfr50f6ZBvo" # only the ID, not full URL
try:
    # FIX: Add parentheses to instantiate the class, and use .fetch()
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])

    # Flatten it to plain text (this remains exactly the same)
    transcript = " ".join(chunk.text for chunk in transcript_list)
    # print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

# Step 1b - Indexing (Text Splitting)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
# print(len(chunks))
# print(chunks[0])

# Step 1c and 1d - Indexing (Embedding Generation and Storing in Vector Store)
# embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store =  FAISS.from_documents(chunks, embeddings)
# print(vector_store.index_to_docstore_id)