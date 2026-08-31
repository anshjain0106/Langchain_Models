from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

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

# Step 2 - Retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
# print(retriever)
# Query
# retriever.invoke('What is Deepmind')
# print(retriever.invoke('What is Deepmind'))

# Step - 3 Augmentation
llm = ChatGoogleGenerativeAI(model = "gemini-3.6-flash", temperature=0.2)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
# retrieved_docs    = retriever.invoke(question)
# print(retrieved_docs)

# Joining the page content of document
# context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
# print(context_text)

# Final Prompt
# final_prompt = prompt.invoke({"context": context_text, "question": question})
# print(final_prompt)

# Step 4 - Generation
# answer = llm.invoke(final_prompt)
# content = answer.content
# text = content[0]["text"] if isinstance(content, list) else content
# print(text)

# Building a chain

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parallel_chain.invoke('who is Demis')

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

print(main_chain.invoke('Can you summarize the video'))