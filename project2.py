from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

print("All imports working well")

video_id = "P26AE7NLx4Q"
transcript = ""

try:
    api = YouTubeTranscriptApi()

    transcript_list = api.fetch(
        video_id,
        languages=["en"]
    )

    transcript = " ".join(
        chunk.text for chunk in transcript_list
    )

    print("Transcript loaded successfully")

except TranscriptsDisabled:
    print("No caption available for this video")

except Exception as e:
    print("Error:", e)


# Split transcript into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.create_documents([transcript])

print("Total Chunks:", len(chunks))


# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Vector Store
vector_store = FAISS.from_documents(
    chunks,
    embeddings
)


# Retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# Function to convert documents into text
def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# Parallel Chain
parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})


# Prompt
prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer ONLY from the provided transcript context.

If the answer is not available in the context,
just say "I don't know".

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)


# Ollama Model
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.2
)


# Output Parser
parser = StrOutputParser()


# Main Chain
main_chain = (
    parallel_chain
    | prompt
    | llm
    | parser
)


# Chat Loop
while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    answer = main_chain.invoke(question)

    print("\nAnswer:")
    print(answer)