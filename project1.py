from youtube_transcript_api import YouTubeTranscriptApi ,TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
print("All import working well")


from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

print("All imports working well")

video_id = "P26AE7NLx4Q"

try:
    api=YouTubeTranscriptApi()
    transcript_list = api.fetch(
        video_id,
        languages=["en"]
    )

    transcript = " ".join(
        chunk.text for chunk in transcript_list
    )

   # print(transcript)

except TranscriptsDisabled:
    print("No caption available for this video")

except Exception as e:
    print("Error:", e)

splitter =RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)   
chunks=splitter.create_documents([transcript])
#print(len(chunks))

# embedding 1c and 1d indexing (embedding generateing)
#embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store=FAISS.from_documents(chunks,embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
#print(retriever.invoke("why Tom Holland Deleted his instagram "))
llm=ChatOllama(
    model="llama3.2:3b",
    temperature=0.2
)
prompt=PromptTemplate(
    template="""
    You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question} """,
      input_variables=['context','question']

)
question="why Tom Holland Deleted his instagram"
retrieved_docs=retriever.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
final_prompt=prompt.invoke({"context":context_text,"question":question})
# generate 
answer=llm.invoke(final_prompt)
print(answer)