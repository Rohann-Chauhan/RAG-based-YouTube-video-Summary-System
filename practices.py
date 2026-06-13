from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.chains.retrieval_qa.base import RetrievalQA

text = """
RAG means Retrieval Augmented Generation.
It helps AI search documents before answering.
Ollama runs AI models locally on your laptop.
"""

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = splitter.create_documents([text])

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(docs, embeddings)

llm = Ollama(model="llama3.2:3b")

"""qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever()
) 

query = "What is RAG?"
print(qa.run(query)) """