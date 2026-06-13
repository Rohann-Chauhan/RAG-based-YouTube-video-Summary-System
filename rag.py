from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("A - Script Started")

try:
    # ----------------------------------
    # Step 1: Create Sample Document
    # ----------------------------------
    text = """
    RAG means Retrieval Augmented Generation.
    It uses external documents to answer questions.
    """

    print("B - Text Created")

    # ----------------------------------
    # Step 2: Split Text Into Chunks
    # ----------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10
    )

    docs = splitter.create_documents([text])

    print("C - Documents Created")

    # ----------------------------------
    # Step 3: Load Embedding Model
    # ----------------------------------
    print("D - Loading Embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("E - Embeddings Loaded")

    # ----------------------------------
    # Step 4: Create Vector Database
    # ----------------------------------
    print("F - Creating Chroma DB...")

    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings
    )

    print("G - Chroma DB Created")

    # ----------------------------------
    # Step 5: Create Retriever
    # ----------------------------------
    retriever = db.as_retriever()

    print("H - Retriever Ready")

    # ----------------------------------
    # Step 6: Connect Ollama
    # ----------------------------------
    llm = OllamaLLM(
        model="llama3.2:3b"
    )

    print("I - Ollama Connected")

    # ----------------------------------
    # Step 7: Ask Question
    # ----------------------------------
    query = "What is RAG?"

    print("J - Query Created")
    print("Question:", query)

    # ----------------------------------
    # Step 8: Retrieve Documents
    # ----------------------------------
    retrieved_docs = retriever.invoke(query)

    print("K - Documents Retrieved")

    for i, doc in enumerate(retrieved_docs):
        print(f"\nDocument {i+1}:")
        print(doc.page_content)

    # ----------------------------------
    # Step 9: Build Context
    # ----------------------------------
    context = "\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    print("\nL - Context Built")
    print(context)

    # ----------------------------------
    # Step 10: Create Prompt
    # ----------------------------------
    final_prompt = f"""
You are a helpful assistant.

Context:
{context}

Question:
{query}

Answer clearly:
"""

    print("\nM - Sending Prompt To Ollama...")

    # ----------------------------------
    # Step 11: Generate Response
    # ----------------------------------
    result = llm.invoke(final_prompt)

    print("\nN - Ollama Response Received")

    # ----------------------------------
    # Step 12: Print Answer
    # ----------------------------------
    print("\n====================")
    print("FINAL ANSWER")
    print("====================\n")

    print(result)

except Exception as e:
    print("\nERROR OCCURRED")
    print("Error Type:", type(e).__name__)
    print("Error Message:", str(e))