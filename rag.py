"""
rag.py

Core RAG pipeline for LogSense AI.

Flow:
    logs -> chunks -> embeddings -> FAISS -> retrieve -> LLM answer
"""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_groq import ChatGroq

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from config import (
    GROQ_API_KEY,
    MODEL_NAME,
    EMBEDDING_MODEL,
    VECTOR_DB_PATH,
)


# ---------------------------------------------------
# Embedding model (turns text into vectors)
# ---------------------------------------------------

embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


# ---------------------------------------------------
# LLM (answers questions using retrieved context)
# ---------------------------------------------------

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=0,
)


# ---------------------------------------------------
# Prompt template
# ---------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a senior DevOps engineer.

Answer ONLY using the provided logs.
If the answer isn't in the logs, say
"I cannot find enough evidence in the logs."

Context:
{context}

Question:
{input}

Return your answer in this format:

Root Cause:
...

Severity:
...

Evidence:
...

Suggested Fix:
...

Prevention:
...
"""
)


# ---------------------------------------------------
# Text splitter (breaks long logs into small chunks)
# ---------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


# ---------------------------------------------------
# Vector store (in-memory reference + on-disk FAISS index)
# ---------------------------------------------------

vector_db = None


def load_vectorstore():
    """Load an existing FAISS index from disk, if one exists."""

    global vector_db

    db_path = Path(VECTOR_DB_PATH)

    if db_path.exists():
        try:
            vector_db = FAISS.load_local(
                VECTOR_DB_PATH,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            print(f"[rag] Could not load existing index: {e}")
            vector_db = None

    return vector_db


# Load whatever was indexed in a previous run
load_vectorstore()


# ---------------------------------------------------
# Ingest logs -> chunks -> embeddings -> FAISS
# ---------------------------------------------------

def ingest_logs(log_text: str) -> int:
    """
    Splits raw log text into chunks and adds them to the FAISS
    index. New logs are appended to the existing index rather
    than replacing it.
    """

    global vector_db

    docs = [Document(page_content=log_text)]
    chunks = text_splitter.split_documents(docs)

    if vector_db is None:
        vector_db = FAISS.from_documents(chunks, embeddings)
    else:
        vector_db.add_documents(chunks)

    vector_db.save_local(VECTOR_DB_PATH)

    return len(chunks)


# ---------------------------------------------------
# Retriever (finds the most relevant chunks for a question)
# ---------------------------------------------------

def get_retriever():

    if vector_db is None:
        raise ValueError("No logs have been indexed.")

    return vector_db.as_retriever(search_kwargs={"k": 5})


# ---------------------------------------------------
# Ask a question -> retrieve context -> LLM answer
# ---------------------------------------------------

def ask(question: str) -> str:

    retriever = get_retriever()

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": question})

    return response["answer"]