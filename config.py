import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "llama-3.3-70b-versatile"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

VECTOR_DB_PATH = "vectorstore"