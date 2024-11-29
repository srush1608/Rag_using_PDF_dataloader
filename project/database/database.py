
import os
from dotenv import load_dotenv
from langchain_community.vectorstores.pgvector import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

DB_CONNECTION_URL_2 = os.getenv("DB_CONNECTION_URL_2")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def validate_env_variables():
    if not DB_CONNECTION_URL_2:
        raise ValueError("Database connection URL is missing!")
    print("Database connection URL loaded successfully.")

    if not GOOGLE_API_KEY:
        raise ValueError("Google API key is missing!")
    print("Google API key loaded successfully.")

validate_env_variables()

def store_embeddings_pgvector(texts):
    try:
        embeddings_generator = GoogleGenerativeAIEmbeddings(
            api_key=GOOGLE_API_KEY,
            model="models/embedding-001"
        )

        vectorstore = PGVector(
            connection_string=DB_CONNECTION_URL_2,
            embedding_function=embeddings_generator,
        )

        vectorstore.add_texts(texts)
        # print(f"Data successfully stored in the '{table_name}' table.")
    except Exception as e:
        print(f"Error storing embeddings in PGVector: {e}")
