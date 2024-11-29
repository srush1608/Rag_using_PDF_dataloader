from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from database.database import store_embeddings_pgvector
from langchain_community.vectorstores import PGVector

def load_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return [doc.page_content for doc in documents]
    except Exception as e:
        print(f"Error processing PDF file {file_path}: {e}")
        return []

def chunk_texts(documents, chunk_size=500, overlap=50):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for doc in documents:
        chunks.extend(splitter.split_text(doc))
    return chunks

def similarity_search(query, vectorstore):
    try:
        results = vectorstore.similarity_search(
            query=query,
            k=5
        )
        relevant_chunks = [result.page_content for result in results]
        return relevant_chunks
    except Exception as e:
        print(f"Error performing similarity search: {e}")
        return []
