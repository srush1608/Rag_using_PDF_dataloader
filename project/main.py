import os
from dotenv import load_dotenv
from retriever.retriever import load_pdf, chunk_texts, similarity_search
from database.database import store_embeddings_pgvector
from embeddings.embeddings import generate_embeddings
from prompts.prompts import create_prompt
from langchain_community.vectorstores.pgvector import PGVector
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DB_CONNECTION_URL_2 = os.getenv("DB_CONNECTION_URL_2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embedding_model = GoogleGenerativeAIEmbeddings(
    api_key=GOOGLE_API_KEY,
    model="models/embedding-001" 
)

vectorstore = PGVector(
    connection_string=DB_CONNECTION_URL_2,
    embedding_function=embedding_model,
)

def generate_llm_response(query, retrieved_chunks):
    context = ' '.join(retrieved_chunks)
    prompt = create_prompt()

    chatgroq = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama3-8b-8192",
        temperature=0.0,
        max_retries=2
    )
    
    chain = prompt | chatgroq
    return chain.invoke({"context": retrieved_chunks, "query": query}).content

def main():
    try:
        pdf_file = "C:/Users/Coditas-Admin/Desktop/rag_pdf/project/attention.pdf"
        pdf_texts = load_pdf(pdf_file)

        chunks = chunk_texts(pdf_texts)

        store_embeddings_pgvector(chunks)

        while True:
            user_query = input("Enter your query (type 'exit' to stop): ")
            if user_query.lower() == "exit":
                print("Exiting the program...")
                break

            retriver = vectorstore.as_retriever(search_kwargs={"k": 5})
            results = retriver.invoke(user_query)


            # results = vectorstore.similarity_search(
            #     query=user_query,
            #     k=5
            # )

            relevant_chunks = [result.page_content for result in results]


            response = generate_llm_response(user_query, relevant_chunks)
            print("\nFinal Response:\n", response)

    except Exception as e:
        print(f"Error in processing: {e}")

if __name__ == "__main__":
    main()







































# import os
# from dotenv import load_dotenv
# from retriever.retriever import load_pdf, chunk_texts, similarity_search
# from database.database import store_embeddings_pgvector
# from embeddings.embeddings import generate_embeddings
# from prompts.prompts import create_prompt
# from langchain_community.vectorstores import PGVector
# from langchain_groq import ChatGroq

# load_dotenv()

# DB_CONNECTION_URL_2 = os.getenv("DB_CONNECTION_URL_2")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# vectorstore = PGVector(
#     connection_string=DB_CONNECTION_URL_2,
#     embedding_function=generate_embeddings,
# )

# def generate_llm_response(query, retrieved_chunks):
#     context = ' '.join(retrieved_chunks)
#     prompt = create_prompt(context, query)

#     chatgroq = ChatGroq(
#         api_key=GROQ_API_KEY,
#         model="llama3-8b-8192",
#         temperature=0.0,
#         max_retries=2
#     )
    
#     response = chatgroq.generate(prompt)
#     return response

# def main():
#     try:
#         pdf_file = "attention.pdf"
#         pdf_texts = load_pdf(pdf_file)

#         chunks = chunk_texts(pdf_texts)

#         store_embeddings_pgvector(chunks, table_name="langchain_pg_embedding")

#         while True:
#             user_query = input("Enter your query (type 'exit' to stop): ")
#             if user_query.lower() == "exit":
#                 print("Exiting the program...")
#                 break

#             retrieved_chunks = similarity_search(user_query, vectorstore)

#             response = generate_llm_response(user_query, retrieved_chunks)
#             print("\nFinal Response:\n", response)

#     except Exception as e:
#         print(f"Error in processing: {e}")

# if __name__ == "__main__":
#     main()
