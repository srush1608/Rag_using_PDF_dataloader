from langchain_core.prompts.prompt import PromptTemplate

def create_prompt():
    """
    This function formats the prompt to be used with the LLM (Groq API).
    """

    # prompt = (
    #     "You are an assistant for question-answering tasks. Use ONLY the retrieved context below "
    #     "to answer the question in detail. Do not use external knowledge or guess. If the context is insufficient, "
    #     "respond with 'I don't know.'\n\n"
    #     "Retrieved context:\n{context}\n"
    #     "Question: {query}\n"
    #     "Answer:"
    # )
    prompt = PromptTemplate.from_template(
        """
        You are an assistant for question-answering tasks.
        Use ONLY the retrieved context below to answer the question in detail. Do not use external knowledge or guess. If the context is insufficient, respond with 'I don't know.'.
        Retrieved context: {context}
        Question: {query}
        Answer: 
        """
    )

    return prompt
