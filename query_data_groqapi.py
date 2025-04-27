# query_data_groqapi.py
import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Path where Chroma DB is stored
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def get_answer_from_query(query_text: str):
    # Prepare the DB
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.3:
        return "Unable to find matching results."

    # Prepare context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Prepare prompt for Groq
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Connect to Groq
    model = ChatGroq(
        api_key=st.secrets["api"]["key"],
        model_name="llama3-70b-8192"  # Use your specific model name here
    )

    # Predict the response
    response_text = model.invoke(prompt)

    # Prepare the response
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text.content}\nSources: {sources}"
    return formatted_response

# import os
# import argparse
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq  # New import
# from langchain.prompts import ChatPromptTemplate
# from dotenv import load_dotenv

# # Load environment variables from .env
# load_dotenv()

# # Path where Chroma DB is stored
# CHROMA_PATH = "chroma"

# # Custom prompt template
# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """

# def main():
#     # Create CLI
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     args = parser.parse_args()
#     query_text = args.query_text

#     # Prepare the DB
#     embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#     # Search the DB
#     results = db.similarity_search_with_relevance_scores(query_text, k=3)
#     if len(results) == 0 or results[0][1] < 0.3:
#         print(f"Unable to find matching results.")
#         return

#     # Prepare context
#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)

#     print("\nGenerated Prompt to LLM:\n")
#     print(prompt)
#     print("\n------------------------\n")

#     # Connect to Groq
#     model = ChatGroq(
#         api_key=os.getenv("GROQ_API_KEY"),
#         model_name="llama3-70b-8192"  # ✅ Updated model name here
#     )

#     # Predict the response
#     response_text = model.invoke(prompt)

#     # Prepare output
#     sources = [doc.metadata.get("source", None) for doc, _score in results]
#     formatted_response = f"Response: {response_text.content}\nSources: {sources}"

#     print(formatted_response)

# if __name__ == "__main__":
#     main()
