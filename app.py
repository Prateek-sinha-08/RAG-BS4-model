import streamlit as st
import subprocess
import os
from scrape import scrape_all_text, save_to_file
from create_database_groqapi import main as create_db_main
from query_data_groqapi import get_answer_from_query
from dotenv import load_dotenv
import os

api_key = st.secrets["api"]["key"]

import sqlite3
print("SQLite version:", sqlite3.sqlite_version)

# Streamlit App
st.set_page_config(page_title="RAG Web QA with GROQ", page_icon="🔎")
st.title("🔎 Website Question Answering with GROQ LLM")

# Input fields
url = st.text_input("Enter Website URL:", placeholder="https://example.com")
query_text = st.text_input("Enter your question:", placeholder="What is this website about?")

# Button
if st.button("Get Answer"):
    if not url or not query_text:
        st.warning("Please enter both URL and question.")
    else:
        with st.spinner('Scraping website...'):
            try:
                # Scrape website and save to file
                scraped_text = scrape_all_text(url)
                os.makedirs("documents", exist_ok=True)
                save_to_file(scraped_text, 'documents/scraped_text.txt')
                st.success("✅ Website scraped and text saved.")
            except Exception as e:
                st.error(f"Error scraping website: {str(e)}")
                st.stop()

        with st.spinner('Building database...'):
            try:
                # Create Chroma DB
                create_db_main()
                st.success("✅ Vector database created.")
            except Exception as e:
                st.error(f"Error creating database: {str(e)}")
                st.stop()

        with st.spinner('Querying the database...'):
            try:
                # Get answer using the query_data_groqapi function
                response = get_answer_from_query(query_text)
                st.write(response)  # Display answer in Streamlit
            except Exception as e:
                st.error(f"Error querying database: {str(e)}")
                st.stop()
