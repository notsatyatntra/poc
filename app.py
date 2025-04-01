import streamlit as st
import asyncio
from backend.llm import get_llm, call_llm
from backend.database import get_vectorstore, has_relevant_data, get_relevant_context, add_to_vector_database
from backend.search import search_and_crawl
from backend.utils import needs_web_search

# Cache the vector store and LLM to avoid reinitialization
@st.cache_resource
def load_vectorstore():
    return get_vectorstore()

@st.cache_resource
def load_llm():
    return get_llm()
    
# Initialize cached objects
vectorstore = load_vectorstore()
llm = load_llm()

# Streamlit UI
st.header("🔍 LLM Web Search")
prompt = st.text_area("Put your query here", placeholder="Add your query...", label_visibility="hidden")
go = st.button("⚡️ Go")

if prompt and go:
    if needs_web_search(prompt):
        if not has_relevant_data(vectorstore, prompt):
            # Run async search and crawl synchronously
            documents = asyncio.run(search_and_crawl(prompt))
            add_to_vector_database(vectorstore, documents)
        context_list = get_relevant_context(vectorstore, prompt)
        context = "\n\n".join(context_list) if context_list else None
        response = call_llm(llm, prompt, context=context)
    else:
        response = call_llm(llm, prompt)
    st.write_stream(response)

    