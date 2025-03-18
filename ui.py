import streamlit as st
from backend import call_llm, process_query, reset_database, get_vector_collection

async def run_ui():
    st.set_page_config(page_title="LLM with Web Search")
    st.header("🔍 LLM Web Search")

    prompt = st.text_area(
        label="Put your query here",
        placeholder="Add your query...",
        label_visibility="hidden",
    )
    go = st.button("⚡️ Go")
    reset_db = st.button("Reset Database")

    _, chroma_client = get_vector_collection()

    if reset_db:
        reset_database(chroma_client)
        st.write("Database reset successfully.")
        _, chroma_client = get_vector_collection()

    if prompt and go:
        context, needs_web_search, _, _ = await process_query(prompt)
        
        if context is None and not needs_web_search:
            llm_response = call_llm(prompt=prompt, require_search=False)
            st.write_stream(llm_response)
        else:
            if context in ["No results found.", "Failed to retrieve sufficient context after crawling."]:
                st.write(context)
            elif context is None:
                st.write("No sufficient context found in database.")
            else:
                llm_response = call_llm(context=context, prompt=prompt, require_search=True)
                st.write_stream(llm_response)
                