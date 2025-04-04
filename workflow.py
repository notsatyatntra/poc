from backend_langgraph.search import search_and_crawl
from backend_langgraph.db import get_vectorstore, get_relevant_context, add_to_vector_database, has_relevant_data
from backend_langgraph.llm import call_llm
from backend_langgraph.util import few_shot_needs_web_search, is_website_request, needs_context

from langgraph.graph import START, END, StateGraph
from typing import List, Optional, Tuple
from typing_extensions import TypedDict

class AgentState(TypedDict):
    prompt: str
    vectorstore: Optional[Tuple]
    is_website: Optional[bool]
    url: Optional[List[str]]
    is_generic: Optional[bool]
    needs_web: Optional[bool]
    context_exists: Optional[bool]
    context: Optional[str]
    documents: Optional[List[str]]
    response: Optional[str]
    llm: Optional[object]
    web_search_count: Optional[int]

def initialize(state: AgentState) -> AgentState:
    print("Initializing agent state...")
    state['vectorstore'] = get_vectorstore()
    #can also call the LLM thread
    return state

def check_prompt(state: AgentState) -> AgentState:
    print(f"Prompt received", {state['prompt']})
    return state

def check_link(state: AgentState) -> AgentState:
    state['url'], state['is_website'] = is_website_request(state['prompt'])
    print(f"URL received", state['url'])
    return state

def check_generic(state: AgentState) -> AgentState:
    state['is_generic'] = needs_context(state['prompt'])
    print(f"Is the prompt generic:", {state['is_generic']})
    return state

def check_needs_web_search(state: AgentState) -> AgentState:
    state['needs_web'] = few_shot_needs_web_search(state['prompt'])
    print(f"Does the prompt needs to be web searched:", {state['needs_web']})
    return state

def check_context_in_db(state: AgentState) -> AgentState:
    state['context_exists'] = has_relevant_data(state['vectorstore'], state['prompt'], state['web_search_count'])
    print(f"Context exists in DB:", {state['context_exists']})
    return state

def retrieve_context(state: AgentState) -> AgentState:
    context_list = get_relevant_context(state['vectorstore'], state['prompt'])
    state['context'] = "\n\n".join(context_list) if context_list else None
    print(f"Retrieved context:", {state['context']})
    return state

async def perform_web_search(state: AgentState) -> AgentState:
    state['documents'] = await search_and_crawl(state['prompt'])
    state['web_search_count'] += 1
    print(f"Web search documents:", state['documents'])
    return state

def add_documents_to_db(state: AgentState) -> AgentState:
    add_to_vector_database(state['vectorstore'], state['documents'])
    print("Documents added to DB")
    return state

def call_llm_state(state: AgentState) -> AgentState:
    state['response'] = call_llm(state['prompt'], state.get('context', ''))
    print(f"LLM Response:", {state['response']})
    return state

def finish(state: AgentState) -> AgentState:
    print("Final state reached.")
    return state

workflow = StateGraph(AgentState)
# Add nodes
workflow.add_node("get_vector_db", initialize)
workflow.add_node("check_prompt", check_prompt)
workflow.add_node("check_generic_query", check_generic)
workflow.add_node("check_database", check_context_in_db)
workflow.add_node("web_search_needed", check_needs_web_search)
workflow.add_node("perform_web_search", perform_web_search)
workflow.add_node("add_documents_to_db", add_documents_to_db)
workflow.add_node("retrieve_context", retrieve_context)
workflow.add_node("call_llm", call_llm_state)
workflow.add_node("check_link", check_link)


# Define edges
workflow.add_edge(START, "check_prompt")
workflow.add_edge("check_prompt", "check_link")
workflow.add_conditional_edges(
    "check_link", 
    lambda state: state["is_website"],
    {True: "perform_web_search", False: "get_vector_db"}
)
# workflow.add_edge("check_prompt","get_vector_db")
workflow.add_edge("get_vector_db", "check_generic_query")
workflow.add_conditional_edges(
    "check_generic_query", 
    lambda state: state["is_generic"],
    {True: "call_llm", False: "check_database"}
)
workflow.add_conditional_edges(
    "check_database",
    lambda state: state["context_exists"],
    {True: "retrieve_context", False: "web_search_needed"}
)
workflow.add_conditional_edges(
    "web_search_needed",
    lambda state: state["needs_web"],
    {False: "call_llm", True: "perform_web_search"}
)
workflow.add_edge("perform_web_search", "add_documents_to_db")
workflow.add_edge("add_documents_to_db", "check_database")
workflow.add_edge("retrieve_context", "call_llm")
workflow.add_edge("call_llm", END)

# Compile the workflow
app = workflow.compile()
