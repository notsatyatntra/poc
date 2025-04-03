from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.util import normalize_url
import tempfile
import os
from langchain_ollama import OllamaEmbeddings


def get_vectorstore():
    """Initialize and return the Chroma vector store."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    return Chroma(
        persist_directory="./web-search-llm-db",
        collection_name="web_llm",
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
        client_settings=Settings(anonymized_telemetry=False)
    )

def has_relevant_data(vectorstore, query, web_search_count, threshold=0.7):
    """"Check if the vector store has relevant data for the query."""
    distance_threshold = 1 - threshold
    results = vectorstore.similarity_search_with_score(query, k=10)
    for doc, distance in results:
        if distance <= distance_threshold:
            print(doc)
            return True
    if  web_search_count>=2 :
        return True       
    return False

def get_relevant_context(vectorstore, query, threshold=0.7):
    """Retrieve relevant context from the vector store."""
    distance_threshold = 1 - threshold
    results = vectorstore.similarity_search_with_score(query, k=10)
    context = [doc.page_content for doc, distance in results if distance <= distance_threshold]
    return context

def add_to_vector_database(vectorstore, documents):
    """Add documents to the vector store."""
    for result in documents:
        documents_list, ids = [], []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )
        if result.markdown:
            markdown_result = result.markdown.fit_markdown
        else:
            continue

        temp_file = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
        temp_file.write(markdown_result)
        temp_file.flush()

        loader = UnstructuredMarkdownLoader(temp_file.name, mode="single")
        docs = loader.load()
        all_splits = text_splitter.split_documents(docs)
        os.unlink(temp_file.name)  # Delete the temporary file

        normalized_url = normalize_url(result.url)

        if all_splits:
            for idx, split in enumerate(all_splits):
                documents_list.append(Document(page_content=split.page_content, metadata={"source": result.url}))
                ids.append(f"{normalized_url}_{idx}")

            vectorstore.add_documents(documents=documents_list, ids=ids)

    