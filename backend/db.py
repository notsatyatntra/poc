from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.util import normalize_url
import tempfile
import os

def get_vectorstore():
    """Initialize and return the Chroma vector store."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", base_url="http://localhost:11434")
    #chromadb
    chroma_client = Chroma.PersistentClient(
        path="./web-search-llm-db", settings=Settings(anonymized_telemetry=False)
    )
    return (
        chroma_client.get_or_create_collection(
            name="web_llm",
            embedding_function=embeddings,
            metadata={"hnsw:space": "cosine"},
        ),
        chroma_client,
    )

def has_relevant_data(vectorstore, query, threshold=0.7):
    """Check if the vector store has relevant data for the query."""
    results = vectorstore.similarity_search_with_score(query, k=1) # returns list of tuples
    if results:
        _, score = results[0]
        return score < threshold  # Lower score = higher similarity
    return False

def get_relevant_context(collection, query, n_results=10):
    """Retrieve relevant context from the vector store."""
    results = collection.similarity_search(query, k=n_results) # returns objects
    return [doc.page_content for doc in results]

def add_to_vector_database(collection, documents):
    """Add documents to the vector store."""
    for result in documents:
        documents, metadatas, ids = [], [], []

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
                documents.append(split.page_content)
                metadatas.append({"source": result.url})
                ids.append(f"{normalized_url}_{idx}")

            print("Upsert collection: ", id(collection))
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
    