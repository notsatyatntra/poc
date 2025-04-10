from langchain_chroma import Chroma
from langchain_core.documents import Document
from chromadb.config import Settings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend_langgraph.util_class import UtilManager, UtilConfig
from langchain_ollama import OllamaEmbeddings
import tempfile
import os
from typing import List, Dict, Optional, Callable, Any

class VectorStoreConfig:
    """Configuration class for vector store settings."""
    
    def __init__(
        self,
        embedding_model: str = "nomic-embed-text:latest",
        embedding_function: Optional[Callable] = None,
        persist_directory: str = "./web-search-llm-db",
        collection_name: str = "web_llm",
        collection_metadata: Dict[str, str] = {"hnsw:space": "cosine"},
        client_settings: Settings = Settings(anonymized_telemetry=False),
        text_splitter_chunk_size: int = 400,
        text_splitter_chunk_overlap: int = 100,
        text_splitter_separators: List[str] = ["\n\n", "\n", ".", "?", "!", " ", ""],
        similarity_threshold: float = 0.7,
        max_web_search_count: int = 2,
        util_config: Optional[UtilConfig] = None
    ):
        """
        Args:
            embedding_model (str): Name of the embedding model (e.g., "nomic-embed-text:latest").
            embedding_function (Callable): Custom embedding function; if None, uses OllamaEmbeddings.
            persist_directory (str): Directory to persist the vector store.
            collection_name (str): Name of the collection in the vector store.
            collection_metadata (Dict[str, str]): Metadata for the collection (e.g., HNSW settings).
            client_settings (Settings): Chroma client settings.
            text_splitter_chunk_size (int): Chunk size for text splitting.
            text_splitter_chunk_overlap (int): Overlap between chunks for text splitting.
            text_splitter_separators (List[str]): Separators for text splitting.
            similarity_threshold (float): Threshold for relevance in similarity search (0 to 1).
            max_web_search_count (int): Maximum web search attempts before assuming relevance.
            util_config (UtilConfig): Configuration for UtilManager (e.g., for normalize_url).
        """
        self.embedding_model = embedding_model
        self.embedding_function = embedding_function or OllamaEmbeddings(model=embedding_model)
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.collection_metadata = collection_metadata
        self.client_settings = client_settings
        self.text_splitter_chunk_size = text_splitter_chunk_size
        self.text_splitter_chunk_overlap = text_splitter_chunk_overlap
        self.text_splitter_separators = text_splitter_separators
        self.similarity_threshold = similarity_threshold
        self.max_web_search_count = max_web_search_count
        self.util_config = util_config or UtilConfig()

class VectorStoreManager:
    """Manages Chroma vector store operations."""
    
    def __init__(self, config: VectorStoreConfig):
        """
        Initialize the VectorStoreManager with a configuration.

        Args:
            config (VectorStoreConfig): Configuration object for vector store settings.
        """
        self.config = config
        self.util_manager = UtilManager(self.config.util_config)  # For normalize_url
        self.vectorstore = self._initialize_vectorstore()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.text_splitter_chunk_size,
            chunk_overlap=self.config.text_splitter_chunk_overlap,
            separators=self.config.text_splitter_separators,
        )

    def _initialize_vectorstore(self) -> Chroma:
        """Initialize and return the Chroma vector store."""
        return Chroma(
            persist_directory=self.config.persist_directory,
            collection_name=self.config.collection_name,
            embedding_function=self.config.embedding_function,
            collection_metadata=self.config.collection_metadata,
            client_settings=self.config.client_settings
        )

    def has_relevant_data(self, query: str, web_search_count: int) -> bool:
        """Check if the vector store has relevant data for the query."""
        distance_threshold = 1 - self.config.similarity_threshold
        results = self.vectorstore.similarity_search_with_score(query, k=10)
        for _, distance in results:
            if distance <= distance_threshold:
                return True
        if web_search_count >= self.config.max_web_search_count:
            return True
        return False

    def get_relevant_context(self, query: str) -> List[str]:
        """Retrieve relevant context from the vector store."""
        distance_threshold = 1 - self.config.similarity_threshold
        results = self.vectorstore.similarity_search_with_score(query, k=10)
        context = [doc.page_content for doc, distance in results if distance <= distance_threshold]
        return context

    def add_to_vector_database(self, documents: List[Any]) -> None:
        """Add documents to the vector store."""
        for result in documents:
            documents_list, ids = [], []

            if not hasattr(result, 'markdown') or not result.markdown:
                continue

            markdown_result = result.markdown.fit_markdown
            with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as temp_file:
                temp_file.write(markdown_result)
                temp_file.flush()

                loader = UnstructuredMarkdownLoader(temp_file.name, mode="single")
                docs = loader.load()
                all_splits = self.text_splitter.split_documents(docs)
                os.unlink(temp_file.name)  # Delete the temporary file

            normalized_url = self.util_manager.normalize_url(result.url)

            if all_splits:
                for idx, split in enumerate(all_splits):
                    documents_list.append(Document(
                        page_content=split.page_content,
                        metadata={"source": result.url}
                    ))
                    ids.append(f"{normalized_url}_{idx}")

                self.vectorstore.add_documents(documents=documents_list, ids=ids)

    def get_vectorstore(self) -> Chroma:
        """Return the underlying vector store instance."""
        return self.vectorstore

# Example usage
if __name__ == "__main__":
    # Define a configuration (customize as needed)
    config = VectorStoreConfig(
        embedding_model="nomic-embed-text:latest",
        persist_directory="./custom-web-db",
        collection_name="custom_collection",
        similarity_threshold=0.8,
        text_splitter_chunk_size=500,
        text_splitter_chunk_overlap=150
    )
    
    # Initialize the vector store manager
    vector_store_manager = VectorStoreManager(config)
    
    # Test the functions
    query = "What is AI?"
    has_data = vector_store_manager.has_relevant_data(query, web_search_count=1)
    print(f"Has relevant data: {has_data}")
    
    context = vector_store_manager.get_relevant_context(query)
    print(f"Relevant context: {context}")
    
    # Example document addition (assuming documents are from crawl results)
    # documents = [some_crawl_result_with_markdown]
    # vector_store_manager.add_to_vector_database(documents)