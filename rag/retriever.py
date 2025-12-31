"""
Retriever module for storing and searching document embeddings using ChromaDB.
"""
import os
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings

from .embedder import Embedder, load_documents


class Retriever:
    """Handles document storage and retrieval using ChromaDB."""
    
    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persist_directory: str = "db",
        embedder: Optional[Embedder] = None
    ):
        """
        Initialize the retriever with ChromaDB.
        
        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory to persist the database.
            embedder: Optional Embedder instance. Creates one if not provided.
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedder
        self.embedder = embedder or Embedder()
        
        # Cache for query embeddings
        self._query_cache: Dict[str, List[float]] = {}
    
    def index_documents(self, data_dir: str = "data", force_reindex: bool = False) -> int:
        """
        Load and index all documents from the data directory.
        
        Args:
            data_dir: Path to the directory containing documents.
            force_reindex: If True, clear existing documents and reindex.
            
        Returns:
            Number of documents indexed.
        """
        # Check if already indexed
        if self.collection.count() > 0 and not force_reindex:
            print(f"Collection already has {self.collection.count()} documents. Use force_reindex=True to reindex.")
            return self.collection.count()
        
        # Clear existing documents if force reindexing
        if force_reindex and self.collection.count() > 0:
            # Delete all existing documents
            existing_ids = self.collection.get()["ids"]
            if existing_ids:
                self.collection.delete(ids=existing_ids)
        
        # Load documents
        documents = load_documents(data_dir)
        
        if not documents:
            print("No documents found to index.")
            return 0
        
        # Prepare data for ChromaDB
        ids = [doc["chunk_id"] for doc in documents]
        contents = [doc["content"] for doc in documents]
        metadatas = [{"source": doc["source"], "chunk_id": doc["chunk_id"]} for doc in documents]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = self.embedder.embed_batch(contents)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )
        
        print(f"Successfully indexed {len(documents)} document chunks.")
        return len(documents)
    
    def search(
        self,
        query: str,
        top_k: int = 3,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Search for relevant documents based on a query.
        
        Args:
            query: The search query.
            top_k: Number of results to return.
            use_cache: Whether to use cached query embeddings.
            
        Returns:
            List of dictionaries containing matched documents and scores.
        """
        # Check cache
        if use_cache and query in self._query_cache:
            query_embedding = self._query_cache[query]
        else:
            query_embedding = self.embedder.embed(query)
            if use_cache:
                self._query_cache[query] = query_embedding
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "content": results["documents"][0][i],
                    "source": results["metadatas"][0][i]["source"],
                    "chunk_id": results["metadatas"][0][i]["chunk_id"],
                    "distance": results["distances"][0][i],
                    "relevance_score": 1 - results["distances"][0][i]  # Convert distance to similarity
                })
        
        return formatted_results
    
    def get_document_count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()
    
    def clear_cache(self):
        """Clear the query embedding cache."""
        self._query_cache.clear()
