"""
Vector Store Module using ChromaDB

Manages document storage and retrieval using ChromaDB vector database.
Optimized for multilingual content with metadata support.
"""

import logging
import uuid
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings
import numpy as np
from pathlib import Path
import json

from .chunker import Chunk
from .embeddings import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB-based vector store for multilingual RAG system.
    
    Features:
    - Persistent storage with ChromaDB
    - Metadata filtering and search
    - Batch operations for performance
    - Multilingual embedding support
    """
    
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "multilingual_rag",
                 embedding_generator: Optional[EmbeddingGenerator] = None):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            embedding_generator: EmbeddingGenerator instance
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding generator
        if embedding_generator is None:
            self.embedding_generator = EmbeddingGenerator()
        else:
            self.embedding_generator = embedding_generator
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=None  # We'll handle embeddings manually
            )
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=None,
                metadata={"description": "Multilingual RAG collection for Bengali-English content"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_chunks(self, chunks: List[Chunk], batch_size: int = 100) -> bool:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of Chunk objects to add
            batch_size: Batch size for processing
            
        Returns:
            True if successful, False otherwise
        """
        if not chunks:
            logger.warning("No chunks provided to add")
            return False
        
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        try:
            # Process in batches
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                self._add_batch(batch_chunks)
                logger.info(f"Added batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
            
            logger.info(f"Successfully added {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks: {e}")
            return False
    
    def _add_batch(self, chunks: List[Chunk]):
        """Add a batch of chunks to the collection."""
        # Extract texts for embedding generation
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings_list = []
        
        for chunk, embedding in zip(chunks, embeddings):
            # Generate unique ID if not provided
            chunk_id = chunk.chunk_id or str(uuid.uuid4())
            
            # Prepare metadata
            metadata = {
                'chunk_id': chunk_id,
                'word_count': chunk.word_count or 0,
                'char_count': len(chunk.text),
                'source_page': chunk.source_page,
                'start_char': chunk.start_char,
                'end_char': chunk.end_char
            }
            
            # Add custom metadata if available
            if chunk.metadata:
                metadata.update(chunk.metadata)
            
            # Convert numpy types to Python types for JSON serialization
            metadata = self._serialize_metadata(metadata)
            
            ids.append(chunk_id)
            documents.append(chunk.text)
            metadatas.append(metadata)
            embeddings_list.append(embedding.tolist())
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings_list
        )
    
    def search(self, 
               query: str, 
               n_results: int = 5,
               where: Optional[Dict] = None,
               threshold: float = 0.0) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            where: Metadata filter conditions
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embeddings(query)
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            processed_results = []
            for i in range(len(results['documents'][0])):
                # Convert distance to similarity score (ChromaDB uses L2 distance)
                distance = results['distances'][0][i]
                similarity = 1 / (1 + distance)  # Convert distance to similarity
                
                if similarity >= threshold:
                    result = {
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity_score': similarity,
                        'distance': distance,
                        'rank': i + 1
                    }
                    processed_results.append(result)
            
            logger.info(f"Found {len(processed_results)} results for query")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def search_by_metadata(self, 
                          where: Dict,
                          n_results: int = 10) -> List[Dict]:
        """
        Search documents by metadata only.
        
        Args:
            where: Metadata filter conditions
            n_results: Number of results to return
            
        Returns:
            List of matching documents
        """
        try:
            results = self.collection.get(
                where=where,
                limit=n_results,
                include=['documents', 'metadatas']
            )
            
            processed_results = []
            for i in range(len(results['documents'])):
                result = {
                    'document': results['documents'][i],
                    'metadata': results['metadatas'][i],
                    'id': results['ids'][i] if 'ids' in results else None
                }
                processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []
    
    def delete_by_metadata(self, where: Dict) -> bool:
        """
        Delete documents by metadata filter.
        
        Args:
            where: Metadata filter conditions
            
        Returns:
            True if successful
        """
        try:
            self.collection.delete(where=where)
            logger.info(f"Deleted documents matching: {where}")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            
            # Get sample documents to analyze
            sample_results = self.collection.peek(limit=min(100, count))
            
            stats = {
                'total_documents': count,
                'collection_name': self.collection_name,
                'persist_directory': str(self.persist_directory)
            }
            
            if sample_results['metadatas']:
                # Analyze metadata
                word_counts = [m.get('word_count', 0) for m in sample_results['metadatas'] if m.get('word_count')]
                char_counts = [m.get('char_count', 0) for m in sample_results['metadatas'] if m.get('char_count')]
                
                if word_counts:
                    stats['avg_word_count'] = sum(word_counts) / len(word_counts)
                    stats['avg_char_count'] = sum(char_counts) / len(char_counts)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}
    
    def reset_collection(self) -> bool:
        """Reset (clear) the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={"description": "Multilingual RAG collection for Bengali-English content"}
            )
            logger.info(f"Reset collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False
    
    def _serialize_metadata(self, metadata: Dict) -> Dict:
        """Convert metadata to JSON-serializable format."""
        serialized = {}
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (int, float, str, bool)):
                serialized[key] = value
            elif isinstance(value, np.integer):
                serialized[key] = int(value)
            elif isinstance(value, np.floating):
                serialized[key] = float(value)
            else:
                # Convert to string for complex types
                serialized[key] = str(value)
        
        return serialized


def main():
    """Test the vector store."""
    from .chunker import DocumentChunker, Chunk
    
    # Initialize components
    vector_store = VectorStore()
    chunker = DocumentChunker()
    
    # Sample Bengali text
    sample_text = """
    অনুপম একজন সাধারণ মানুষ ছিল। তার জীবনে অনেক ঘটনা ঘটেছিল। 
    সে তার মামার কাছে অনেক কিছু শিখেছিল। মামা ছিলেন তার ভাগ্য দেবতা।
    
    শুম্ভুনাথ ছিলেন একজন সুপুরুষ। অনুপমের ভাষায় তিনি ছিলেন আদর্শ মানুষ।
    তার চরিত্রে অনেক ভালো গুণ ছিল।
    
    কল্যাণীর বিয়ের সময় তার বয়স ছিল মাত্র পনেরো বছর। 
    সে ছিল খুবই সুন্দর এবং বুদ্ধিমতী একটি মেয়ে।
    """
    
    # Create chunks
    chunks = chunker.chunk_document(sample_text)
    print(f"Created {len(chunks)} chunks")
    
    # Add to vector store
    success = vector_store.add_chunks(chunks)
    print(f"Added chunks to vector store: {success}")
    
    # Get collection stats
    stats = vector_store.get_collection_stats()
    print(f"Collection stats: {stats}")
    
    # Test search
    query = "অনুপমের ভাগ্য দেবতা কে?"
    results = vector_store.search(query, n_results=3)
    
    print(f"\nSearch results for: {query}")
    for result in results:
        print(f"Similarity: {result['similarity_score']:.4f}")
        print(f"Text: {result['document'][:100]}...")
        print()


if __name__ == "__main__":
    main()