"""
Embedding Generation Module

Handles multilingual text embeddings optimized for Bengali and English content.
Uses sentence-transformers with multilingual models for semantic similarity.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Union
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path
import pickle
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Multilingual embedding generator optimized for Bengali-English content.
    
    Features:
    - Multilingual sentence transformer models
    - Caching for performance optimization
    - Batch processing for large documents
    - GPU acceleration when available
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 cache_dir: str = "./embeddings_cache",
                 device: Optional[str] = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: HuggingFace model name for sentence transformers
            cache_dir: Directory to cache embeddings
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Load the model
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        # Model info
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, 
                          texts: Union[str, List[str]], 
                          batch_size: int = 32,
                          use_cache: bool = True,
                          show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            use_cache: Whether to use caching
            show_progress: Show progress bar
            
        Returns:
            numpy array of embeddings
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        if not texts:
            logger.warning("Empty text list provided")
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Check cache if enabled
        if use_cache:
            cached_embeddings, missing_indices = self._check_cache(texts)
            if not missing_indices:
                logger.info("All embeddings found in cache")
                return cached_embeddings[0] if single_text else cached_embeddings
        else:
            cached_embeddings = None
            missing_indices = list(range(len(texts)))
        
        # Generate embeddings for missing texts
        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            logger.info(f"Generating {len(missing_texts)} new embeddings")
            
            try:
                new_embeddings = self.model.encode(
                    missing_texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # L2 normalization for better similarity
                )
                
                # Cache new embeddings
                if use_cache:
                    self._cache_embeddings(missing_texts, new_embeddings)
                
                # Combine with cached embeddings
                if cached_embeddings is not None:
                    all_embeddings = cached_embeddings.copy()
                    for i, idx in enumerate(missing_indices):
                        all_embeddings[idx] = new_embeddings[i]
                else:
                    all_embeddings = new_embeddings
                    
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                raise
        else:
            all_embeddings = cached_embeddings
        
        return all_embeddings[0] if single_text else all_embeddings
    
    def compute_similarity(self, 
                          query_embedding: np.ndarray, 
                          doc_embeddings: np.ndarray,
                          method: str = "cosine") -> np.ndarray:
        """
        Compute similarity between query and document embeddings.
        
        Args:
            query_embedding: Query embedding vector
            doc_embeddings: Document embedding matrix
            method: Similarity method ("cosine", "dot", "euclidean")
            
        Returns:
            Array of similarity scores
        """
        if method == "cosine":
            # Cosine similarity (assuming normalized embeddings)
            similarities = np.dot(doc_embeddings, query_embedding)
        elif method == "dot":
            # Dot product similarity
            similarities = np.dot(doc_embeddings, query_embedding)
        elif method == "euclidean":
            # Negative euclidean distance (higher is more similar)
            distances = np.linalg.norm(doc_embeddings - query_embedding, axis=1)
            similarities = -distances
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        return similarities
    
    def find_most_similar(self, 
                         query: str, 
                         documents: List[str],
                         top_k: int = 5,
                         threshold: float = 0.0) -> List[Dict]:
        """
        Find most similar documents to a query.
        
        Args:
            query: Query text
            documents: List of document texts
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of dictionaries with document info and similarity scores
        """
        if not documents:
            return []
        
        # Generate embeddings
        query_embedding = self.generate_embeddings(query)
        doc_embeddings = self.generate_embeddings(documents)
        
        # Compute similarities
        similarities = self.compute_similarity(query_embedding, doc_embeddings)
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])
            if similarity >= threshold:
                results.append({
                    'document_index': int(idx),
                    'document': documents[idx],
                    'similarity_score': similarity,
                    'rank': len(results) + 1
                })
        
        return results
    
    def _check_cache(self, texts: List[str]) -> tuple:
        """Check cache for existing embeddings."""
        cached_embeddings = [None] * len(texts)
        missing_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cached_embeddings[i] = pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cached embedding: {e}")
                    missing_indices.append(i)
            else:
                missing_indices.append(i)
        
        # Convert to numpy array if we have cached embeddings
        if any(emb is not None for emb in cached_embeddings):
            cached_array = np.array([emb for emb in cached_embeddings if emb is not None])
            return cached_array, missing_indices
        
        return None, missing_indices
    
    def _cache_embeddings(self, texts: List[str], embeddings: np.ndarray):
        """Cache embeddings to disk."""
        for text, embedding in zip(texts, embeddings):
            cache_key = self._get_cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(embedding, f)
            except Exception as e:
                logger.warning(f"Failed to cache embedding: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Use hash of text + model name for unique key
        content = f"{self.model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'device': self.device,
            'max_sequence_length': getattr(self.model, 'max_seq_length', 'Unknown'),
            'cache_dir': str(self.cache_dir)
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            logger.info("Embedding cache cleared")


def main():
    """Test the embedding generator."""
    # Initialize embedding generator
    embedder = EmbeddingGenerator()
    
    # Test texts in Bengali and English
    test_texts = [
        "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "Who is referred to as a good man in Anupam's language?",
        "শুম্ভুনাথ ছিলেন একজন আদর্শ মানুষ।",
        "Shumbhunath was an ideal person.",
        "কল্যাণীর বিয়ের সময় বয়স ছিল পনেরো বছর।"
    ]
    
    print("Model Info:")
    print(embedder.get_model_info())
    
    # Generate embeddings
    print(f"\nGenerating embeddings for {len(test_texts)} texts...")
    embeddings = embedder.generate_embeddings(test_texts)
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Test similarity
    query = "অনুপমের ভাগ্য দেবতা কে?"
    documents = [
        "মামা ছিলেন অনুপমের ভাগ্য দেবতা।",
        "শুম্ভুনাথ একজন সুপুরুষ ছিলেন।",
        "কল্যাণী খুব সুন্দর ছিল।"
    ]
    
    print(f"\nFinding similar documents for query: {query}")
    results = embedder.find_most_similar(query, documents, top_k=3)
    
    for result in results:
        print(f"Rank {result['rank']}: {result['document']}")
        print(f"Similarity: {result['similarity_score']:.4f}\n")


if __name__ == "__main__":
    main()