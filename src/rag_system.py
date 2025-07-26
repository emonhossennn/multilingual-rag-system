"""
Main RAG System Implementation

Combines all components to create a complete multilingual RAG system.
Handles query processing, retrieval, and response generation.
"""

import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import os
from pathlib import Path

from .pdf_processor import PDFProcessor
from .chunker import DocumentChunker, Chunk
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .memory import ConversationMemory
from .llm_client import LLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from the RAG system."""
    answer: str
    retrieved_chunks: List[Dict]
    confidence_score: float
    processing_time: float
    query_language: str
    sources: List[str]
    metadata: Dict


class MultilingualRAGSystem:
    """
    Complete multilingual RAG system for Bengali and English queries.
    
    Features:
    - PDF document processing and indexing
    - Multilingual query understanding
    - Semantic retrieval with context
    - LLM-based response generation
    - Conversation memory management
    """
    
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 max_retrieved_chunks: int = 5):
        """
        Initialize the RAG system.
        
        Args:
            persist_directory: Directory for vector database persistence
            embedding_model: Embedding model name
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            max_retrieved_chunks: Maximum chunks to retrieve per query
        """
        self.persist_directory = persist_directory
        self.max_retrieved_chunks = max_retrieved_chunks
        
        # Initialize components
        logger.info("Initializing RAG system components...")
        
        self.pdf_processor = PDFProcessor()
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        self.vector_store = VectorStore(
            persist_directory=persist_directory,
            embedding_generator=self.embedding_generator
        )
        self.memory = ConversationMemory()
        self.llm_client = LLMClient()
        
        logger.info("RAG system initialized successfully")
    
    def index_document(self, pdf_path: str, metadata: Optional[Dict] = None) -> bool:
        """
        Index a PDF document into the vector store.
        
        Args:
            pdf_path: Path to the PDF file
            metadata: Additional metadata for the document
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Indexing document: {pdf_path}")
            
            # Extract text from PDF
            extracted_data = self.pdf_processor.extract_text_from_pdf(pdf_path)
            
            if not extracted_data['text'].strip():
                logger.error("No text extracted from PDF")
                return False
            
            # Create document metadata
            doc_metadata = {
                'source_file': pdf_path,
                'total_pages': extracted_data['total_pages'],
                'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            if metadata:
                doc_metadata.update(metadata)
            
            # Chunk the document
            chunks = self.chunker.chunk_document(
                text=extracted_data['text'],
                strategy="semantic_paragraph",
                metadata=doc_metadata
            )
            
            if not chunks:
                logger.error("No chunks created from document")
                return False
            
            # Add page information to chunks
            for chunk in chunks:
                # Try to determine which page this chunk belongs to
                chunk.metadata['source_file'] = pdf_path
                if chunk.metadata is None:
                    chunk.metadata = {}
                chunk.metadata.update(doc_metadata)
            
            # Add chunks to vector store
            success = self.vector_store.add_chunks(chunks)
            
            if success:
                logger.info(f"Successfully indexed {len(chunks)} chunks from {pdf_path}")
            else:
                logger.error(f"Failed to index document: {pdf_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error indexing document {pdf_path}: {e}")
            return False
    
    def query(self, 
              user_query: str, 
              conversation_id: Optional[str] = None,
              include_context: bool = True) -> RAGResponse:
        """
        Process a user query and generate a response.
        
        Args:
            user_query: User's question
            conversation_id: ID for conversation tracking
            include_context: Whether to include conversation context
            
        Returns:
            RAGResponse object with answer and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: {user_query[:100]}...")
            
            # Detect query language
            query_language = self._detect_language(user_query)
            logger.info(f"Detected language: {query_language}")
            
            # Get conversation context if available
            context = ""
            if include_context and conversation_id:
                context = self.memory.get_context(conversation_id)
            
            # Enhance query with context if available
            enhanced_query = self._enhance_query_with_context(user_query, context)
            
            # Retrieve relevant chunks
            retrieved_chunks = self.vector_store.search(
                query=enhanced_query,
                n_results=self.max_retrieved_chunks,
                threshold=0.1  # Minimum similarity threshold
            )
            
            if not retrieved_chunks:
                logger.warning("No relevant chunks found")
                return self._create_fallback_response(user_query, query_language, start_time)
            
            # Generate response using LLM
            response_text, confidence = self._generate_response(
                query=user_query,
                retrieved_chunks=retrieved_chunks,
                context=context,
                language=query_language
            )
            
            # Update conversation memory
            if conversation_id:
                self.memory.add_interaction(
                    conversation_id=conversation_id,
                    user_query=user_query,
                    system_response=response_text,
                    retrieved_chunks=retrieved_chunks
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Extract sources
            sources = list(set([
                chunk['metadata'].get('source_file', 'Unknown')
                for chunk in retrieved_chunks
            ]))
            
            # Create response
            response = RAGResponse(
                answer=response_text,
                retrieved_chunks=retrieved_chunks,
                confidence_score=confidence,
                processing_time=processing_time,
                query_language=query_language,
                sources=sources,
                metadata={
                    'num_chunks_retrieved': len(retrieved_chunks),
                    'conversation_id': conversation_id,
                    'enhanced_query': enhanced_query != user_query
                }
            )
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            processing_time = time.time() - start_time
            return self._create_error_response(user_query, str(e), processing_time)
    
    def _detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        try:
            from langdetect import detect
            detected = detect(text)
            
            # Map language codes to our supported languages
            if detected == 'bn':
                return 'bengali'
            elif detected == 'en':
                return 'english'
            else:
                # Check for Bengali characters
                bengali_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
                if bengali_chars > len(text) * 0.1:  # More than 10% Bengali characters
                    return 'bengali'
                else:
                    return 'english'
                    
        except Exception:
            # Fallback: check for Bengali Unicode range
            bengali_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
            if bengali_chars > len(text) * 0.1:
                return 'bengali'
            else:
                return 'english'
    
    def _enhance_query_with_context(self, query: str, context: str) -> str:
        """Enhance query with conversation context."""
        if not context:
            return query
        
        # Simple context enhancement - can be made more sophisticated
        enhanced = f"Context: {context}\n\nQuestion: {query}"
        return enhanced
    
    def _generate_response(self, 
                          query: str, 
                          retrieved_chunks: List[Dict],
                          context: str,
                          language: str) -> Tuple[str, float]:
        """Generate response using LLM."""
        try:
            # Prepare context from retrieved chunks
            chunk_texts = [chunk['document'] for chunk in retrieved_chunks]
            combined_context = "\n\n".join(chunk_texts)
            
            # Generate response
            response, confidence = self.llm_client.generate_response(
                query=query,
                context=combined_context,
                conversation_context=context,
                language=language
            )
            
            return response, confidence
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}", 0.0
    
    def _create_fallback_response(self, query: str, language: str, start_time: float) -> RAGResponse:
        """Create a fallback response when no relevant chunks are found."""
        if language == 'bengali':
            answer = "দুঃখিত, আপনার প্রশ্নের জন্য আমি কোনো প্রাসঙ্গিক তথ্য খুঁজে পাইনি। অনুগ্রহ করে আরো স্পষ্ট করে প্রশ্ন করুন।"
        else:
            answer = "I'm sorry, I couldn't find relevant information for your question. Please try rephrasing your query."
        
        return RAGResponse(
            answer=answer,
            retrieved_chunks=[],
            confidence_score=0.0,
            processing_time=time.time() - start_time,
            query_language=language,
            sources=[],
            metadata={'fallback_response': True}
        )
    
    def _create_error_response(self, query: str, error: str, processing_time: float) -> RAGResponse:
        """Create an error response."""
        return RAGResponse(
            answer=f"An error occurred while processing your query: {error}",
            retrieved_chunks=[],
            confidence_score=0.0,
            processing_time=processing_time,
            query_language='unknown',
            sources=[],
            metadata={'error': True, 'error_message': error}
        )
    
    def get_system_stats(self) -> Dict:
        """Get system statistics."""
        vector_stats = self.vector_store.get_collection_stats()
        memory_stats = self.memory.get_stats()
        
        return {
            'vector_store': vector_stats,
            'memory': memory_stats,
            'embedding_model': self.embedding_generator.get_model_info(),
            'system_status': 'operational'
        }
    
    def reset_system(self) -> bool:
        """Reset the entire system (clear all data)."""
        try:
            # Reset vector store
            self.vector_store.reset_collection()
            
            # Reset memory
            self.memory.clear_all_conversations()
            
            logger.info("System reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting system: {e}")
            return False


def main():
    """Test the RAG system."""
    # Initialize RAG system
    rag = MultilingualRAGSystem()
    
    # Test with sample document (you'll need to provide the actual PDF)
    pdf_path = "data/hsc_bangla_1st_paper.pdf"
    
    if Path(pdf_path).exists():
        print("Indexing document...")
        success = rag.index_document(pdf_path)
        print(f"Document indexed: {success}")
        
        if success:
            # Test queries
            test_queries = [
                "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
                "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
                "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
                "Who is referred to as a good man in Anupam's language?"
            ]
            
            for query in test_queries:
                print(f"\nQuery: {query}")
                response = rag.query(query)
                print(f"Answer: {response.answer}")
                print(f"Confidence: {response.confidence_score:.2f}")
                print(f"Processing time: {response.processing_time:.2f}s")
                print(f"Language: {response.query_language}")
    else:
        print(f"Please place your HSC Bangla PDF at: {pdf_path}")
    
    # Get system stats
    stats = rag.get_system_stats()
    print(f"\nSystem Stats: {stats}")


if __name__ == "__main__":
    main()