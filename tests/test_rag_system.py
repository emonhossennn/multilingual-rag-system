"""
Unit tests for the RAG system components
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_system import MultilingualRAGSystem, RAGResponse
from src.chunker import DocumentChunker, Chunk
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.memory import ConversationMemory


class TestMultilingualRAGSystem:
    """Test cases for the main RAG system."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def rag_system(self, temp_dir):
        """Create RAG system instance for testing."""
        return MultilingualRAGSystem(
            persist_directory=temp_dir,
            chunk_size=256,
            chunk_overlap=25,
            max_retrieved_chunks=3
        )
    
    @pytest.fixture
    def sample_text(self):
        """Sample Bengali text for testing."""
        return """
        অনুপম একজন সাধারণ মানুষ ছিল। তার জীবনে অনেক ঘটনা ঘটেছিল। 
        সে তার মামার কাছে অনেক কিছু শিখেছিল। মামা ছিলেন তার ভাগ্য দেবতা।
        
        শুম্ভুনাথ ছিলেন একজন সুপুরুষ। অনুপমের ভাষায় তিনি ছিলেন আদর্শ মানুষ।
        তার চরিত্রে অনেক ভালো গুণ ছিল।
        """
    
    def test_system_initialization(self, rag_system):
        """Test RAG system initialization."""
        assert rag_system is not None
        assert rag_system.vector_store is not None
        assert rag_system.memory is not None
        assert rag_system.embedding_generator is not None
    
    def test_language_detection(self, rag_system):
        """Test language detection functionality."""
        bengali_text = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
        english_text = "Who is referred to as a good man?"
        
        bengali_lang = rag_system._detect_language(bengali_text)
        english_lang = rag_system._detect_language(english_text)
        
        assert bengali_lang == 'bengali'
        assert english_lang == 'english'
    
    def test_text_indexing(self, rag_system, sample_text):
        """Test text indexing functionality."""
        # Create chunks manually
        from src.chunker import DocumentChunker
        chunker = DocumentChunker(chunk_size=256, chunk_overlap=25)
        
        chunks = chunker.chunk_document(
            text=sample_text,
            strategy="paragraph",
            metadata={"source": "test_data"}
        )
        
        # Add to vector store
        success = rag_system.vector_store.add_chunks(chunks)
        assert success is True
        
        # Check if chunks were added
        stats = rag_system.get_system_stats()
        assert stats['vector_store']['total_documents'] > 0
    
    def test_query_processing(self, rag_system, sample_text):
        """Test query processing."""
        # First add some data
        from src.chunker import DocumentChunker
        chunker = DocumentChunker(chunk_size=256, chunk_overlap=25)
        chunks = chunker.chunk_document(sample_text, metadata={"source": "test"})
        rag_system.vector_store.add_chunks(chunks)
        
        # Test query
        query = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
        response = rag_system.query(query)
        
        assert isinstance(response, RAGResponse)
        assert response.answer is not None
        assert response.query_language == 'bengali'
        assert response.processing_time > 0
        assert isinstance(response.confidence_score, float)
    
    def test_conversation_memory(self, rag_system, sample_text):
        """Test conversation memory functionality."""
        # Add sample data
        from src.chunker import DocumentChunker
        chunker = DocumentChunker(chunk_size=256, chunk_overlap=25)
        chunks = chunker.chunk_document(sample_text, metadata={"source": "test"})
        rag_system.vector_store.add_chunks(chunks)
        
        # Test conversation
        conv_id = "test_conversation"
        
        # First query
        response1 = rag_system.query("অনুপম কে?", conversation_id=conv_id)
        assert response1 is not None
        
        # Second query (should have context)
        response2 = rag_system.query("তার মামা কে?", conversation_id=conv_id)
        assert response2 is not None
        
        # Check conversation history
        summary = rag_system.memory.get_conversation_summary(conv_id)
        assert summary['total_interactions'] == 2
    
    def test_system_stats(self, rag_system):
        """Test system statistics."""
        stats = rag_system.get_system_stats()
        
        assert 'vector_store' in stats
        assert 'memory' in stats
        assert 'embedding_model' in stats
        assert 'system_status' in stats
    
    def test_system_reset(self, rag_system, sample_text):
        """Test system reset functionality."""
        # Add some data first
        from src.chunker import DocumentChunker
        chunker = DocumentChunker(chunk_size=256, chunk_overlap=25)
        chunks = chunker.chunk_document(sample_text, metadata={"source": "test"})
        rag_system.vector_store.add_chunks(chunks)
        
        # Verify data exists
        stats_before = rag_system.get_system_stats()
        assert stats_before['vector_store']['total_documents'] > 0
        
        # Reset system
        success = rag_system.reset_system()
        assert success is True
        
        # Verify data is cleared
        stats_after = rag_system.get_system_stats()
        assert stats_after['vector_store']['total_documents'] == 0


class TestDocumentChunker:
    """Test cases for document chunker."""
    
    @pytest.fixture
    def chunker(self):
        """Create chunker instance."""
        return DocumentChunker(chunk_size=100, chunk_overlap=20)
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for chunking."""
        return """
        This is the first paragraph. It contains some text.
        
        This is the second paragraph. It has different content.
        
        This is the third paragraph. It concludes the document.
        """
    
    def test_paragraph_chunking(self, chunker, sample_text):
        """Test paragraph-based chunking."""
        chunks = chunker.chunk_document(sample_text, strategy="paragraph")
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.text.strip() for chunk in chunks)
    
    def test_sentence_chunking(self, chunker, sample_text):
        """Test sentence-based chunking."""
        chunks = chunker.chunk_document(sample_text, strategy="sentence")
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
    
    def test_chunk_metadata(self, chunker, sample_text):
        """Test chunk metadata."""
        metadata = {"source": "test_document", "author": "test_author"}
        chunks = chunker.chunk_document(sample_text, metadata=metadata)
        
        for chunk in chunks:
            assert chunk.metadata is not None
            assert chunk.metadata["source"] == "test_document"
            assert chunk.metadata["author"] == "test_author"


class TestEmbeddingGenerator:
    """Test cases for embedding generator."""
    
    @pytest.fixture
    def embedding_generator(self):
        """Create embedding generator instance."""
        return EmbeddingGenerator(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    def test_single_embedding(self, embedding_generator):
        """Test single text embedding."""
        text = "This is a test sentence."
        embedding = embedding_generator.generate_embeddings(text)
        
        assert embedding is not None
        assert len(embedding.shape) == 1
        assert embedding.shape[0] > 0
    
    def test_batch_embeddings(self, embedding_generator):
        """Test batch embedding generation."""
        texts = [
            "This is the first sentence.",
            "This is the second sentence.",
            "অনুপম একজন সাধারণ মানুষ।"
        ]
        
        embeddings = embedding_generator.generate_embeddings(texts)
        
        assert embeddings is not None
        assert len(embeddings.shape) == 2
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] > 0
    
    def test_multilingual_embeddings(self, embedding_generator):
        """Test multilingual embedding generation."""
        bengali_text = "অনুপম একজন সাধারণ মানুষ।"
        english_text = "Anupam is an ordinary person."
        
        bengali_embedding = embedding_generator.generate_embeddings(bengali_text)
        english_embedding = embedding_generator.generate_embeddings(english_text)
        
        assert bengali_embedding is not None
        assert english_embedding is not None
        assert bengali_embedding.shape == english_embedding.shape


class TestVectorStore:
    """Test cases for vector store."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def vector_store(self, temp_dir):
        """Create vector store instance."""
        return VectorStore(persist_directory=temp_dir)
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        chunks = [
            Chunk(
                text="অনুপম একজন সাধারণ মানুষ ছিল।",
                chunk_id="chunk_1",
                metadata={"source": "test", "language": "bengali"}
            ),
            Chunk(
                text="শুম্ভুনাথ ছিলেন একজন সুপুরুষ।",
                chunk_id="chunk_2", 
                metadata={"source": "test", "language": "bengali"}
            )
        ]
        return chunks
    
    def test_add_chunks(self, vector_store, sample_chunks):
        """Test adding chunks to vector store."""
        success = vector_store.add_chunks(sample_chunks)
        assert success is True
        
        stats = vector_store.get_collection_stats()
        assert stats['total_documents'] == len(sample_chunks)
    
    def test_search(self, vector_store, sample_chunks):
        """Test searching in vector store."""
        # Add chunks first
        vector_store.add_chunks(sample_chunks)
        
        # Search
        query = "অনুপম কে?"
        results = vector_store.search(query, n_results=2)
        
        assert len(results) > 0
        assert all('document' in result for result in results)
        assert all('similarity_score' in result for result in results)
    
    def test_metadata_search(self, vector_store, sample_chunks):
        """Test metadata-based search."""
        # Add chunks first
        vector_store.add_chunks(sample_chunks)
        
        # Search by metadata
        results = vector_store.search_by_metadata({"language": "bengali"})
        
        assert len(results) > 0
        assert all(result['metadata']['language'] == 'bengali' for result in results)


class TestConversationMemory:
    """Test cases for conversation memory."""
    
    @pytest.fixture
    def memory(self):
        """Create memory instance."""
        return ConversationMemory(max_interactions_per_conversation=5)
    
    def test_add_interaction(self, memory):
        """Test adding interactions."""
        conv_id = "test_conv"
        
        memory.add_interaction(
            conversation_id=conv_id,
            user_query="Test question",
            system_response="Test answer",
            retrieved_chunks=[]
        )
        
        summary = memory.get_conversation_summary(conv_id)
        assert summary['total_interactions'] == 1
    
    def test_get_context(self, memory):
        """Test getting conversation context."""
        conv_id = "test_conv"
        
        # Add multiple interactions
        for i in range(3):
            memory.add_interaction(
                conversation_id=conv_id,
                user_query=f"Question {i}",
                system_response=f"Answer {i}",
                retrieved_chunks=[]
            )
        
        context = memory.get_context(conv_id)
        assert context is not None
        assert len(context) > 0
        assert "Question" in context
        assert "Answer" in context
    
    def test_clear_conversation(self, memory):
        """Test clearing conversations."""
        conv_id = "test_conv"
        
        # Add interaction
        memory.add_interaction(
            conversation_id=conv_id,
            user_query="Test question",
            system_response="Test answer",
            retrieved_chunks=[]
        )
        
        # Verify it exists
        summary = memory.get_conversation_summary(conv_id)
        assert summary['total_interactions'] == 1
        
        # Clear it
        success = memory.clear_conversation(conv_id)
        assert success is True
        
        # Verify it's gone
        summary = memory.get_conversation_summary(conv_id)
        assert summary == {}
    
    def test_memory_stats(self, memory):
        """Test memory statistics."""
        # Add some interactions
        for i in range(3):
            memory.add_interaction(
                conversation_id=f"conv_{i}",
                user_query=f"Question {i}",
                system_response=f"Answer {i}",
                retrieved_chunks=[]
            )
        
        stats = memory.get_stats()
        assert stats['total_conversations'] == 3
        assert stats['total_interactions'] == 3


if __name__ == "__main__":
    pytest.main([__file__])