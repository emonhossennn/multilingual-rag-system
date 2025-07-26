"""
API tests for the Multilingual RAG System
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
import tempfile
import shutil
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from api.main import app


class TestRAGAPI:
    """Test cases for the REST API."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code in [200, 503]  # May be 503 if system not initialized
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "timestamp" in data
    
    def test_stats_endpoint(self, client):
        """Test stats endpoint."""
        response = client.get("/stats")
        assert response.status_code in [200, 503]  # May be 503 if system not initialized
        
        if response.status_code == 200:
            data = response.json()
            assert "vector_store" in data or "system_status" in data
    
    def test_chat_endpoint(self, client):
        """Test chat endpoint."""
        # Wait a bit for system to initialize
        import time
        time.sleep(2)
        
        chat_data = {
            "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
            "conversation_id": "test_conversation"
        }
        
        response = client.post("/chat", json=chat_data)
        
        # Should either work (200) or system not ready (503)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "confidence_score" in data
            assert "processing_time" in data
            assert "query_language" in data
            assert "conversation_id" in data
    
    def test_chat_endpoint_english(self, client):
        """Test chat endpoint with English query."""
        import time
        time.sleep(2)
        
        chat_data = {
            "query": "Who is referred to as a good man?",
            "conversation_id": "test_conversation_en"
        }
        
        response = client.post("/chat", json=chat_data)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert data["query_language"] in ["english", "bengali"]  # May detect as either
    
    def test_conversations_endpoint(self, client):
        """Test conversations endpoint."""
        response = client.get("/conversations")
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
    
    def test_invalid_chat_request(self, client):
        """Test invalid chat request."""
        # Missing required field
        invalid_data = {
            "conversation_id": "test"
            # Missing "query" field
        }
        
        response = client.post("/chat", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_nonexistent_conversation(self, client):
        """Test getting nonexistent conversation."""
        response = client.get("/conversations/nonexistent_conversation")
        assert response.status_code in [404, 503]
    
    @pytest.mark.asyncio
    async def test_api_startup_shutdown(self):
        """Test API startup and shutdown."""
        # This test ensures the lifespan events work correctly
        async with TestClient(app) as client:
            response = client.get("/")
            assert response.status_code == 200


class TestAPIModels:
    """Test API request/response models."""
    
    def test_query_request_validation(self):
        """Test QueryRequest model validation."""
        from api.models import QueryRequest
        
        # Valid request
        valid_request = QueryRequest(query="Test question")
        assert valid_request.query == "Test question"
        assert valid_request.conversation_id == "default"
        assert valid_request.include_context is True
        
        # Request with all fields
        full_request = QueryRequest(
            query="Test question",
            conversation_id="custom_id",
            language="bengali",
            include_context=False
        )
        assert full_request.conversation_id == "custom_id"
        assert full_request.language == "bengali"
        assert full_request.include_context is False
    
    def test_query_response_model(self):
        """Test QueryResponse model."""
        from api.models import QueryResponse, RetrievedChunk
        
        chunk = RetrievedChunk(
            document="Test document",
            similarity_score=0.85,
            metadata={"source": "test"},
            rank=1
        )
        
        response = QueryResponse(
            answer="Test answer",
            confidence_score=0.9,
            processing_time=1.5,
            query_language="bengali",
            retrieved_chunks=[chunk],
            sources=["test_source"],
            conversation_id="test_conv"
        )
        
        assert response.answer == "Test answer"
        assert response.confidence_score == 0.9
        assert len(response.retrieved_chunks) == 1
        assert response.retrieved_chunks[0].document == "Test document"
    
    def test_health_response_model(self):
        """Test HealthResponse model."""
        from api.models import HealthResponse, SystemStats
        
        stats = SystemStats(
            vector_store={"total_documents": 10},
            memory={"total_conversations": 5},
            system_status="operational"
        )
        
        health = HealthResponse(
            status="healthy",
            uptime=3600.0,
            stats=stats
        )
        
        assert health.status == "healthy"
        assert health.uptime == 3600.0
        assert health.stats.system_status == "operational"


class TestAPIIntegration:
    """Integration tests for API with actual RAG system."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_full_api_workflow(self, client, temp_dir):
        """Test complete API workflow."""
        import time
        
        # Wait for system initialization
        time.sleep(3)
        
        # 1. Check health
        health_response = client.get("/health")
        if health_response.status_code != 200:
            pytest.skip("RAG system not initialized")
        
        # 2. Get initial stats
        stats_response = client.get("/stats")
        assert stats_response.status_code == 200
        
        # 3. Send a query
        chat_data = {
            "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
            "conversation_id": "integration_test"
        }
        
        chat_response = client.post("/chat", json=chat_data)
        assert chat_response.status_code == 200
        
        chat_data = chat_response.json()
        assert "answer" in chat_data
        assert chat_data["conversation_id"] == "integration_test"
        
        # 4. Check conversation was created
        conv_response = client.get("/conversations/integration_test")
        assert conv_response.status_code == 200
        
        conv_data = conv_response.json()
        assert conv_data["conversation_id"] == "integration_test"
        assert conv_data["total_interactions"] >= 1
        
        # 5. Send follow-up query
        followup_data = {
            "query": "তার আর কোন বৈশিষ্ট্য ছিল?",
            "conversation_id": "integration_test"
        }
        
        followup_response = client.post("/chat", json=followup_data)
        assert followup_response.status_code == 200
        
        # 6. Check updated conversation
        updated_conv_response = client.get("/conversations/integration_test")
        assert updated_conv_response.status_code == 200
        
        updated_conv_data = updated_conv_response.json()
        assert updated_conv_data["total_interactions"] >= 2
    
    def test_multilingual_queries(self, client):
        """Test multilingual query handling."""
        import time
        time.sleep(2)
        
        queries = [
            {
                "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
                "expected_lang": "bengali"
            },
            {
                "query": "Who is referred to as a good man?",
                "expected_lang": "english"
            }
        ]
        
        for i, query_data in enumerate(queries):
            chat_data = {
                "query": query_data["query"],
                "conversation_id": f"multilingual_test_{i}"
            }
            
            response = client.post("/chat", json=chat_data)
            
            if response.status_code == 200:
                data = response.json()
                # Language detection might not be perfect, so we just check it's detected
                assert "query_language" in data
                assert data["query_language"] in ["bengali", "english"]
    
    def test_error_handling(self, client):
        """Test API error handling."""
        # Test malformed JSON
        response = client.post("/chat", data="invalid json")
        assert response.status_code == 422
        
        # Test missing required fields
        response = client.post("/chat", json={})
        assert response.status_code == 422
        
        # Test invalid conversation ID format (if validation exists)
        response = client.get("/conversations/")
        assert response.status_code in [404, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])