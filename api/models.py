"""
Pydantic models for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class QueryRequest(BaseModel):
    """Request model for chat queries."""
    query: str = Field(..., description="User's question in Bengali or English")
    conversation_id: Optional[str] = Field(default="default", description="Conversation identifier")
    language: Optional[str] = Field(default=None, description="Preferred response language")
    include_context: Optional[bool] = Field(default=True, description="Include conversation context")


class RetrievedChunk(BaseModel):
    """Model for retrieved document chunks."""
    document: str = Field(..., description="Retrieved text content")
    similarity_score: float = Field(..., description="Similarity score with query")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    rank: int = Field(..., description="Rank in retrieval results")


class QueryResponse(BaseModel):
    """Response model for chat queries."""
    answer: str = Field(..., description="Generated answer")
    confidence_score: float = Field(..., description="Confidence in the answer")
    processing_time: float = Field(..., description="Processing time in seconds")
    query_language: str = Field(..., description="Detected query language")
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list, description="Retrieved chunks")
    sources: List[str] = Field(default_factory=list, description="Source documents")
    conversation_id: str = Field(..., description="Conversation identifier")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Response timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ConversationResponse(BaseModel):
    """Response model for conversation history."""
    conversation_id: str = Field(..., description="Conversation identifier")
    total_interactions: int = Field(..., description="Total number of interactions")
    languages_used: List[str] = Field(default_factory=list, description="Languages used in conversation")
    created_at: Optional[float] = Field(default=None, description="Conversation creation timestamp")
    last_updated: Optional[float] = Field(default=None, description="Last update timestamp")
    recent_queries: List[str] = Field(default_factory=list, description="Recent queries")
    duration_minutes: Optional[float] = Field(default=None, description="Conversation duration in minutes")


class SystemStats(BaseModel):
    """System statistics model."""
    vector_store: Dict[str, Any] = Field(default_factory=dict, description="Vector store statistics")
    memory: Dict[str, Any] = Field(default_factory=dict, description="Memory statistics")
    embedding_model: Dict[str, Any] = Field(default_factory=dict, description="Embedding model info")
    system_status: str = Field(default="operational", description="System status")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="System health status")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Health check timestamp")
    version: str = Field(default="1.0.0", description="System version")
    uptime: Optional[float] = Field(default=None, description="System uptime in seconds")
    stats: Optional[SystemStats] = Field(default=None, description="System statistics")


class DocumentUpload(BaseModel):
    """Document upload request model."""
    filename: str = Field(..., description="Document filename")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Document metadata")


class DocumentUploadResponse(BaseModel):
    """Document upload response model."""
    success: bool = Field(..., description="Upload success status")
    message: str = Field(..., description="Upload status message")
    document_id: Optional[str] = Field(default=None, description="Document identifier")
    chunks_created: Optional[int] = Field(default=None, description="Number of chunks created")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")


class EvaluationRequest(BaseModel):
    """Evaluation request model."""
    test_queries: List[Dict[str, Any]] = Field(..., description="Test queries for evaluation")
    include_detailed_results: Optional[bool] = Field(default=True, description="Include detailed results")


class EvaluationResponse(BaseModel):
    """Evaluation response model."""
    total_queries: int = Field(..., description="Total number of test queries")
    avg_groundedness: float = Field(..., description="Average groundedness score")
    avg_relevance: float = Field(..., description="Average relevance score")
    avg_confidence: float = Field(..., description="Average confidence score")
    avg_processing_time: float = Field(..., description="Average processing time")
    category_performance: Dict[str, Any] = Field(default_factory=dict, description="Performance by category")
    detailed_results: Optional[List[Dict[str, Any]]] = Field(default=None, description="Detailed evaluation results")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Evaluation timestamp")