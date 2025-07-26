"""
Multilingual RAG System Package

A complete Retrieval-Augmented Generation system for Bengali and English queries.
"""

__version__ = "1.0.0"
__author__ = "RAG System Developer"

from .rag_system import MultilingualRAGSystem, RAGResponse
from .vector_store import VectorStore
from .embeddings import EmbeddingGenerator
from .memory import ConversationMemory
from .chunker import DocumentChunker, Chunk
from .pdf_processor import PDFProcessor
from .llm_client import LLMClient

__all__ = [
    "MultilingualRAGSystem",
    "RAGResponse", 
    "VectorStore",
    "EmbeddingGenerator",
    "ConversationMemory",
    "DocumentChunker",
    "Chunk",
    "PDFProcessor",
    "LLMClient"
]