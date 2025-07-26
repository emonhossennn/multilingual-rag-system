"""
FastAPI REST API for Multilingual RAG System
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_system import MultilingualRAGSystem
from .models import (
    QueryRequest, QueryResponse, ConversationResponse, 
    SystemStats, HealthResponse, DocumentUpload, DocumentUploadResponse,
    EvaluationRequest, EvaluationResponse, RetrievedChunk
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
rag_system: Optional[MultilingualRAGSystem] = None
startup_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global rag_system, startup_time
    
    # Startup
    logger.info("🚀 Starting Multilingual RAG API...")
    startup_time = time.time()
    
    try:
        # Initialize RAG system
        rag_system = MultilingualRAGSystem(
            persist_directory=os.getenv("VECTOR_DB_PATH", "./chroma_db"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            max_retrieved_chunks=int(os.getenv("MAX_RETRIEVED_CHUNKS", "5"))
        )
        
        # Load sample data if no documents exist
        stats = rag_system.get_system_stats()
        if stats.get('vector_store', {}).get('total_documents', 0) == 0:
            logger.info("📚 Loading sample data...")
            await load_sample_data()
        
        logger.info("✅ RAG system initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down RAG API...")


# Create FastAPI app
app = FastAPI(
    title="Multilingual RAG System API",
    description="REST API for Bengali-English Retrieval-Augmented Generation System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def load_sample_data():
    """Load sample Bengali data for testing."""
    sample_text = """
    অনুপম একজন সাধারণ মানুষ ছিল। তার জীবনে অনেক ঘটনা ঘটেছিল। 
    সে তার মামার কাছে অনেক কিছু শিখেছিল। মামা ছিলেন তার ভাগ্য দেবতা।
    
    শুম্ভুনাথ ছিলেন একজন সুপুরুষ। অনুপমের ভাষায় তিনি ছিলেন আদর্শ মানুষ।
    তার চরিত্রে অনেক ভালো গুণ ছিল। তিনি ছিলেন একজন ভদ্রলোক।
    
    কল্যাণীর বিয়ের সময় তার বয়স ছিল মাত্র পনেরো বছর। 
    সে ছিল খুবই সুন্দর এবং বুদ্ধিমতী একটি মেয়ে। তার পরিবার ছিল সম্ভ্রান্ত।
    
    গল্পে দেখা যায় যে, অনুপম তার জীবনে অনেক সংগ্রাম করেছে।
    তার মামা তাকে জীবনের অনেক শিক্ষা দিয়েছেন। এই শিক্ষাগুলো তার জীবনে কাজে লেগেছে।
    
    বিয়ের পর কল্যাণী একজন আদর্শ স্ত্রী হয়ে উঠেছিল।
    সে তার স্বামীর সাথে সুখে থাকত। তাদের সংসার ছিল শান্তিপূর্ণ।
    """
    
    # Create chunks and add to system
    from src.chunker import DocumentChunker
    chunker = DocumentChunker(chunk_size=256, chunk_overlap=25)
    
    chunks = chunker.chunk_document(
        text=sample_text,
        strategy="paragraph",
        metadata={"source": "sample_data", "type": "demo", "language": "bengali"}
    )
    
    # Add to vector store
    rag_system.vector_store.add_chunks(chunks)
    logger.info(f"✅ Loaded {len(chunks)} sample chunks")


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Multilingual RAG System API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "chat": "POST /chat",
            "health": "GET /health", 
            "stats": "GET /stats",
            "conversations": "GET /conversations/{conversation_id}",
            "upload": "POST /upload",
            "evaluate": "POST /evaluate"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Get system stats
        stats = rag_system.get_system_stats()
        
        # Calculate uptime
        uptime = time.time() - startup_time
        
        return HealthResponse(
            status="healthy",
            uptime=uptime,
            stats=SystemStats(**stats)
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """Main chat endpoint for querying the RAG system."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Query the RAG system
        response = rag_system.query(
            user_query=request.query,
            conversation_id=request.conversation_id,
            include_context=request.include_context
        )
        
        # Convert retrieved chunks to response format
        retrieved_chunks = [
            RetrievedChunk(
                document=chunk['document'],
                similarity_score=chunk['similarity_score'],
                metadata=chunk.get('metadata', {}),
                rank=i + 1
            )
            for i, chunk in enumerate(response.retrieved_chunks)
        ]
        
        return QueryResponse(
            answer=response.answer,
            confidence_score=response.confidence_score,
            processing_time=response.processing_time,
            query_language=response.query_language,
            retrieved_chunks=retrieved_chunks,
            sources=response.sources,
            conversation_id=request.conversation_id,
            metadata=response.metadata
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/stats", response_model=SystemStats)
async def get_stats():
    """Get comprehensive system statistics."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        stats = rag_system.get_system_stats()
        return SystemStats(**stats)
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str):
    """Get conversation history for a specific conversation."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        summary = rag_system.memory.get_conversation_summary(conversation_id)
        
        if not summary:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return ConversationResponse(**summary)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting conversation: {str(e)}")


@app.get("/conversations", response_model=List[ConversationResponse])
async def get_all_conversations():
    """Get all conversation summaries."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        conversations = rag_system.memory.get_all_conversations()
        return [ConversationResponse(**conv) for conv in conversations]
        
    except Exception as e:
        logger.error(f"Error getting conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting conversations: {str(e)}")


@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: Optional[str] = None
):
    """Upload and process a PDF document."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Parse metadata if provided
        doc_metadata = {}
        if metadata:
            import json
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON, using empty metadata")
        
        # Add filename to metadata
        doc_metadata['filename'] = file.filename
        doc_metadata['uploaded_via_api'] = True
        
        # Process document in background
        def process_document():
            try:
                start_time = time.time()
                success = rag_system.index_document(str(file_path), doc_metadata)
                processing_time = time.time() - start_time
                
                if success:
                    logger.info(f"✅ Successfully processed {file.filename} in {processing_time:.2f}s")
                else:
                    logger.error(f"❌ Failed to process {file.filename}")
                
                # Clean up uploaded file
                file_path.unlink(missing_ok=True)
                
            except Exception as e:
                logger.error(f"Error processing document {file.filename}: {e}")
                file_path.unlink(missing_ok=True)
        
        background_tasks.add_task(process_document)
        
        return DocumentUploadResponse(
            success=True,
            message=f"Document {file.filename} uploaded and queued for processing",
            document_id=file.filename,
            chunks_created=None,  # Will be determined during processing
            processing_time=None   # Will be determined during processing
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_system(request: EvaluationRequest):
    """Run evaluation on the RAG system."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Import evaluation system
        from evaluation.evaluator import RAGEvaluator
        
        evaluator = RAGEvaluator(rag_system)
        
        # Run evaluation
        results = evaluator.comprehensive_evaluation(request.test_queries)
        
        # Prepare response
        response_data = {
            "total_queries": results['total_queries'],
            "avg_groundedness": results['aggregate_metrics']['avg_groundedness'],
            "avg_relevance": results['aggregate_metrics']['avg_relevance'],
            "avg_confidence": results['aggregate_metrics']['avg_confidence'],
            "avg_processing_time": results['aggregate_metrics']['avg_processing_time'],
            "category_performance": results['category_performance']
        }
        
        if request.include_detailed_results:
            response_data["detailed_results"] = results['individual_results']
        
        return EvaluationResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Error running evaluation: {str(e)}")


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a specific conversation."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        success = rag_system.memory.clear_conversation(conversation_id)
        
        if success:
            return {"message": f"Conversation {conversation_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {str(e)}")


@app.post("/reset")
async def reset_system():
    """Reset the entire RAG system (clear all data)."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        success = rag_system.reset_system()
        
        if success:
            # Reload sample data
            await load_sample_data()
            return {"message": "System reset successfully and sample data reloaded"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reset system")
            
    except Exception as e:
        logger.error(f"Error resetting system: {e}")
        raise HTTPException(status_code=500, detail=f"Error resetting system: {str(e)}")


def start_server(host: str = None, port: int = None, reload: bool = None):
    """Start the FastAPI server."""
    host = host or os.getenv("API_HOST", "0.0.0.0")
    port = port or int(os.getenv("API_PORT", "8000"))
    reload = reload if reload is not None else os.getenv("DEBUG", "False").lower() == "true"
    
    logger.info(f"🌐 Starting API server on http://{host}:{port}")
    logger.info(f"📖 API Documentation: http://localhost:{port}/docs")
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    start_server()