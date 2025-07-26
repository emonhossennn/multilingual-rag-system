#!/usr/bin/env python3
"""
Main entry point for the Multilingual RAG System

Provides command-line interface for different modes of operation:
- Interactive demo
- API server
- Document processing
- System evaluation
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add src to Python path
sys.path.append(str(Path(__file__).parent))

from src.rag_system import MultilingualRAGSystem
from api.main import start_server
from evaluation.evaluator import RAGEvaluator
from evaluation.test_datasets import create_test_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_environment():
    """Load environment variables from .env file if it exists."""
    try:
        from dotenv import load_dotenv
        env_path = Path('.env')
        if env_path.exists():
            load_dotenv(env_path)
            logger.info("Environment variables loaded from .env file")
        else:
            logger.info("No .env file found, using system environment variables")
    except ImportError:
        logger.warning("python-dotenv not installed, skipping .env file loading")


def initialize_rag_system(args) -> MultilingualRAGSystem:
    """Initialize the RAG system with configuration."""
    logger.info("🚀 Initializing Multilingual RAG System...")
    
    # Get configuration from environment or use defaults
    config = {
        'persist_directory': args.vector_db_path or os.getenv('VECTOR_DB_PATH', './chroma_db'),
        'embedding_model': args.embedding_model or os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
        'chunk_size': args.chunk_size or int(os.getenv('CHUNK_SIZE', '512')),
        'chunk_overlap': args.chunk_overlap or int(os.getenv('CHUNK_OVERLAP', '50')),
        'max_retrieved_chunks': args.max_chunks or int(os.getenv('MAX_RETRIEVED_CHUNKS', '5'))
    }
    
    logger.info(f"Configuration: {config}")
    
    # Initialize RAG system
    rag_system = MultilingualRAGSystem(**config)
    
    # Load documents if specified
    if args.documents:
        for doc_path in args.documents:
            if Path(doc_path).exists():
                logger.info(f"📚 Loading document: {doc_path}")
                success = rag_system.index_document(doc_path)
                if success:
                    logger.info(f"✅ Successfully indexed {doc_path}")
                else:
                    logger.error(f"❌ Failed to index {doc_path}")
            else:
                logger.warning(f"Document not found: {doc_path}")
    
    # Check if system has any documents
    stats = rag_system.get_system_stats()
    total_docs = stats.get('vector_store', {}).get('total_documents', 0)
    
    if total_docs == 0:
        logger.info("📝 No documents found, loading sample data...")
        load_sample_data(rag_system)
    
    logger.info("✅ RAG system initialized successfully!")
    return rag_system


def load_sample_data(rag_system: MultilingualRAGSystem):
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
    
    রবীন্দ্রনাথ ঠাকুর বাংলা সাহিত্যের অন্যতম শ্রেষ্ঠ কবি ও সাহিত্যিক।
    তিনি ১৯১৩ সালে সাহিত্যে নোবেল পুরস্কার পান। তার বিখ্যাত কাব্যগ্রন্থ গীতাঞ্জলি।
    
    বাংলা ব্যাকরণে সন্ধি একটি গুরুত্বপূর্ণ বিষয়। দুটি বর্ণের মিলনকে সন্ধি বলে।
    উপসর্গ শব্দের আগে বসে নতুন অর্থ সৃষ্টি করে।
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


def run_interactive_demo(rag_system: MultilingualRAGSystem):
    """Run interactive demo mode."""
    print("\n" + "="*60)
    print("🚀 Multilingual RAG System - Interactive Demo")
    print("="*60)
    print("Ask questions in Bengali or English!")
    print("Commands:")
    print("  'quit' or 'exit' - Exit the demo")
    print("  'stats' - Show system statistics")
    print("  'help' - Show this help message")
    print("="*60)
    
    conversation_id = "interactive_demo"
    
    while True:
        try:
            # Get user input
            query = input("\n💬 আপনার প্রশ্ন (Your question): ").strip()
            
            # Handle commands
            if query.lower() in ['quit', 'exit', 'q', 'বের হও']:
                print("👋 ধন্যবাদ! Thank you!")
                break
            
            if query.lower() in ['stats', 'পরিসংখ্যান']:
                stats = rag_system.get_system_stats()
                print(f"\n📊 System Statistics:")
                print(f"   📚 Documents: {stats.get('vector_store', {}).get('total_documents', 0)}")
                print(f"   💬 Conversations: {stats.get('memory', {}).get('total_conversations', 0)}")
                print(f"   🔄 Interactions: {stats.get('memory', {}).get('total_interactions', 0)}")
                print(f"   🤖 Model: {stats.get('embedding_model', {}).get('model_name', 'N/A')}")
                continue
            
            if query.lower() in ['help', 'সাহায্য']:
                print("\n📖 Available Commands:")
                print("   • Ask any question in Bengali or English")
                print("   • 'stats' - Show system statistics")
                print("   • 'quit' - Exit the demo")
                print("   • 'help' - Show this help")
                continue
            
            if not query:
                continue
            
            # Process query
            print("🤔 প্রক্রিয়াকরণ... (Processing...)")
            
            response = rag_system.query(
                user_query=query,
                conversation_id=conversation_id
            )
            
            # Display response
            print(f"\n🤖 উত্তর (Answer): {response.answer}")
            print(f"📊 আত্মবিশ্বাস (Confidence): {response.confidence_score:.2f}")
            print(f"🌐 ভাষা (Language): {response.query_language}")
            print(f"⏱️ সময় (Time): {response.processing_time:.2f}s")
            
            if response.retrieved_chunks:
                print(f"📚 {len(response.retrieved_chunks)}টি প্রাসঙ্গিক অংশ পাওয়া গেছে")
                
                # Show top chunk if confidence is low
                if response.confidence_score < 0.5 and response.retrieved_chunks:
                    print(f"📄 প্রসঙ্গ: {response.retrieved_chunks[0]['document'][:150]}...")
            
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Error in interactive demo: {e}")


def run_evaluation(rag_system: MultilingualRAGSystem, args):
    """Run system evaluation."""
    logger.info("🧪 Starting RAG system evaluation...")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag_system, output_dir=args.output_dir or "evaluation_results")
    
    # Create or load test dataset
    if args.test_dataset:
        logger.info(f"Loading test dataset from: {args.test_dataset}")
        from evaluation.test_datasets import load_test_dataset
        test_queries = load_test_dataset(args.test_dataset)
    else:
        logger.info("Using default HSC Bangla test dataset")
        test_queries = create_test_dataset()
    
    # Run evaluation
    results = evaluator.comprehensive_evaluation(
        test_queries=test_queries,
        save_results=True
    )
    
    # Display summary
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("="*60)
    
    agg_metrics = results.get('aggregate_metrics', {})
    print(f"Total Queries: {results['total_queries']}")
    print(f"Success Rate: {agg_metrics.get('success_rate', 0):.1%}")
    print(f"Average Groundedness: {agg_metrics.get('avg_groundedness', 0):.3f}")
    print(f"Average Relevance: {agg_metrics.get('avg_relevance', 0):.3f}")
    print(f"Average Confidence: {agg_metrics.get('avg_confidence', 0):.3f}")
    print(f"Average Processing Time: {agg_metrics.get('avg_processing_time', 0):.3f}s")
    print(f"Queries per Second: {agg_metrics.get('queries_per_second', 0):.2f}")
    
    # Category performance
    category_perf = results.get('category_performance', {})
    if category_perf:
        print(f"\n📈 Performance by Category:")
        for category, metrics in category_perf.items():
            print(f"  {category}: G={metrics['avg_groundedness']:.3f}, R={metrics['avg_relevance']:.3f}, C={metrics['avg_confidence']:.3f}")
    
    print(f"\n📁 Detailed results saved to: {evaluator.output_dir}")
    print("="*60)


def run_api_server(args):
    """Run the REST API server."""
    logger.info("🌐 Starting REST API server...")
    
    host = args.host or os.getenv('API_HOST', '0.0.0.0')
    port = args.port or int(os.getenv('API_PORT', '8000'))
    reload = args.reload if args.reload is not None else os.getenv('DEBUG', 'False').lower() == 'true'
    
    start_server(host=host, port=port, reload=reload)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multilingual RAG System - Bengali & English Q&A System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s demo                                    # Interactive demo
  %(prog)s demo --documents data/book.pdf          # Demo with custom document
  %(prog)s api                                     # Start REST API server
  %(prog)s api --port 8080                         # API on custom port
  %(prog)s evaluate                                # Run evaluation
  %(prog)s evaluate --test-dataset tests.json     # Evaluate with custom dataset
  %(prog)s process data/book.pdf                   # Process document only
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Demo mode
    demo_parser = subparsers.add_parser('demo', help='Run interactive demo')
    demo_parser.add_argument('--documents', nargs='+', help='PDF documents to load')
    demo_parser.add_argument('--vector-db-path', help='Vector database path')
    demo_parser.add_argument('--embedding-model', help='Embedding model name')
    demo_parser.add_argument('--chunk-size', type=int, help='Document chunk size')
    demo_parser.add_argument('--chunk-overlap', type=int, help='Chunk overlap size')
    demo_parser.add_argument('--max-chunks', type=int, help='Max retrieved chunks')
    
    # API mode
    api_parser = subparsers.add_parser('api', help='Start REST API server')
    api_parser.add_argument('--host', help='API host address')
    api_parser.add_argument('--port', type=int, help='API port number')
    api_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    api_parser.add_argument('--documents', nargs='+', help='PDF documents to load')
    api_parser.add_argument('--vector-db-path', help='Vector database path')
    api_parser.add_argument('--embedding-model', help='Embedding model name')
    api_parser.add_argument('--chunk-size', type=int, help='Document chunk size')
    api_parser.add_argument('--chunk-overlap', type=int, help='Chunk overlap size')
    api_parser.add_argument('--max-chunks', type=int, help='Max retrieved chunks')
    
    # Evaluation mode
    eval_parser = subparsers.add_parser('evaluate', help='Run system evaluation')
    eval_parser.add_argument('--test-dataset', help='Path to test dataset JSON file')
    eval_parser.add_argument('--output-dir', help='Output directory for results')
    eval_parser.add_argument('--documents', nargs='+', help='PDF documents to load')
    eval_parser.add_argument('--vector-db-path', help='Vector database path')
    eval_parser.add_argument('--embedding-model', help='Embedding model name')
    eval_parser.add_argument('--chunk-size', type=int, help='Document chunk size')
    eval_parser.add_argument('--chunk-overlap', type=int, help='Chunk overlap size')
    eval_parser.add_argument('--max-chunks', type=int, help='Max retrieved chunks')
    
    # Process mode
    process_parser = subparsers.add_parser('process', help='Process documents only')
    process_parser.add_argument('documents', nargs='+', help='PDF documents to process')
    process_parser.add_argument('--vector-db-path', help='Vector database path')
    process_parser.add_argument('--embedding-model', help='Embedding model name')
    process_parser.add_argument('--chunk-size', type=int, help='Document chunk size')
    process_parser.add_argument('--chunk-overlap', type=int, help='Chunk overlap size')
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Load environment
    load_environment()
    
    # Default to demo mode if no mode specified
    if not args.mode:
        args.mode = 'demo'
        args.documents = None
    
    try:
        if args.mode == 'api':
            # API mode doesn't need RAG system initialization here
            # It's handled in the API startup
            run_api_server(args)
        
        elif args.mode in ['demo', 'evaluate', 'process']:
            # Initialize RAG system
            rag_system = initialize_rag_system(args)
            
            if args.mode == 'demo':
                run_interactive_demo(rag_system)
            elif args.mode == 'evaluate':
                run_evaluation(rag_system, args)
            elif args.mode == 'process':
                print(f"✅ Processed {len(args.documents)} documents successfully!")
                stats = rag_system.get_system_stats()
                print(f"📊 Total documents in system: {stats.get('vector_store', {}).get('total_documents', 0)}")
        
        else:
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()