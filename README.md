# 🚀 Multilingual RAG System (Bengali-English)

A comprehensive Retrieval-Augmented Generation system designed for Bengali and English queries, specifically optimized for HSC Bangla 1st Paper content and educational materials.

## ✨ Features

### 🌐 Core Capabilities
- **Multilingual Support**: Seamless handling of Bengali and English queries
- **PDF Processing**: Advanced Bengali text extraction from PDF documents
- **Semantic Search**: Context-aware document retrieval using multilingual embeddings
- **Conversation Memory**: Maintains chat history and context across interactions
- **Real-time Processing**: Fast query processing with confidence scoring

### 🎯 Bonus Features
- **REST API**: Complete FastAPI-based web service with OpenAPI documentation
- **Comprehensive Evaluation**: Advanced metrics for groundedness, relevance, and quality
- **Industry Integration**: Support for multiple vector databases and LLM providers
- **Performance Analytics**: Detailed system monitoring and performance metrics

## 🏗️ Architecture

```
User Query (EN/BN) → Language Detection → Embedding Generation → Vector Search
                                                                        ↓
Response Generation ← Context Assembly ← Chunk Retrieval ← Similarity Matching
        ↓
Conversation Memory ← Confidence Scoring ← Answer Validation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB+ RAM (for embedding models)
- 2GB+ disk space

### Installation

1. **Clone and Setup**
```bash
git clone https://github.com/emonhossennn/multilingual-rag-system.git
cd multilingual-rag-system
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Environment Configuration**
```bash
cp .env.example .env
# Edit .env with your API keys (optional)
```

### 🎮 Usage Modes

#### 1. Interactive Demo
```bash
python main.py demo
```
- Real-time Q&A in Bengali/English
- Automatic sample data loading
- Conversation history tracking

#### 2. REST API Server
```bash
python main.py api
```
- Full REST API at `http://localhost:8000`
- Interactive docs at `http://localhost:8000/docs`
- Production-ready with async support

#### 3. Document Processing
```bash
python main.py process data/your-document.pdf
```
- Batch PDF processing
- Automatic text extraction and chunking
- Vector database indexing

#### 4. System Evaluation
```bash
python main.py evaluate
```
- Comprehensive performance analysis
- Multiple evaluation metrics
- Detailed reporting

## 📖 API Documentation

### Core Endpoints

#### Chat Query
```http
POST /chat
Content-Type: application/json

{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
  "conversation_id": "user_session_123",
  "include_context": true
}
```

**Response:**
```json
{
  "answer": "শুম্ভুনাথ",
  "confidence_score": 0.92,
  "processing_time": 0.45,
  "query_language": "bengali",
  "retrieved_chunks": [...],
  "sources": ["sample_data"],
  "conversation_id": "user_session_123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### System Health
```http
GET /health
```

#### Upload Document
```http
POST /upload
Content-Type: multipart/form-data

file: your-document.pdf
metadata: {"category": "literature", "subject": "bangla"}
```

#### Run Evaluation
```http
POST /evaluate
Content-Type: application/json

{
  "test_queries": [...],
  "include_detailed_results": true
}
```

## 🧪 Evaluation Metrics

### Groundedness (0-1)
Measures how well answers are supported by retrieved context:
- **Keyword Overlap**: Direct word matching between answer and context
- **Quote Detection**: Identification of direct quotes from source material
- **Semantic Similarity**: Conceptual alignment between answer and context

### Relevance (0-1)
Evaluates quality of document retrieval:
- **Similarity Scores**: Vector similarity between query and retrieved chunks
- **Query-Document Overlap**: Keyword matching between query and documents
- **Ranking Quality**: Position-based relevance scoring

### Answer Quality (0-1)
Comprehensive answer assessment:
- **Length Appropriateness**: Optimal answer length for question type
- **Language Consistency**: Proper language detection and usage
- **Completeness**: Whether answer fully addresses the question
- **Coherence**: Logical flow and readability

### Performance Metrics
- **Processing Time**: Query response latency
- **Throughput**: Queries processed per second
- **Success Rate**: Percentage of successful query processing
- **Resource Usage**: Memory and CPU utilization

## 🎯 Sample Queries & Expected Results

### Bengali Literature Questions
```
Q: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
A: শুম্ভুনাথ
Confidence: 0.95

Q: কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
A: মামাকে
Confidence: 0.88

Q: বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
A: ১৫ বছর
Confidence: 0.92
```

### Cross-Language Queries
```
Q: Who is referred to as a good man in Anupam's language?
A: Shumbhunath
Confidence: 0.87

Q: What was Kalyani's age at marriage?
A: 15 years
Confidence: 0.90
```

### Grammar Questions
```
Q: সন্ধি কাকে বলে?
A: দুটি বর্ণের মিলনকে সন্ধি বলে
Confidence: 0.85
```

## 🔧 Configuration

### Environment Variables
```bash
# Core Configuration
VECTOR_DB_PATH=./chroma_db
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_RETRIEVED_CHUNKS=5

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Optional: External LLM Integration
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
```

### Advanced Configuration
```python
# Custom RAG system initialization
rag_system = MultilingualRAGSystem(
    persist_directory="./custom_db",
    embedding_model="custom-model-name",
    chunk_size=256,
    chunk_overlap=25,
    max_retrieved_chunks=3
)
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Test Categories
```bash
# Unit tests
pytest tests/test_rag_system.py -v

# API tests
pytest tests/test_api.py -v

# Integration tests
pytest tests/ -k "integration" -v
```

### Test Coverage
```bash
pip install pytest-cov
pytest tests/ --cov=src --cov-report=html
```

## 📊 Performance Benchmarks

### Typical Performance (on standard hardware)
- **Query Processing**: 0.3-1.5 seconds
- **Document Indexing**: 2-5 seconds per page
- **Memory Usage**: 1-2GB for standard models
- **Throughput**: 10-50 queries per second

### Optimization Tips
1. **Use GPU**: Significantly faster embedding generation
2. **Batch Processing**: Process multiple documents together
3. **Chunk Size Tuning**: Balance between context and speed
4. **Model Selection**: Smaller models for faster inference

## 🔌 Integration Examples

### Python Integration
```python
from src.rag_system import MultilingualRAGSystem

# Initialize system
rag = MultilingualRAGSystem()

# Add documents
rag.index_document("data/bangla-literature.pdf")

# Query system
response = rag.query("অনুপমের চরিত্র কেমন?")
print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence_score}")
```

### API Integration
```python
import requests

# Query via API
response = requests.post("http://localhost:8000/chat", json={
    "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
    "conversation_id": "my_session"
})

result = response.json()
print(f"Answer: {result['answer']}")
```

### Evaluation Integration
```python
from evaluation.evaluator import RAGEvaluator
from evaluation.test_datasets import create_test_dataset

# Run evaluation
evaluator = RAGEvaluator(rag_system)
test_queries = create_test_dataset()
results = evaluator.comprehensive_evaluation(test_queries)

print(f"Average Groundedness: {results['aggregate_metrics']['avg_groundedness']:.3f}")
```

## 🛠️ Tools, Libraries & Packages Used

### Core Dependencies
- **sentence-transformers==2.2.2**: Multilingual embedding generation
- **chromadb==0.4.15**: Vector database for semantic search
- **PyMuPDF==1.23.5**: PDF text extraction with Bengali support
- **langdetect==1.0.9**: Automatic language detection
- **fastapi==0.104.1**: Modern REST API framework
- **uvicorn==0.24.0**: ASGI server for API deployment
- **nltk==3.8.1**: Natural language processing utilities
- **numpy==1.24.3**: Numerical computations for embeddings
- **scikit-learn==1.3.0**: Machine learning utilities and metrics

### Development & Testing
- **pytest==7.4.3**: Testing framework
- **black==23.9.1**: Code formatting
- **flake8==6.1.0**: Code linting
- **python-dotenv==1.0.0**: Environment variable management

### Optional Integrations
- **openai==1.3.5**: OpenAI API integration
- **requests==2.31.0**: HTTP client for external APIs
- **matplotlib==3.7.2**: Visualization for evaluation
- **seaborn==0.12.2**: Statistical plotting

## 📋 Technical Implementation Details

For detailed technical documentation including:
- PDF extraction methodology and challenges
- Chunking strategy rationale
- Embedding model selection criteria
- Similarity comparison methods
- Query handling for vague/missing context
- Performance optimization strategies

**See**: [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)

## 🛠️ Development

### Project Structure
```
multilingual-rag-system/
├── src/                    # Core RAG system
│   ├── rag_system.py      # Main system orchestrator
│   ├── vector_store.py    # ChromaDB integration
│   ├── embeddings.py      # Multilingual embeddings
│   ├── memory.py          # Conversation management
│   ├── chunker.py         # Document processing
│   ├── pdf_processor.py   # PDF text extraction
│   └── llm_client.py      # LLM integration
├── api/                   # REST API
│   ├── main.py           # FastAPI application
│   └── models.py         # Pydantic models
├── evaluation/           # Evaluation system
│   ├── evaluator.py     # Main evaluator
│   ├── metrics.py       # Evaluation metrics
│   └── test_datasets.py # Test data management
├── tests/               # Test suite
├── main.py             # CLI entry point
└── requirements.txt    # Dependencies
```

### Adding New Features

1. **New Evaluation Metric**
```python
# evaluation/metrics.py
class CustomMetric(BaseMetric):
    def calculate(self, **kwargs) -> float:
        # Your metric implementation
        return score
```

2. **New API Endpoint**
```python
# api/main.py
@app.post("/custom-endpoint")
async def custom_endpoint(request: CustomRequest):
    # Your endpoint implementation
    return response
```

3. **New Document Processor**
```python
# src/processors/custom_processor.py
class CustomProcessor:
    def process(self, file_path: str) -> str:
        # Your processing logic
        return extracted_text
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest tests/ -v`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings for all public functions
- Maintain test coverage above 80%

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Sentence Transformers**: Multilingual embedding models
- **ChromaDB**: Vector database infrastructure
- **FastAPI**: Modern web framework
- **PyMuPDF**: PDF processing capabilities
- **HSC Bangla Curriculum**: Educational content inspiration

## 📞 Support

- **Documentation**: Check the `/docs` endpoint when running the API
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join community discussions
- **Email**: Contact the development team

## 🔮 Roadmap

### Version 2.0 (Planned)
- [ ] Advanced LLM integration (GPT-4, Claude)
- [ ] Real-time collaborative features
- [ ] Enhanced Bengali NLP processing
- [ ] Mobile app integration
- [ ] Advanced analytics dashboard

### Version 1.5 (In Progress)
- [ ] Improved evaluation metrics
- [ ] Better error handling
- [ ] Performance optimizations
- [ ] Extended language support

---

**Built with ❤️ for Bengali education and multilingual AI**#   m u l t i l i n g u a l - r a g - s y s t e m  
 