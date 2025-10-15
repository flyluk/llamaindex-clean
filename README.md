# LlamaIndex with Ollama, ChromaDB, and Docling

A document indexing and querying system using LlamaIndex with Ollama backend, ChromaDB for persistent storage, and Docling for advanced document parsing.

## Features

- **Ollama Backend**: Local LLM inference with multiple model support
- **Local Embeddings**: Uses nomic-embed-text for embeddings
- **Persistent Storage**: ChromaDB for vector storage
- **Advanced Parsing**: Docling for PDF, Word, and complex document formats
- **Hybrid Search**: Query fusion with multiple query generation and reciprocal reranking
- **Chat History**: Conversational context maintained across queries with RAG prompt template
- **Clickable Chat Messages**: Click user messages to populate search box for easy re-querying
- **LLM-Powered Follow-ups**: Auto-generated contextual follow-up questions after each response
- **Smart Citations**: Inline source citations with proper document references
- **Large Document Support**: Smart chunking for large files
- **Auto-Summary Generation**: Automatically creates summaries when missing
- **Structured Document Processing**: Hierarchical extraction and RAG ingestion
- **Legal Document Auto-Detection**: Automatic detection and structure extraction for legal documents
- **Real-time Processing**: Live output streaming during document processing
- **Multiple Knowledge Bases**: Create and manage separate knowledge bases
- **Search History**: Collapsible history with chat context and model tracking
- **JSON File Persistence**: Search history with chat context saved to JSON file
- **Web Interface**: Enhanced Streamlit frontend with improved UX
- **Django Alternative**: Modern Django web interface with dark theme and progress indicators
- **Context Length Configuration**: Adjustable context length (8K-128K) with save button in search interface
- **Flexible Document Processing**: Choose between structure extraction or sentence splitter during upload
- **Custom Model Pulling**: Pull any Ollama model by name with automatic config integration
- **Model Management**: Visual progress bars for Ollama model pulling and deletion with improved error handling
- **Unified Collections**: Single ChromaDB collection for all LLM services (Ollama, OpenAI, Azure)
- **Chunk Browser**: Interactive document chunk viewer for database inspection
- **CLI Tools**: Document indexing, structure extraction, and database administration
- **Docker Support**: Containerized deployment with GPU support
- **Persistent Configuration**: Server URLs and preferences saved automatically
- **Structure Extraction**: JSON output for document hierarchy analysis
- **Error Handling**: Improved URL validation and Chrome DevTools compatibility

## Prerequisites

1. **Install Ollama**
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Pull Required Models**
   ```bash
   ollama pull gpt-oss:20b
   ollama pull nomic-embed-text
   ```

3. **Install Docling Server (Optional)**
   ```bash
   pip install docling-serve
   ```

## Installation

### Local Installation
```bash
pip install -r requirements.txt
```

### Docker Installation
```bash
# Build and run with Docker Compose
docker-compose up --build

# Access Django interface at http://localhost:8000
```

## Usage

### Web Interface

**Streamlit (Original)**
```bash
streamlit run streamlit_app.py
```

**Django (Alternative)**
```bash
cd django_app
python manage.py runserver
# Access at http://localhost:8000
```

### Chunk Browser

The Django interface includes a chunk browser for inspecting document storage:

1. **Navigate to Chunks tab** in the web interface
2. **Select a document** from the left panel to view its chunks
3. **Select a chunk** from the middle panel to view full content and metadata
4. **Inspect metadata** including hierarchy paths and document structure

### CLI Tools

**Index Documents**
```bash
./index_doc_cli.py -t /path/to/docs [-d db_path] [-m model] [--sync]
```

**Structure Extraction**
```bash
# Extract document structure to JSON
python structure_extractor.py input.md [output.json]
```

**Database Administration**
```bash
./admin_cli.py summaries [db_path] [--model model]
./admin_cli.py stats [db_path] [--model model]
./admin_cli.py clear-db [db_path]
```

**Structured Document Processing**
```bash
# Extract structure from markdown to text
./extract_to_text.py input.md output.txt

# Ingest structured documents with hierarchy
./structured_ingest.py ingest document.md [db_path] [model]

# Query with structure awareness
./structured_ingest.py query "What are share transfer requirements?" [db_path]
```

**Model Examples**
```bash
# Different Ollama models
./index_doc_cli.py -t /docs --model llama3.2:3b
./index_doc_cli.py -t /docs --model qwen2.5:7b
./admin_cli.py summaries --model deepseek-r1:14b
```

## Configuration

The application uses `config.json` for unified service configuration:

```json
{
  "base_url": "http://localhost:11434",
  "embed_url": "http://localhost:11434",
  "api_key": "",
  "default_model": "deepseek-r1:14b",
  "embed_model": "nomic-embed-text",
  "default_kb": "default"
}
```

### Service Auto-Detection
- **Ollama**: No API key required, uses local endpoints
- **Azure OpenAI**: Detected by "azure" in URLs, requires API key
- **OpenAI**: API key provided with non-Azure URLs

### Configuration Examples

**Ollama (Local)**
```json
{
  "base_url": "http://localhost:11434",
  "api_key": null,
  "default_model": "deepseek-r1:14b",
  "embed_model": "nomic-embed-text"
}
```

**Azure OpenAI**
```json
{
  "base_url": "https://your-resource.cognitiveservices.azure.com/",
  "embed_url": "https://your-resource.cognitiveservices.azure.com/",
  "api_key": "your-azure-api-key",
  "default_model": "gpt-4o",
  "embed_model": "text-embedding-3-small"
}
```

**OpenAI**
```json
{
  "base_url": "https://api.openai.com/v1",
  "api_key": "your-openai-api-key",
  "default_model": "gpt-4o",
  "embed_model": "text-embedding-3-small"
}
```

### Configuration Options
- **base_url**: LLM service endpoint
- **embed_url**: Embedding service endpoint (defaults to base_url)
- **api_key**: Authentication key (null for Ollama)
- **default_model**: LLM model name
- **embed_model**: Embedding model name
- **default_kb**: Default knowledge base
- **context_length**: Model context length (8K-128K tokens, configurable in search interface)
- **ollama_library**: Array of available Ollama models (auto-updated when pulling custom models)
- **embed_library**: Array of available embedding models
- **Auto-Save**: Settings persist between sessions
- **Search History**: Stored in `history.json` with 50-entry limit

## Docker Deployment

### Features
- **Django Interface**: Runs Django web interface by default (port 8000)
- **Auto-initialization**: Creates default chroma_db on first startup
- **GPU Support**: NVIDIA GPU acceleration for faster processing
- **Pre-cached Models**: Docling models downloaded during build
- **Volume Persistence**: Data and config preserved between runs
- **Network Tools**: Built-in utilities for troubleshooting

### Commands
```bash
# Standard deployment (Django interface)
docker-compose up --build

# With GPU support (requires nvidia-docker)
docker-compose up --build

# Test Docling PDF conversion
python test_docling.py sample.pdf
```

## Systemd Service

### Docling Server Service
Run Docling server as a systemd service:

```bash
# Install and start service
./install-docling-service.sh

# Manual service management
sudo systemctl start docling
sudo systemctl stop docling
sudo systemctl status docling
```

**Service Features:**
- Auto-restart on failure
- GPU support (CUDA device 1)
- Verbose logging
- Port 5001 (configurable)
- Runs as user 'fly'

## Dynamic Structure Detection

### Auto-Detection Patterns
- **Legal Documents**: Schedule, Part, Division, Subdivision, Section
- **Academic Papers**: Chapter, Section, Subsection
- **Technical Docs**: Article-based hierarchy
- **Markdown**: Standard headers (#, ##, ###)
- **Mixed Formats**: Combines multiple pattern types

### Supported Structures
- **Schedule**: Alphanumeric (1, 5A, 5B) with optional content
- **Part**: Numbered sections within schedules or standalone
- **Division**: Subdivisions within parts
- **Chapter**: Academic/book chapters
- **Article**: Article-based documents
- **Headers**: Markdown headers (H1-H6)

### Smart Processing
- Analyzes document to detect structure patterns
- Adapts extraction based on detected format
- Preserves hierarchical relationships
- Creates structured metadata for enhanced retrieval
- Fallback to markdown headers when no patterns found

### RAG Benefits
- **Precise References**: Exact section citations
- **Context Preservation**: Parent-child content relationships
- **Format Agnostic**: Works with any document structure
- **Enhanced Search**: Structure-aware retrieval

## Search Modes

- **Hybrid Search (Default)**: Query fusion with 4 query variations and reciprocal reranking for enhanced retrieval accuracy
- **Vector Search Only**: Pure vector search (checkbox option: "Use vector search only")
- **Chat History Context**: Previous conversation context included in queries
- **Clickable Messages**: Click any user message in chat to populate search box
- **Context Length Control**: Adjustable context length (8K-128K) with save functionality in search interface
- **Follow-up Prompts**: LLM-generated contextual follow-up questions as clickable buttons
- **Section Search**: Direct section number matching
- **Legal Document Search**: Structure-aware search for legal documents
- **Smart Citations**: Automatic inline citations with [source_id] format

## Document Processing Options

- **Structure Extraction (Default)**: Hierarchical document parsing with metadata preservation
  - Best for formal documents (legal, academic, technical)
  - Preserves document hierarchy (schedules, parts, sections, chapters)
  - Enhanced retrieval with structure-aware search
- **Sentence Splitter**: Standard sentence-based chunking
  - Best for general text and informal documents
  - Faster processing without structure analysis
  - Selectable via checkbox during document upload

## Model Support

### Ollama Models
- `gpt-oss:20b`
- `llama3.2:3b`
- `llama3.1:8b` 
- `qwen2.5:7b`
- `deepseek-r1:14b` (default)
- `deepseek-r1:8b`
- Any Ollama model

### Azure OpenAI Models
- `DeepSeek-R1`
- `gpt-4o`
- `gpt-4`
- `gpt-35-turbo`
- Any deployed Azure model

### OpenAI Models
- `gpt-4o`
- `gpt-4-turbo`
- `gpt-3.5-turbo`
- Any OpenAI model

### Embedding Models
- **Ollama**: `nomic-embed-text`, `mxbai-embed-large`
- **Azure/OpenAI**: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`

### Cost Tracking

- **Input tokens**: $0.00135 per 1000 tokens
- **Output tokens**: $0.0054 per 1000 tokens
- **Real-time cost estimation** displayed during queries
- **Token counting** based on character length approximation

## Architecture

- **DocumentIndexer**: Core class with hybrid search, chat history, and flexible processing modes
- **RAG Prompt Template**: Structured prompts with citation guidelines and chat context
- **Query Fusion**: Multiple query generation with custom prompts for better retrieval
- **Chat History Management**: Persistent conversation context across sessions
- **Smart Citations**: Automatic source_id tagging for inline document references
- **Multi-Service Support**: Ollama, OpenAI, and Azure OpenAI with automatic detection
- **Unified Collections**: Single ChromaDB collection across all LLM services
- **Docling Integration**: Advanced document parsing with pre-cached models
- **Enhanced Search**: Hybrid retrieval with query fusion and reciprocal reranking
- **Smart Chunking**: Adaptive chunking based on document size and processing mode
- **Flexible Processing**: Structure extraction or sentence splitter selectable per document
- **Auto-Recovery**: Generates missing summaries automatically
- **Structured Processing**: Hierarchical document extraction with metadata
- **Legal Document Auto-Detection**: Automatic pattern recognition for legal document structures
- **Real-time Feedback**: Live processing output with line-by-line updates
- **Multi-KB Support**: Multiple knowledge base management with context tracking
- **Enhanced UI**: Collapsible search history, server configuration, and follow-up prompts
- **Custom Model Support**: Pull any Ollama model with automatic config integration
- **Follow-up Generation**: LLM-powered contextual follow-up questions
- **Content Extraction**: Markdown cleaning and content preservation
- **Structure Validation**: JSON schema validation and extraction method comparison
- **Persistent Config**: Automatic saving of user preferences and server settings
- **Improved Error Handling**: JSON response validation and better debug output

## File Structure

```
├── streamlit_app.py              # Original Streamlit web interface
├── django_app/                   # Django web interface (alternative)
│   ├── manage.py                 # Django management script
│   ├── llamaindex_app/           # Main Django application
│   │   ├── views.py              # Django views with DocumentIndexer integration
│   │   ├── urls.py               # URL routing with Chrome DevTools handler
│   │   └── templates/            # HTML templates with progress bars
│   └── requirements.txt          # Django-specific dependencies
├── test_docling.py               # Docling PDF conversion test script
├── index_doc_cli.py              # Document indexing CLI
├── admin_cli.py                  # Database administration CLI
├── document_indexer.py           # Core DocumentIndexer class
├── structure_detect_n_extract.py # Hierarchical structure extraction and detection
├── structure_extractor.py        # JSON structure extraction tool
├── extract_to_text.py            # Structure to text conversion
├── structured_ingest.py          # Structured RAG ingestion
├── extraction_patterns.py        # Document type classification patterns
├── test/                         # Testing utilities
│   ├── test_structure_comparison.py  # Structure extraction testing
│   └── structure_schema.json         # JSON schema for structure validation
├── docker-compose.yml            # Docker deployment configuration
├── Dockerfile                    # Container build instructions
├── docling.service               # Systemd service configuration for docling-serve
├── install-docling-service.sh    # Systemd service installation script
├── config.json                   # Persistent application configuration (URL validation)
├── history.json                  # Search history (JSON file persistence)
├── requirements.txt              # Dependencies
├── README.md                    # This file
└── chroma_db/                   # Default ChromaDB storage
```