# LlamaIndex with Ollama, ChromaDB, and Docling

A document indexing and querying system using LlamaIndex with Ollama backend, ChromaDB for persistent storage, and Docling for advanced document parsing.

## Features

- **Ollama Backend**: Local LLM inference with multiple model support
- **Local Embeddings**: Uses nomic-embed-text for embeddings
- **Persistent Storage**: ChromaDB for vector storage
- **Advanced Parsing**: Docling for PDF, Word, and complex document formats
- **Smart Search**: Sentence-first keyword search with word fallback, section, and vector search
- **Large Document Support**: Smart chunking for large files
- **Auto-Summary Generation**: Automatically creates summaries when missing
- **Structured Document Processing**: Hierarchical extraction and RAG ingestion
- **Legal Document Auto-Detection**: Automatic detection and structure extraction for legal documents
- **Real-time Processing**: Live output streaming during document processing
- **Multiple Knowledge Bases**: Create and manage separate knowledge bases
- **Search History**: Collapsible history with knowledge base and model tracking
- **JSON File Persistence**: Search history saved to JSON file instead of localStorage
- **Web Interface**: Enhanced Streamlit frontend with improved UX
- **Django Alternative**: Modern Django web interface with dark theme and progress indicators
- **Model Management**: Visual progress bars for Ollama model pulling and deletion
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

- **Default**: Sentence-first keyword → Word fallback → Vector search cascade
- **Direct**: Pure vector search only (checkbox option in web interface)
- **Section Search**: Direct section number matching
- **Legal Document Search**: Structure-aware search for legal documents

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

## Architecture

- **DocumentIndexer**: Core class with automatic legal document detection and structure extraction
- **Ollama Backend**: Local LLM inference with configurable server URLs
- **Docling Integration**: Advanced document parsing with pre-cached models
- **Enhanced Search**: Sentence-first keyword search with word fallback and vector search cascade
- **Smart Chunking**: Adaptive chunking based on document size
- **Auto-Recovery**: Generates missing summaries automatically
- **Structured Processing**: Hierarchical document extraction with metadata
- **Legal Document Auto-Detection**: Automatic pattern recognition for legal document structures
- **Real-time Feedback**: Live processing output with line-by-line updates
- **Multi-KB Support**: Multiple knowledge base management with context tracking
- **Enhanced UI**: Collapsible search history and server configuration
- **Content Extraction**: Markdown cleaning and content preservation
- **Structure Validation**: JSON schema validation and extraction method comparison
- **Persistent Config**: Automatic saving of user preferences and server settings

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
├── config.json                   # Persistent application configuration (URL validation)
├── history.json                  # Search history (JSON file persistence)
├── requirements.txt              # Dependencies
├── README.md                    # This file
└── chroma_db/                   # Default ChromaDB storage
```