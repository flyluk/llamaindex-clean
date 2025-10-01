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
- **Django Alternative**: Modern Django web interface with dark theme
- **CLI Tools**: Document indexing, structure extraction, and database administration
- **Docker Support**: Containerized deployment with GPU support
- **Persistent Configuration**: Server URLs and preferences saved automatically
- **Structure Extraction**: JSON output for document hierarchy analysis

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

# Access at http://localhost:8501
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

The application uses `config.json` for persistent settings:

```json
{
  "embed_base_url": "http://localhost:11434",
  "ollama_base_url": "http://localhost:11434",
  "default_model": "deepseek-r1:14b",
  "default_kb": "default"
}
```

- **Server URLs**: Configure embedding and Ollama server endpoints
- **Default Model**: Auto-select preferred model on startup
- **Default KB**: Remember last used knowledge base
- **Auto-Save**: Settings persist between sessions
- **Search History**: Stored in `history.json` with 50-entry limit

## Docker Deployment

### Features
- **GPU Support**: NVIDIA GPU acceleration for faster processing
- **Pre-cached Models**: Docling models downloaded during build
- **Volume Persistence**: Data and config preserved between runs
- **Network Tools**: Built-in utilities for troubleshooting

### Commands
```bash
# Standard deployment
docker-compose up --build

# With GPU support (requires nvidia-docker)
docker-compose up --build
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
- `gpt-oss:20b` (default)
- `llama3.2:3b`
- `llama3.1:8b` 
- `qwen2.5:7b`
- `deepseek-r1:14b`
- `deepseek-r1:8b`
- Any Ollama model

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
│   │   ├── urls.py               # URL routing
│   │   └── templates/            # HTML templates with dark theme
│   └── requirements.txt          # Django-specific dependencies
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
├── config.json                   # Persistent application configuration
├── history.json                  # Search history (JSON file persistence)
├── requirements.txt              # Dependencies
├── README.md                    # This file
└── chroma_db/                   # Default ChromaDB storage
```