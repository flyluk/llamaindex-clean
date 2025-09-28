# LlamaIndex with Ollama, ChromaDB, and Docling

A document indexing and querying system using LlamaIndex with Ollama backend, ChromaDB for persistent storage, and Docling for advanced document parsing.

## Features

- **Ollama Backend**: Local LLM inference with multiple model support
- **Local Embeddings**: Uses nomic-embed-text for embeddings
- **Persistent Storage**: ChromaDB for vector storage
- **Advanced Parsing**: Docling for PDF, Word, and complex document formats
- **Smart Search**: Keyword, section, and vector search with fallbacks
- **Large Document Support**: Smart chunking for large files
- **Auto-Summary Generation**: Automatically creates summaries when missing
- **Structured Document Processing**: Hierarchical extraction and RAG ingestion
- **Legal Document Support**: Schedule/Part/Division/Subdivision/Section hierarchy
- **Web Interface**: Streamlit frontend
- **CLI Tools**: Document indexing, structure extraction, and database administration

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

```bash
pip install -r requirements.txt
```

## Usage

### Web Interface
```bash
streamlit run streamlit_app.py
```

### CLI Tools

**Index Documents**
```bash
./index_doc_cli.py -t /path/to/docs [-d db_path] [-m model] [--sync]
```

**Database Administration**
```bash
./admin_cli.py summaries [db_path] [--model model]
./admin_cli.py stats [db_path] [--model model]
./admin_cli.py clear-db [db_path]
```

**AI Workflow - Complete Pipeline**
```bash
# Process single file: convert → extract → ingest
./ai_workflow.py document.pdf [db_path] [model]

# Batch process directory
./batch_workflow.py /path/to/docs [db_path] [model]

# Query workflow database
./workflow_query.py "What are the requirements?" [db_path] [model]
```

**Structure Testing & Validation**
```bash
# Compare structure extraction methods
python test/test_structure_comparison.py document.md

# Validate against JSON schema
# Outputs: mse_output.json, dae_output.json with schema validation
```

**Structured Document Processing**
```bash
# Extract structure from markdown to text
./extract_to_text.py input.md output.txt

# Ingest structured documents with hierarchy
./structured_ingest.py ingest document.md [db_path] [model]

# Query with structure awareness
./structured_ingest.py query "What are share transfer requirements?" [db_path]

# Test structure extraction consistency
python test/test_structure_comparison.py document.md
```

**Model Examples**
```bash
# Different Ollama models
./index_doc_cli.py -t /docs --model llama3.2:3b
./index_doc_cli.py -t /docs --model qwen2.5:7b
./admin_cli.py summaries --model deepseek-r1:14b
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

- **Default**: Keyword → Summary → Vector search cascade
- **Direct**: Pure vector search only
- **Section Search**: Direct section number matching

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

- **DocumentIndexer**: Core class handling all indexing, querying, and large document processing
- **Ollama Backend**: Local LLM inference with model flexibility
- **Docling Integration**: Advanced document parsing for multiple formats
- **Multi-stage Search**: Keyword → Summary → Vector search cascade with section detection
- **Smart Chunking**: Adaptive chunking based on document size
- **Auto-Recovery**: Generates missing summaries automatically
- **Structured Processing**: Hierarchical document extraction with metadata
- **Legal Document Support**: Schedule (5A, 5B), Part, Division, Subdivision, Section hierarchy
- **Content Extraction**: Markdown cleaning and content preservation
- **Structure Validation**: JSON schema validation and extraction method comparison
- **Simple CLI**: Document indexing, structure extraction, and administration tools

## File Structure

```
├── streamlit_app.py              # Web interface
├── index_doc_cli.py              # Document indexing CLI
├── admin_cli.py                  # Database administration CLI
├── document_indexer.py           # Core DocumentIndexer class
├── structure_detect_n_extract.py # Hierarchical structure extraction and detection
├── extract_to_text.py            # Structure to text conversion
├── structured_ingest.py          # Structured RAG ingestion
├── test/                         # Testing utilities
│   ├── test_structure_comparison.py  # Structure extraction testing
│   └── structure_schema.json         # JSON schema for structure validation
├── ai_workflow.py                # Complete AI workflow pipeline
├── batch_workflow.py             # Batch processing workflow
├── workflow_query.py             # Workflow database querying
├── requirements.txt              # Dependencies
├── README.md                    # This file
├── chroma_db/                   # Default ChromaDB storage
├── large_doc_db/               # Large document ChromaDB storage
├── structured_db/              # Structured document storage
├── workflow_db/                # AI workflow database storage
└── batch_db/                   # Batch processing database storage
```