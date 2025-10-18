# DocumentIndexer Refactoring

## Overview
The DocumentIndexer class has been refactored from a monolithic 800+ line class into modular components for better maintainability and testability.

## New Structure

### 1. `llm_factory.py` - LLM and Embedding Creation
- **LLMFactory**: Factory class for creating LLM and embedding instances
- Handles Ollama, OpenAI, and Azure OpenAI detection
- Centralizes service type detection logic
- Methods:
  - `create_llm()`: Create LLM instance based on configuration
  - `create_embedding()`: Create embedding instance
  - `setup_settings()`: Setup global LlamaIndex settings

### 2. `document_processor.py` - Document Processing
- **DocumentProcessor**: Handles document conversion and structure detection
- Static methods for document operations
- Methods:
  - `convert_with_docling()`: Convert documents to markdown
  - `detect_legal_document()`: Detect legal document patterns
  - `process_document()`: Process single document with structure detection
  - `create_structured_documents()`: Create hierarchical documents
  - `create_general_structured_documents()`: Create general structured documents

### 3. `storage_manager.py` - Storage Management
- **StorageManager**: Manages ChromaDB collections and operations
- Handles collection creation, file tracking, and deletion
- Methods:
  - `get_processed_files()`: Get processed file paths
  - `get_completed_files()`: Get completed file paths
  - `mark_file_complete()`: Mark file as complete
  - `delete_file()`: Delete file from all collections
  - `get_status()`: Get complete document status

### 4. `document_indexer_refactored.py` - Main Indexer
- **DocumentIndexer**: Simplified main class using modular components
- Reduced from 800+ lines to ~350 lines
- Delegates to specialized classes
- Cleaner separation of concerns

## Benefits

### Maintainability
- Each class has a single responsibility
- Easier to locate and fix bugs
- Clearer code organization

### Testability
- Components can be tested independently
- Easier to mock dependencies
- Better unit test coverage

### Reusability
- Components can be used in other projects
- Factory pattern allows easy extension
- Storage manager can be used standalone

### Readability
- Smaller, focused classes
- Clear method names and purposes
- Better documentation structure

## Migration Path

### Option 1: Gradual Migration (Recommended)
1. Keep `document_indexer.py` as is
2. New features use `document_indexer_refactored.py`
3. Gradually migrate existing code
4. Remove old file when complete

### Option 2: Direct Replacement
1. Backup `document_indexer.py`
2. Rename `document_indexer_refactored.py` to `document_indexer.py`
3. Test all functionality
4. Remove backup if successful

## Compatibility

The refactored version maintains the same public API:
- All public methods have the same signatures
- Same initialization parameters
- Same return values
- Backward compatible with existing code

## Testing

Test the refactored version:
```bash
# Test document processing
python -c "from document_indexer_refactored import DocumentIndexer; indexer = DocumentIndexer(); indexer.load_or_create_index()"

# Test with Django
# Update views.py to import from document_indexer_refactored
```

## Future Improvements

1. **Add unit tests** for each component
2. **Add type hints** throughout
3. **Add async support** for parallel processing
4. **Extract query logic** into separate QueryEngine class
5. **Add caching layer** for frequently accessed data
6. **Add logging** instead of print statements
