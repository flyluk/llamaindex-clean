#!/usr/bin/env python3

import sys
from structure_detect_n_extract import extract_structure
from document_indexer import DocumentIndexer
from llama_index.core import Document, Settings
import os
import json



def ingest_structured_content(markdown_file, db_path="./structured_db", model="gpt-oss:20b"):
    """Ingest structured content into RAG system"""
    
    # Extract structure from markdown
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    structure = extract_structure(markdown_content)
    
    # Initialize indexer to use its method
    temp_indexer = DocumentIndexer(target_dir="", db_path=db_path, model=model)
    
    # Create structured documents
    documents = temp_indexer.create_structured_documents(structure)
    
    print(f"Created {len(documents)} structured documents")
    
    
    # Initialize indexer
    indexer = DocumentIndexer(target_dir="", db_path=db_path, model=model)
    indexer.load_or_create_index()
    
    # Insert documents with structure-aware summaries
    for i, doc in enumerate(documents, 1):
        print(f"Processing {i}/{len(documents)}: {doc.metadata['hierarchy_path']}")
        
        # Insert document
        indexer.index.insert(doc)
    
    # Dump documents to JSON
    doc_data = []
    for doc in documents:
        doc_data.append({
            'text': doc.text,
            'metadata': doc.metadata
        })
    
    with open('ingest.json', 'w', encoding='utf-8') as f:
        json.dump(doc_data, f, indent=2, ensure_ascii=False)
    
    print(f"Ingestion completed. Database: {os.path.abspath(db_path)}")
    print(f"Documents dumped to ingest.json")

def query_structured_content(query, db_path="./structured_db", model="gpt-oss:20b", use_direct=False, similarity_top_k=5):
    """Query structured content with hierarchy awareness"""
    
    # Clean up corrupted database if needed
    import shutil
    if os.path.exists(db_path):
        try:
            # Test if database is accessible
            test_indexer = DocumentIndexer(target_dir="", db_path=db_path, model=model)
        except Exception as e:
            if "_type" in str(e) or "KeyError" in str(e):
                print(f"Corrupted database detected, removing: {db_path}")
                shutil.rmtree(db_path)
    
    indexer = DocumentIndexer(target_dir="", db_path=db_path, model=model)
    indexer.load_or_create_index()
    
    return indexer.query(query, use_direct=use_direct, similarity_top_k=similarity_top_k)

def tune_parameters(db_path="./structured_db", model="gpt-oss:20b"):
    """Tune query parameters for optimal performance"""
    indexer = DocumentIndexer(target_dir="", db_path=db_path, model=model)
    indexer.load_or_create_index()
    return indexer.tune_parameters()

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Ingest: python structured_ingest.py ingest <markdown_file> [db_path] [model]")
        print("  Query:  python structured_ingest.py query <query_text> [db_path] [model] [--direct] [--top-k N]")
        print("  Tune:   python structured_ingest.py tune [db_path] [model]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "ingest":
        if len(sys.argv) < 3:
            print("Error: markdown file required for ingestion")
            sys.exit(1)
        
        markdown_file = sys.argv[2]
        db_path = sys.argv[3] if len(sys.argv) > 3 else "./structured_db"
        model = sys.argv[4] if len(sys.argv) > 4 else "gpt-oss:20b"
        
        ingest_structured_content(markdown_file, db_path, model)
        
    elif command == "query":
        if len(sys.argv) < 3:
            print("Error: query text required")
            sys.exit(1)
        
        query_text = sys.argv[2]
        db_path = sys.argv[3] if len(sys.argv) > 3 else "./structured_db"
        model = sys.argv[4] if len(sys.argv) > 4 else "gpt-oss:20b"
        use_direct = "--direct" in sys.argv
        
        # Parse top-k parameter
        similarity_top_k = 5
        if "--top-k" in sys.argv:
            try:
                top_k_idx = sys.argv.index("--top-k")
                similarity_top_k = int(sys.argv[top_k_idx + 1])
            except (IndexError, ValueError):
                print("Error: --top-k requires a numeric value")
                sys.exit(1)
        
        result = query_structured_content(query_text, db_path, model, use_direct, similarity_top_k)
        print(f"\nAnswer:\n{result}")
        
    elif command == "tune":
        db_path = sys.argv[2] if len(sys.argv) > 2 else "./structured_db"
        model = sys.argv[3] if len(sys.argv) > 3 else "gpt-oss:20b"
        
        best_params = tune_parameters(db_path, model)
        print(f"\nOptimal parameters: {best_params}")
        
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()