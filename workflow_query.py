#!/usr/bin/env python3

import sys
from document_indexer import DocumentIndexer

def query_workflow_db(query, db_path="./workflow_db", model="gpt-oss:20b", use_direct=False, similarity_top_k=5):
    """Query workflow database"""
    indexer = DocumentIndexer(target_dir="", db_path=db_path, model=model)
    indexer.load_or_create_index()
    return indexer.query(query, use_direct=use_direct, similarity_top_k=similarity_top_k)

def main():
    if len(sys.argv) < 2:
        print("Usage: python workflow_query.py <query> [db_path] [model] [--direct] [--top-k N]")
        sys.exit(1)
    
    query_text = sys.argv[1]
    db_path = sys.argv[2] if len(sys.argv) > 2 else "./workflow_db"
    model = sys.argv[3] if len(sys.argv) > 3 else "gpt-oss:20b"
    use_direct = "--direct" in sys.argv
    
    similarity_top_k = 5
    if "--top-k" in sys.argv:
        try:
            top_k_idx = sys.argv.index("--top-k")
            similarity_top_k = int(sys.argv[top_k_idx + 1])
        except (IndexError, ValueError):
            print("Error: --top-k requires a numeric value")
            sys.exit(1)
    
    result = query_workflow_db(query_text, db_path, model, use_direct, similarity_top_k)
    print(f"\nAnswer:\n{result}")

if __name__ == "__main__":
    main()