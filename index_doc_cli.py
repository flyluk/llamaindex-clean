#!/usr/bin/env python3
from document_indexer import DocumentIndexer
from llama_index.core import Document
from structure_detect_n_extract import extract_structure, StructureDetector
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Index documents using LlamaIndex')
    parser.add_argument('target', help='File or directory to index')
    parser.add_argument('--db-path', '-d', default='./chroma_db', help='Database path')
    parser.add_argument('--model', '-m', default='gpt-oss:20b', help='Model to use')
    parser.add_argument('--sync', '-s', action='store_true', help='Sync new files only')
    args = parser.parse_args()
    
    if not os.path.exists(args.target):
        print(f"Error: Path not found: {args.target}")
        return
    
    # Handle both files and directories
    if os.path.isfile(args.target):
        # For individual files, use parent directory as target_dir
        target_dir = os.path.dirname(args.target)
        print(f"Indexing file: {args.target}")
        indexer = DocumentIndexer(target_dir=target_dir, db_path=args.db_path, model=args.model)
        indexer.load_or_create_index()
        indexer.process_single_file(args.target)
    else:
        # For directories, use existing logic
        print(f"Indexing directory: {args.target}")
        indexer = DocumentIndexer(target_dir=args.target, db_path=args.db_path, model=args.model)
        indexer.init_and_process_files()
        
        if args.sync:
            print("Syncing new files...")
            indexer.sync_new_files()
            print("Sync completed")
    
    print("Indexing completed")

if __name__ == "__main__":
    main()