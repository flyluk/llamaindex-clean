#!/usr/bin/env python3
from document_indexer import DocumentIndexer
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Index documents using LlamaIndex')
    parser.add_argument('--target-dir', '-t', required=True, help='Directory to index')
    parser.add_argument('--db-path', '-d', default='./chroma_db', help='Database path')
    parser.add_argument('--model', '-m', default='gpt-oss:20b', help='Model to use')
    parser.add_argument('--sync', '-s', action='store_true', help='Sync new files only')
    args = parser.parse_args()
    
    if not os.path.exists(args.target_dir):
        print(f"Error: Directory not found: {args.target_dir}")
        return
    
    indexer = DocumentIndexer(target_dir=args.target_dir, db_path=args.db_path, model=args.model)
    
    indexer.init_and_process_files()
    
    if args.sync:
        print("Syncing new files...")
        indexer.sync_new_files()
        print("Sync completed")
    else:
        print("Indexing completed")

if __name__ == "__main__":
    main()