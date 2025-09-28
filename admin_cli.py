#!/usr/bin/env python3
from document_indexer import DocumentIndexer
import argparse
import os



def main():
    parser = argparse.ArgumentParser(description='Database administration')
    parser.add_argument('command', choices=['summaries', 'stats', 'clear-db', 'delete'], help='Command to execute')
    parser.add_argument('--file', '-f', help='File path to delete (for delete command)')
    parser.add_argument('--db-path', '-d', default='./chroma_db', help='Database path')
    parser.add_argument('--model', '-m', default='gpt-oss:20b', help='Model to use')
    args = parser.parse_args()
    
    indexer = DocumentIndexer(target_dir="", db_path=args.db_path, model=args.model)
    
    if args.command == "summaries":
        indexer.load_or_create_index()
        indexer.display_all_summaries()
    
    elif args.command == "stats":
        indexer.load_or_create_index()
        doc_count = len(indexer.doc_collection.get()["ids"])
        summary_count = len(indexer.summary_collection.get()["ids"])
        print(f"Database: {os.path.abspath(args.db_path)}")
        print(f"Documents: {doc_count}")
        print(f"Summaries: {summary_count}")
    
    elif args.command == "delete":
        if not args.file:
            print("Error: --file argument required for delete command")
            return
        indexer.load_or_create_index()
        if indexer.delete_file(args.file):
            print(f"Successfully deleted: {args.file}")
        else:
            print(f"File not found in database: {args.file}")
    
    elif args.command == "clear-db":
        if os.path.exists(args.db_path):
            import shutil
            shutil.rmtree(args.db_path)
            print(f"Cleared database: {args.db_path}")
        else:
            print(f"Database not found: {args.db_path}")

if __name__ == "__main__":
    main()