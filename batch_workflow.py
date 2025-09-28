#!/usr/bin/env python3

import sys
import os
from pathlib import Path
from ai_workflow import AIWorkflow
import glob

def process_directory(directory_path, db_path="./batch_db", model="gpt-oss:20b", file_patterns=None):
    """Process all files in directory matching patterns"""
    
    if file_patterns is None:
        file_patterns = ["*.pdf", "*.docx", "*.doc", "*.txt", "*.md"]
    
    workflow = AIWorkflow(db_path, model)
    processed_count = 0
    total_docs = 0
    
    for pattern in file_patterns:
        files = glob.glob(os.path.join(directory_path, "**", pattern), recursive=True)
        
        for file_path in files:
            try:
                print(f"\n{'='*60}")
                docs_created = workflow.process_file(file_path)
                processed_count += 1
                total_docs += docs_created
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Batch processing completed:")
    print(f"Files processed: {processed_count}")
    print(f"Total documents created: {total_docs}")
    print(f"Database: {os.path.abspath(db_path)}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_workflow.py <directory_path> [db_path] [model]")
        print("Supported formats: PDF, DOCX, DOC, TXT, MD")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    db_path = sys.argv[2] if len(sys.argv) > 2 else "./batch_db"
    model = sys.argv[3] if len(sys.argv) > 3 else "gpt-oss:20b"
    
    if not os.path.exists(directory_path):
        print(f"Error: Directory not found: {directory_path}")
        sys.exit(1)
    
    process_directory(directory_path, db_path, model)

if __name__ == "__main__":
    main()