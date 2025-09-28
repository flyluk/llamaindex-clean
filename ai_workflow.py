#!/usr/bin/env python3

import sys
import os
import json
from pathlib import Path
from docling.document_converter import DocumentConverter
from structure_detect_n_extract import detect_and_extract
from document_indexer import DocumentIndexer
import tempfile
import shutil

class AIWorkflow:
    def __init__(self, db_path="./workflow_db", model="gpt-oss:20b"):
        self.db_path = db_path
        self.model = model
        self.converter = DocumentConverter()
    
    def convert_to_markdown(self, file_path):
        """Convert file to markdown using Docling"""
        result = self.converter.convert(file_path)
        markdown_content = result.document.export_to_markdown()
        cleaned_content = self._clean_markdown(markdown_content)
        
        # Save cleaned content to markdown file
        md_file = f"{os.path.splitext(file_path)[0]}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        print(f"Cleaned markdown saved to: {md_file}")
        
        return cleaned_content
    
    def _clean_markdown(self, markdown_content):
        """Remove alphabetical letter patterns from markdown"""
        import re
        
        lines = markdown_content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines with just single letters A-V
            if re.match(r'^[A-V]$', line.strip()):
                continue
            # Skip lines that are sequences of A-V letters separated by whitespace
            if re.match(r'^[A-V](?:\s+[A-V])*\s*$', line.strip()):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def process_file(self, file_path):
        """Complete workflow: convert → extract → ingest"""
        print(f"Processing: {file_path}")
        
        # Convert to markdown
        print("Converting to markdown...")
        markdown_content = self.convert_to_markdown(file_path)
        
        # Extract structure
        print("Detecting and extracting structure...")
        structure = detect_and_extract(markdown_content)
        
        # Initialize indexer
        indexer = DocumentIndexer(target_dir="", db_path=self.db_path, model=self.model)
        
        # Create structured documents
        documents = indexer.create_structured_documents(structure)
        print(f"Created {len(documents)} structured documents")
        
        # Load or create index
        indexer.load_or_create_index()
        
        # Insert documents
        for i, doc in enumerate(documents, 1):
            print(f"Ingesting {i}/{len(documents)}: {doc.metadata['hierarchy_path']}")
            indexer.index.insert(doc)
        
        # Output documents to JSON
        doc_data = []
        for doc in documents:
            doc_data.append({
                'text': doc.text,
                'metadata': doc.metadata
            })
        
        with open('ingest.json', 'w', encoding='utf-8') as f:
            json.dump(doc_data, f, indent=2, ensure_ascii=False)
        
        print(f"Workflow completed. Database: {os.path.abspath(self.db_path)}")
        print(f"Documents exported to ingest.json")
        return len(documents)

def main():
    if len(sys.argv) < 2:
        print("Usage: python ai_workflow.py <file_path> [db_path] [model]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    db_path = sys.argv[2] if len(sys.argv) > 2 else "./workflow_db"
    model = sys.argv[3] if len(sys.argv) > 3 else "gpt-oss:20b"
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    workflow = AIWorkflow(db_path, model)
    workflow.process_file(file_path)

if __name__ == "__main__":
    main()