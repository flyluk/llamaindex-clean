#!/usr/bin/env python3

import sys
import os
from pathlib import Path

try:
    import win32com.client
except ImportError:
    print("Error: pywin32 not installed. Install with: pip install pywin32")
    sys.exit(1)

def convert_doc_to_docx(doc_path, docx_path=None):
    """Convert DOC file to DOCX format"""
    doc_path = Path(doc_path)
    
    if not doc_path.exists():
        raise FileNotFoundError(f"File not found: {doc_path}")
    
    if docx_path is None:
        docx_path = doc_path.with_suffix('.docx')
    else:
        docx_path = Path(docx_path)
    
    # Remove existing DOCX file if it exists
    if docx_path.exists():
        docx_path.unlink()
    
    # Initialize Word application
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    
    try:
        # Open DOC file
        doc = word.Documents.Open(str(doc_path.absolute()))
        
        # Save as DOCX (format code 16 = docx)
        doc.SaveAs2(str(docx_path.absolute()), FileFormat=16)
        doc.Close()
        
        print(f"Converted: {doc_path} -> {docx_path}")
        
    finally:
        word.Quit()

def convert_folder(folder_path):
    """Convert all DOC files in folder to DOCX"""
    folder = Path(folder_path)
    doc_files = list(folder.glob('*.doc'))
    
    if not doc_files:
        print(f"No .doc files found in {folder}")
        return
    
    print(f"Found {len(doc_files)} .doc files")
    
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    
    try:
        for doc_path in doc_files:
            docx_path = doc_path.with_suffix('.docx')
            try:
                # Remove existing DOCX file if it exists
                if docx_path.exists():
                    docx_path.unlink()
                
                doc = word.Documents.Open(str(doc_path.absolute()))
                doc.SaveAs2(str(docx_path.absolute()), FileFormat=16)
                doc.Close()
                print(f"Converted: {doc_path.name} -> {docx_path.name}")
            except Exception as e:
                print(f"Failed to convert {doc_path.name}: {e}")
    finally:
        word.Quit()

def main():
    if len(sys.argv) < 2:
        print("Usage: python doc_to_docx.py <input.doc|folder> [output.docx]")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    try:
        if input_path.is_dir():
            convert_folder(input_path)
        elif input_path.suffix.lower() == '.doc':
            output_file = sys.argv[2] if len(sys.argv) > 2 else None
            convert_doc_to_docx(input_path, output_file)
        else:
            print(f"Error: {input_path} is not a .doc file or directory")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()