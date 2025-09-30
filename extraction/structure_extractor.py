#!/usr/bin/env python3

from typing import Dict, Any
import os
import json
from extraction_patterns import extract_content_by_type, classify_document_type

class StructureExtractor:
    def __init__(self, input_file: str, output_file: str = None):
        self.input_file = input_file
        self.output_file = output_file or f"{os.path.splitext(input_file)[0]}_structure.json"
    
    def extract_and_save(self) -> str:
        """Extract structure from input file and save to output JSON file"""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        doc_type = classify_document_type(content, self.input_file)
        structure = extract_content_by_type(content, doc_type)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(structure, f, indent=2, ensure_ascii=False)
        
        return self.output_file
    


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python structure_extractor.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    extractor = StructureExtractor(input_file, output_file)
    output_path = extractor.extract_and_save()
    print(f"Structure extracted to: {output_path}")