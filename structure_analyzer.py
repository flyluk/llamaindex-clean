#!/usr/bin/env python3
"""
Structure Analyzer - Class to detect and extract document structure from markdown files
"""
import os
from structure_detect_n_extract import StructureDetector, extract_structure

class StructureAnalyzer:
    def __init__(self, md_file_path):
        self.md_file_path = md_file_path
        self.content = None
        self.structure = None
        self.detector = StructureDetector()
    
    def load_file(self):
        """Load markdown file content"""
        if not os.path.exists(self.md_file_path):
            raise FileNotFoundError(f"File not found: {self.md_file_path}")
        
        with open(self.md_file_path, 'r', encoding='utf-8') as f:
            self.content = f.read()
        return self.content
    
    def detect_structure(self):
        """Detect and extract structure from loaded markdown content"""
        if self.content is None:
            self.load_file()
        
        detection_result = self.detector.detect_structure(self.content)
        self.structure = extract_structure(self.content)
        
        return {
            'structure': self.structure,
            'patterns': detection_result['patterns'],
            'hierarchy': detection_result['hierarchy']
        }
    
    def get_structure_summary(self):
        """Get summary of detected structure"""
        if self.structure is None:
            self.detect_structure()
        
        summary = {
            'total_elements': len(self.structure),
            'types': {},
            'hierarchy_depth': 0
        }
        
        for item in self.structure:
            item_type = item.get('type', 'unknown')
            summary['types'][item_type] = summary['types'].get(item_type, 0) + 1
        
        return summary

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python structure_analyzer.py <markdown_file>")
        sys.exit(1)
    
    analyzer = StructureAnalyzer(sys.argv[1])
    structure = analyzer.detect_structure()
    summary = analyzer.get_structure_summary()
    
    print(f"Structure Analysis for: {sys.argv[1]}")
    print(f"Total elements: {summary['total_elements']}")
    print(f"Element types: {summary['types']}")