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
    
    def detect_structure(self, debug=False):
        """Detect and extract structure from loaded markdown content"""
        if self.content is None:
            self.load_file()
        
        if debug:
            print(f"DEBUG: Analyzing file: {self.md_file_path}")
            print(f"DEBUG: Content length: {len(self.content)} characters")
        
        detection_result = self.detector.detect_structure(self.content)
        
        if debug:
            print(f"DEBUG: Detected patterns: {list(detection_result['patterns'].keys())}")
            print(f"DEBUG: Hierarchy: {detection_result['hierarchy']}")
            
            # Show grouped headers
            if 'grouped_headers' in detection_result:
                print(f"DEBUG: Grouped headers ({len(detection_result['grouped_headers'])} groups):")
                for i, group in enumerate(detection_result['grouped_headers']):
                    if group.get('type') == 'prelims':
                        print(f"  Group {i+1}: PRELIMS (content before first structure)")
                    elif group.get('type') == 'header_group':
                        header = group['header']
                        items = group['items']
                        print(f"  Group {i+1}: {header['text']} (Level {header['level']})")
                        for item in items:
                            print(f"    - {item['number']}. {item['text']}")
                    else:
                        print(f"  Standalone: {group.get('number', '')} {group.get('text', '')}")
        
        self.structure = extract_structure(self.content)
        
        if debug:
            print(f"DEBUG: Extracted {len(self.structure)} structural elements")
            for i, item in enumerate(self.structure[:5]):  # Show first 5
                print(f"DEBUG: Element {i+1}: {item.get('type')} {item.get('number', '')} - {item.get('title', '')[:50]}")
        
        return {
            'structure': self.structure,
            'patterns': detection_result['patterns'],
            'hierarchy': detection_result['hierarchy'],
            'grouped_headers': detection_result.get('grouped_headers', [])
        }
    
    def get_structure_summary(self):
        """Get summary of detected structure"""
        result = self.detect_structure() if self.structure is None else {'grouped_headers': []}
        
        summary = {
            'total_elements': len(self.structure) if self.structure else 0,
            'types': {},
            'hierarchy_depth': 0,
            'has_prelims': False
        }
        
        if self.structure:
            for item in self.structure:
                item_type = item.get('type', 'unknown')
                summary['types'][item_type] = summary['types'].get(item_type, 0) + 1
        
        # Check for prelims in grouped headers
        if 'grouped_headers' in result:
            for group in result['grouped_headers']:
                if group.get('type') == 'prelims':
                    summary['has_prelims'] = True
                    break
        
        return summary

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python structure_analyzer.py <markdown_file>")
        sys.exit(1)
    
    analyzer = StructureAnalyzer(sys.argv[1])
    result = analyzer.detect_structure(debug=True)
    summary = analyzer.get_structure_summary()
    
    print(f"\nStructure Analysis for: {sys.argv[1]}")
    print(f"Total elements: {summary['total_elements']}")
    print(f"Element types: {summary['types']}")
    print(f"Detected patterns: {list(result['patterns'].keys())}")
    print(f"Hierarchy: {result['hierarchy']}")
    
    # Show grouped structure
    if 'grouped_headers' in result:
        print(f"\nGrouped Structure ({len(result['grouped_headers'])} groups):")
        for i, group in enumerate(result['grouped_headers']):
            if group.get('type') == 'prelims':
                print(f"  PRELIMS - Content before first structure")
            elif group.get('type') == 'header_group':
                header = group['header']
                items = group['items']
                print(f"  {header['text']} (Level {header['level']}) - {len(items)} items")
                for item in items[:3]:  # Show first 3 items
                    print(f"    - {item['number']}. {item['text'][:50]}...")
                if len(items) > 3:
                    print(f"    ... and {len(items) - 3} more")
            else:
                print(f"  Standalone: {group.get('number', '')} {group.get('text', '')[:50]}...")