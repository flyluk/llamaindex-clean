#!/usr/bin/env python3

import sys
from structure_detect_n_extract import extract_structure

def format_structure(structure, indent=0):
    """Format structure as indented text"""
    result = []
    for item in structure:
        prefix = "  " * indent
        
        if item['type'] == 'schedule':
            title = f": {item['title']}" if item['title'] else ""
            result.append(f"{prefix}Schedule {item['number']}{title}")
            if item.get('content'):
                content_lines = item['content'].strip().split('\n')
                for line in content_lines:
                    if line.strip():
                        result.append(f"{prefix}  {line.strip()}")
            if item['parts']:
                result.extend(format_structure(item['parts'], indent + 1))
            if item['sections']:
                result.extend(format_structure(item['sections'], indent + 1))
                
        elif item['type'] == 'part':
            result.append(f"{prefix}Part {item['number']}: {item['title']}")
            if item.get('content'):
                content_lines = item['content'].strip().split('\n')
                for line in content_lines:
                    if line.strip():
                        result.append(f"{prefix}  {line.strip()}")
            if item['divisions']:
                result.extend(format_structure(item['divisions'], indent + 1))
            if 'sections' in item:
                result.extend(format_structure(item['sections'], indent + 1))
                
        elif item['type'] == 'division':
            title = f": {item['title']}" if item['title'] else ""
            result.append(f"{prefix}Division {item['number']}{title}")
            if item.get('content'):
                content_lines = item['content'].strip().split('\n')
                for line in content_lines:
                    if line.strip():
                        result.append(f"{prefix}  {line.strip()}")
            if item['subdivisions']:
                result.extend(format_structure(item['subdivisions'], indent + 1))
            if item['sections']:
                result.extend(format_structure(item['sections'], indent + 1))
                
        elif item['type'] == 'subdivision':
            title = f": {item['title']}" if item['title'] else ""
            result.append(f"{prefix}Subdivision {item['number']}{title}")
            if item.get('content'):
                content_lines = item['content'].strip().split('\n')
                for line in content_lines:
                    if line.strip():
                        result.append(f"{prefix}  {line.strip()}")
            if item['sections']:
                result.extend(format_structure(item['sections'], indent + 1))
                
        elif item['type'] == 'section':
            result.append(f"{prefix}Section {item['number']}: {item['title']}")
            if item.get('content'):
                content_lines = item['content'].strip().split('\n')
                for line in content_lines:
                    if line.strip():
                        result.append(f"{prefix}  {line.strip()}")
    
    return result

def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_to_text.py <input_markdown_file> <output_text_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        structure = extract_structure(markdown_content)
        formatted_lines = format_structure(structure)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(formatted_lines))
        
        print(f"Structure extracted to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()