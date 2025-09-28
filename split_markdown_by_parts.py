#!/usr/bin/env python3

import sys
import os
import re

def split_markdown_by_parts(input_file, output_dir=None):
    """Split markdown file by parts and save to separate files"""
    
    if not output_dir:
        output_dir = os.path.splitext(input_file)[0] + "_parts"
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by parts using regex
    parts = re.split(r'\n\*\*Part (\d+)\*\*\n', content)
    
    # First part is everything before Part 1
    if parts[0].strip():
        with open(os.path.join(output_dir, "00_preamble.md"), 'w', encoding='utf-8') as f:
            f.write(parts[0].strip())
        print(f"Created: 00_preamble.md")
    
    # Process remaining parts (pairs of part_number and content)
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            part_num = parts[i]
            part_content = parts[i + 1]
            
            # Extract part title from first line
            lines = part_content.strip().split('\n')
            title = lines[0].strip('*').strip() if lines else "Unknown"
            
            # Create filename
            filename = f"part_{part_num.zfill(2)}_{title.lower().replace(' ', '_').replace('—', '_')}.md"
            filename = re.sub(r'[^\w\-_.]', '', filename)
            
            # Write part file
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"**Part {part_num}**\n\n{part_content.strip()}")
            
            print(f"Created: {filename}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python split_markdown_by_parts.py <input_file> [output_dir]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        sys.exit(1)
    
    split_markdown_by_parts(input_file, output_dir)
    print(f"Split completed. Files saved to: {output_dir or os.path.splitext(input_file)[0] + '_parts'}")

if __name__ == "__main__":
    main()