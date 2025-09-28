from docling.document_converter import DocumentConverter
import os
import sys

class DocConverter:
    def __init__(self):
        self.converter = DocumentConverter()
    
    def convert_to_markdown(self, input_path, output_path=None):
        """Convert document to markdown format"""
        result = self.converter.convert(input_path)
        markdown_content = result.document.export_to_markdown()
        
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = f"{base_name}.md"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return output_path, markdown_content
    
    def convert_to_doctag(self, input_path, output_path=None):
        """Convert document to doctag format"""
        result = self.converter.convert(input_path)
        doctag_content = result.document.export_to_doctags()
        
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = f"{base_name}.doctag"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doctag_content)
        
        return output_path, doctag_content

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python doc_converter.py <format> <input_file> [output_file]")
        print("Formats: markdown, doctag")
        sys.exit(1)
    
    format_type = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    converter = DocConverter()
    
    if format_type == "markdown":
        output_path, _ = converter.convert_to_markdown(input_file, output_file)
    elif format_type == "doctag":
        output_path, _ = converter.convert_to_doctag(input_file, output_file)
    else:
        print("Invalid format. Use 'markdown' or 'doctag'")
        sys.exit(1)
    
    print(f"Converted to: {output_path}")