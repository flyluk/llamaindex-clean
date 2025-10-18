"""Document processing and structure detection."""

import os
import re
from llama_index.core import Document
from docling.document_converter import DocumentConverter
from structure_detect_n_extract import extract_structure, StructureDetector


class DocumentProcessor:
    """Handles document conversion and structure detection."""
    
    @staticmethod
    def convert_with_docling(file_path):
        """Convert document to markdown using Docling."""
        converter = DocumentConverter()
        result = converter.convert(file_path)
        return result.document.export_to_markdown()
    
    @staticmethod
    def detect_legal_document(text):
        """Detect if document is a legal document based on structure patterns."""
        legal_patterns = [
            r'\*\*Schedule\s+\d+[A-Z]*',
            r'\*\*Part\s+\d+',
            r'\*\*Division\s+\d+',
            r'\*\*Subdivision\s+\d+',
            r'\*\*\d+\.\s+[^*]+\*\*'
        ]
        
        total_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                          for pattern in legal_patterns)
        return total_matches >= 3
    
    @staticmethod
    def process_document(file_path, use_sentence_splitter=False):
        """Process a single document and return Document object with structure info."""
        # Handle markdown files
        if file_path.endswith('.md'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            # Convert non-markdown files
            text = DocumentProcessor.convert_with_docling(file_path)
            # Save markdown version
            md_path = os.path.splitext(file_path)[0] + '.md'
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(text)
        
        doc = Document(
            text=text,
            metadata={
                'file_name': os.path.basename(file_path),
                'file_path': file_path
            }
        )
        
        # Detect structure if not using sentence splitter
        if not use_sentence_splitter:
            detector = StructureDetector()
            detection_result = detector.detect_structure(text)
            is_legal = DocumentProcessor.detect_legal_document(text)
            has_structure = (detection_result.get('grouped_headers') and 
                           len(detection_result['grouped_headers']) > 0)
            
            return doc, {
                'is_legal': is_legal,
                'has_structure': has_structure,
                'detection_result': detection_result
            }
        
        return doc, None
    
    @staticmethod
    def create_structured_documents(structure, parent_path="", parent_content="", file_path=""):
        """Create documents with hierarchical structure and metadata."""
        documents = []
        
        for item in structure:
            item_type = item.get('type', 'section')
            item_number = item.get('number', '')
            item_title = item.get('title', '')
            item_content = item.get('content', '')
            
            # Build hierarchical path
            if item_number:
                current_path = f"{parent_path}/{item_type.title()} {item_number}" if parent_path else f"{item_type.title()} {item_number}"
            else:
                current_path = f"{parent_path}/{item_type.title()}" if parent_path else item_type.title()
            
            # Create title with hierarchy
            title = f"{item_type.title()} {item_number}" if item_number else item_type.title()
            if item_title:
                title += f": {item_title}"
            
            full_content = item_content if item_content else ""
            
            # Create document for this structural element
            if full_content.strip() or item_title:
                doc = Document(
                    text=f"{title}\n\n{full_content}" if full_content else title,
                    metadata={
                        'type': item_type,
                        'number': item_number,
                        'title': item_title,
                        'hierarchy_path': current_path,
                        'level': len(current_path.split('/')) if current_path else 1,
                        'parent_path': parent_path,
                        'has_subsections': bool(item.get('children') or item.get('parts') or 
                                              item.get('divisions') or item.get('subdivisions') or 
                                              item.get('sections')),
                        'file_path': file_path,
                        'file_name': os.path.basename(file_path) if file_path else ''
                    }
                )
                documents.append(doc)
            
            # Process nested elements
            nested_items = []
            if item.get('children'):
                nested_items.extend(item['children'])
            for nested_key in ['parts', 'divisions', 'subdivisions', 'sections']:
                if item.get(nested_key):
                    nested_items.extend(item[nested_key])
            
            if nested_items:
                nested_docs = DocumentProcessor.create_structured_documents(
                    nested_items, current_path, full_content, file_path
                )
                documents.extend(nested_docs)
        
        return documents
    
    @staticmethod
    def create_general_structured_documents(detection_result, original_doc):
        """Create structured documents from general document structure detection."""
        documents = []
        grouped_headers = detection_result.get('grouped_headers', [])
        text = original_doc.text
        file_path = original_doc.metadata.get('file_path', '')
        
        for group in grouped_headers:
            if group.get('type') == 'prelims':
                prelims_content = text[:group['end_position']].strip()
                if prelims_content:
                    doc = Document(
                        text=prelims_content,
                        metadata={
                            'type': 'prelims',
                            'title': 'Preliminary Content',
                            'hierarchy_path': 'Prelims',
                            'level': 1,
                            'file_path': file_path,
                            'file_name': original_doc.metadata.get('file_name', '')
                        }
                    )
                    documents.append(doc)
            
            elif group.get('type') == 'header_group':
                header = group['header']
                items = group.get('items', [])
                
                # Create document for header section
                header_start = header['position']
                
                if items:
                    first_item_pos = items[0]['position']
                    header_content = text[header_start:first_item_pos].strip()
                else:
                    next_pos = len(text)
                    for other_group in grouped_headers:
                        if (other_group.get('type') == 'header_group' and 
                            other_group['header']['position'] > header_start):
                            next_pos = min(next_pos, other_group['header']['position'])
                    header_content = text[header_start:next_pos].strip()
                
                if header_content:
                    doc = Document(
                        text=header_content,
                        metadata={
                            'type': 'header',
                            'title': header['text'],
                            'hierarchy_path': f"H{header['level']}: {header['text']}",
                            'level': header['level'],
                            'file_path': file_path,
                            'file_name': original_doc.metadata.get('file_name', '')
                        }
                    )
                    documents.append(doc)
                
                # Create documents for numbered items
                for i, item in enumerate(items):
                    item_start = item['position']
                    
                    if i + 1 < len(items):
                        item_end = items[i + 1]['position']
                    else:
                        item_end = len(text)
                        for other_group in grouped_headers:
                            if (other_group.get('type') == 'header_group' and 
                                other_group['header']['position'] > item_start):
                                item_end = min(item_end, other_group['header']['position'])
                    
                    item_content = text[item_start:item_end].strip()
                    
                    if item_content:
                        doc = Document(
                            text=item_content,
                            metadata={
                                'type': 'numbered_item',
                                'number': item['number'],
                                'title': item['text'],
                                'hierarchy_path': f"H{header['level']}: {header['text']} / {item['number']}. {item['text']}",
                                'level': header['level'] + 1,
                                'parent_header': header['text'],
                                'file_path': file_path,
                                'file_name': original_doc.metadata.get('file_name', '')
                            }
                        )
                        documents.append(doc)
        
        return documents
