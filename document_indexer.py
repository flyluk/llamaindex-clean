import os
import site
import chromadb
from llama_index.core import VectorStoreIndex, Settings, StorageContext, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from docling.document_converter import DocumentConverter
from structure_detect_n_extract import extract_structure, StructureDetector
import re
import sys
from rag_prompt_template import create_rag_prompt

# Setup TensorRT environment
site_packages = site.getsitepackages()[0]
tensorrt_lib_path = os.path.join(site_packages, 'tensorrt_cu13_libs')
if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['LD_LIBRARY_PATH'] = f"{tensorrt_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
else:
    os.environ['LD_LIBRARY_PATH'] = tensorrt_lib_path

try:
    from llama_index.core.evaluation import SemanticSimilarityEvaluator
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False

class DocumentIndexer:
    def __init__(self, target_dir="", db_path="./chroma_db", model="deepseek-r1:7b", base_url="http://localhost:11434", embed_url=None, api_key=None, embed_model="embeddinggemma", context_length=32768, use_sentence_splitter=False):
        self.target_dir = target_dir
        self.db_path = db_path
        self.model = model
        self.base_url = base_url
        self.embed_url = embed_url or base_url
        self.api_key = api_key
        self.embed_model = embed_model
        self.context_length = context_length
        self.use_sentence_splitter = use_sentence_splitter
        self._setup_settings()
        self._setup_storage()
        
    def _setup_settings(self):
        Settings.node_parser = SentenceSplitter(chunk_size=3000, chunk_overlap=300)
        
        # Auto-detect service types
        base_is_azure = "azure" in self.base_url.lower()
        embed_is_azure = "azure" in self.embed_url.lower()
        embed_is_ollama = not embed_is_azure and ("localhost" in self.embed_url or "11434" in self.embed_url)
        
        # Setup embeddings - use Ollama if embed_url points to local Ollama, even if base_url is Azure
        if embed_is_ollama:
            Settings.embed_model = OllamaEmbedding(
                model_name=self.embed_model,
                base_url=self.embed_url,
                embed_batch_size=1
            )
        elif embed_is_azure:
            from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
            Settings.embed_model = AzureOpenAIEmbedding(
                api_key=self.api_key,
                azure_endpoint=self.embed_url,
                engine=self.embed_model,
                api_version="2024-02-01"
            )
        elif self.api_key:
            from llama_index.embeddings.openai import OpenAIEmbedding
            Settings.embed_model = OpenAIEmbedding(
                api_key=self.api_key,
                api_base=self.embed_url,
                model=self.embed_model
            )
        else:
            Settings.embed_model = OllamaEmbedding(
                model_name=self.embed_model,
                base_url=self.embed_url,
                embed_batch_size=1
            )
        
        # Setup LLM based on base_url
        if base_is_azure:
            from llama_index.llms.azure_openai import AzureOpenAI
            Settings.llm = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.base_url,
                engine=self.model,
                api_version="2024-02-01"
            )
        elif self.api_key and not base_is_azure:
            from llama_index.llms.openai import OpenAI
            Settings.llm = OpenAI(
                api_key=self.api_key,
                model=self.model,
                api_base=self.base_url
            )
        else:
            Settings.llm = Ollama(
                model=self.model,
                request_timeout=240.0,
                context_window=self.context_length,
                base_url=self.base_url
            )
    
    def _setup_storage(self):
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        collection_name = "documents"
        status_collection_name = "status"
        
        try:
            self.doc_collection = self.client.get_or_create_collection(collection_name)
            self.status_collection = self.client.get_or_create_collection(status_collection_name)
        except Exception as e:
            # If there's a dimension mismatch, delete and recreate
            if "dimension" in str(e).lower():
                try:
                    self.client.delete_collection(collection_name)
                    self.client.delete_collection(status_collection_name)
                except:
                    pass
                self.doc_collection = self.client.create_collection(collection_name)
                self.status_collection = self.client.create_collection(status_collection_name)
            else:
                raise e
        
        self.doc_store = ChromaVectorStore(chroma_collection=self.doc_collection)
        self.doc_context = StorageContext.from_defaults(vector_store=self.doc_store)
    

    
    def _get_processed_files(self):
        """Get set of already processed file paths"""
        doc_data = self.doc_collection.get()
        return {metadata.get('file_path') for metadata in doc_data.get('metadatas', []) if metadata.get('file_path')}
    
    def _get_completed_files(self):
        """Get set of files marked as fully completed"""
        try:
            status_data = self.status_collection.get()
            return {metadata.get('file_path') for metadata in status_data.get('metadatas', []) if metadata.get('file_path')}
        except:
            return set()
    
    def _mark_file_complete(self, file_path):
        """Mark individual file as complete"""
        file_id = f"complete_{hash(file_path)}"
        self.status_collection.upsert(
            ids=[file_id],
            documents=[f"File processing completed: {os.path.basename(file_path)}"],
            metadatas=[{"file_path": file_path, "status": "complete"}]
        )
    

    
    def _convert_with_docling(self, file_path):
        converter = DocumentConverter()
        result = converter.convert(file_path)
        return result.document.export_to_markdown()
    
    def _detect_legal_document(self, text):
        """Detect if document is a legal document based on structure patterns"""
        legal_patterns = [
            r'\*\*Schedule\s+\d+[A-Z]*',
            r'\*\*Part\s+\d+',
            r'\*\*Division\s+\d+',
            r'\*\*Subdivision\s+\d+',
            r'\*\*\d+\.\s+[^*]+\*\*'
        ]
        
        total_matches = 0
        for pattern in legal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            total_matches += len(matches)
        
        return total_matches >= 3
    
    def _load_documents(self):
        documents = []
        
        for root, _, files in os.walk(self.target_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Handle markdown files
                if file.endswith('.md'):
                    base_path = os.path.splitext(file_path)[0]
                    # Check if original document exists (pdf, docx, txt)
                    original_exists = any(os.path.exists(f"{base_path}.{ext}") 
                                        for ext in ['pdf', 'docx', 'txt'])
                    if original_exists:
                        continue  # Skip, will be processed via original document
                    else:
                        # Process standalone markdown file
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text = f.read()
                            doc = Document(
                                text=text,
                                metadata={
                                    'file_name': file,
                                    'file_path': file_path
                                }
                            )
                            documents.append(doc)
                        except Exception as e:
                            print(f"Error processing {file}: {e}")
                        continue
                
                # Process non-markdown files
                md_path = os.path.splitext(file_path)[0] + '.md'
                try:
                    # Check if markdown file already exists
                    if os.path.exists(md_path):
                        with open(md_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                    else:
                        text = self._convert_with_docling(file_path)
                        with open(md_path, 'w', encoding='utf-8') as f:
                            f.write(text)
                    
                    doc = Document(
                        text=text,
                        metadata={
                            'file_name': file,
                            'file_path': file_path
                        }
                    )
                    documents.append(doc)
                except Exception as e:
                    print(f"Error processing {file}: {e}")
        
        return documents
    
    def load_or_create_index(self):
        """Load existing index or create empty index structure"""
        num_docs = len(self.doc_collection.get()["ids"])
        print(f"ChromaDB status: {num_docs} documents (DB: {self.db_path})")
        
        if os.path.exists(self.db_path) and num_docs > 0:
            print("Loading existing index")
            self.index = VectorStoreIndex.from_vector_store(self.doc_store, storage_context=self.doc_context)
        else:
            print("Creating new index")
            self.index = VectorStoreIndex([], storage_context=self.doc_context)
        
        print("Index loading/creation completed")
    
    def process_single_file(self, file_path):
        """Process a single file"""
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
        
        print(f"Processing single file: {os.path.basename(file_path)}")
        
        try:
            # Handle markdown files
            if file_path.endswith('.md'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                # Convert non-markdown files
                text = self._convert_with_docling(file_path)
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
            
            # Use sentence splitter if enabled, otherwise use structure extraction
            if self.use_sentence_splitter:
                print(f"📝 Using sentence splitter")
                self.index.insert(doc)
            else:
                # Detect structure and process
                detector = StructureDetector()
                detection_result = detector.detect_structure(doc.text)
                
                is_legal = self._detect_legal_document(doc.text)
                has_structure = (detection_result.get('grouped_headers') and 
                               len(detection_result['grouped_headers']) > 0)
                
                if is_legal:
                    print(f"📋 Legal document detected")
                    structure = extract_structure(doc.text)
                    if structure:
                        print(f"✅ Extracted {len(structure)} legal structural elements")
                        structured_documents = self.create_structured_documents(structure, file_path=file_path)
                        print(f"➡️ Created {len(structured_documents)} structured documents")
                        for struct_doc in structured_documents:
                            self.index.insert(struct_doc)
                elif has_structure:
                    print(f"📄 Structured document detected")
                    structured_documents = self._create_general_structured_documents(detection_result, doc)
                    print(f"✅ Created {len(structured_documents)} general structured documents")
                    for struct_doc in structured_documents:
                        self.index.insert(struct_doc)
                else:
                    print(f"📝 Plain document")
                    self.index.insert(doc)
            
            self._mark_file_complete(file_path)
            print(f"✅ Successfully processed: {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"❌ Failed to process {os.path.basename(file_path)}: {e}")
    
    def process_files(self, target_path=None):
        """Process files from target directory"""
        original_target_dir = self.target_dir
        if target_path:
            self.target_dir = target_path
            
        if not self.target_dir or not os.path.exists(self.target_dir):
            self.target_dir = original_target_dir
            return
            
        documents = self._load_documents()
        self.target_dir = original_target_dir
        if not documents:
            return
                    
        completed_files = self._get_completed_files()
        remaining_docs = [doc for doc in documents if doc.metadata.get('file_path') not in completed_files]
        
        if remaining_docs:
            print(f"Processing: {len(remaining_docs)} files remaining (skipped {len(documents) - len(remaining_docs)} already processed)")
            
            for i, doc in enumerate(remaining_docs, 1):
                print(f"Processing {i}/{len(remaining_docs)}: {doc.metadata.get('file_name', 'Unknown')}")
                
                try:
                    # Use sentence splitter if enabled, otherwise use structure extraction
                    if self.use_sentence_splitter:
                        print(f"📝 Using sentence splitter: {doc.metadata.get('file_name')}")
                        self.index.insert(doc)
                    else:
                        # Detect document structure for all documents
                        detector = StructureDetector()
                        detection_result = detector.detect_structure(doc.text)
                        
                        # Check if document has any structure (legal or general)
                        is_legal = self._detect_legal_document(doc.text)
                        has_structure = (detection_result.get('grouped_headers') and 
                                       len(detection_result['grouped_headers']) > 0)
                        
                        if is_legal:
                            print(f"📋 Legal document detected: {doc.metadata.get('file_name')}")
                            structure = extract_structure(doc.text)
                            if structure:
                                print(f"✅ Extracted {len(structure)} legal structural elements")
                                structured_documents = self.create_structured_documents(structure, file_path=doc.metadata.get('file_path'))
                                print(f"➡️ Created {len(structured_documents)} structured documents")
                                for struct_doc in structured_documents:
                                    self.index.insert(struct_doc)
                        elif has_structure:
                            print(f"📄 Structured document detected: {doc.metadata.get('file_name')}")
                            structured_documents = self._create_general_structured_documents(detection_result, doc)
                            print(f"✅ Created {len(structured_documents)} general structured documents")
                            for struct_doc in structured_documents:
                                self.index.insert(struct_doc)
                        else:
                            print(f"📝 Plain document: {doc.metadata.get('file_name')}")
                            self.index.insert(doc)
                        
                    self._mark_file_complete(doc.metadata.get('file_path'))
                except Exception as e:
                    print(f"Failed to process {doc.metadata.get('file_name', 'Unknown')}: {e}")
                    continue
    

    
    def sync_new_files(self):
        """Sync new files that aren't in the index yet"""
        self._sync_missing_files()
    
    def init_and_process_files(self):
        """Initialize index and process files if target_dir is set"""
        self.load_or_create_index()
        if self.target_dir:
            self.process_files()
    
    def _sync_missing_files(self):
        if not os.path.exists(self.target_dir):
            return
            
        stored_files = {m.get("file_path", "") for m in self.doc_collection.get()["metadatas"]}
        current_files = set()
        
        for root, _, files in os.walk(self.target_dir):
            for file in files:
                current_files.add(os.path.join(root, file))
        
        missing_files = current_files - stored_files
        if missing_files:
            print(f"Indexing {len(missing_files)} new files")
            new_documents = []
            for file_path in missing_files:
                try:
                    text = self._convert_with_docling(file_path)
                    doc = Document(
                        text=text,
                        metadata={
                            'file_name': os.path.basename(file_path),
                            'file_path': file_path
                        }
                    )
                    new_documents.append(doc)
                except Exception as e:
                    print(f"Error processing {os.path.basename(file_path)}: {e}")
            
            if new_documents:
                # Process each new document individually
                for i, doc in enumerate(new_documents, 1):
                    print(f"Processing new document {i}/{len(new_documents)}: {doc.metadata.get('file_name', 'Unknown')}")
                    
                    # Detect structure for new documents too
                    detector = StructureDetector()
                    detection_result = detector.detect_structure(doc.text)
                    
                    is_legal = self._detect_legal_document(doc.text)
                    has_structure = (detection_result.get('grouped_headers') and 
                                   len(detection_result['grouped_headers']) > 0)
                    
                    if is_legal:
                        structure = extract_structure(doc.text)
                        if structure:
                            structured_documents = self.create_structured_documents(structure, file_path=doc.metadata.get('file_path'))
                            for struct_doc in structured_documents:
                                self.index.insert(struct_doc)
                        else:
                            self.index.insert(doc)
                    elif has_structure:
                        structured_documents = self._create_general_structured_documents(detection_result, doc)
                        for struct_doc in structured_documents:
                            self.index.insert(struct_doc)
                    else:
                        self.index.insert(doc)
                
                print(f"Processed {len(new_documents)} new documents")
    
    def section_search(self, section_num):
        """Direct search for specific section number"""
        data = self.doc_collection.get(include=["metadatas", "documents"])
        results = []
        
        print(f"Direct section search for: Section {section_num}")
        print(f"Searching through {len(data['documents'])} documents")
        
        # Try multiple search patterns
        patterns = [
            f"Section {section_num}:",
            f"Section {section_num} ",
            f"Section {section_num}\n",
            f"**Section {section_num}:",
            f"# Section {section_num}"
        ]
        
        for i, doc_text in enumerate(data["documents"]):
            found = False
            for pattern in patterns:
                if pattern in doc_text:
                    results.append({
                        'score': 1.0,  # Perfect match
                        'text': doc_text,
                        'metadata': data["metadatas"][i]
                    })
                    print(f"Found Section {section_num} in document {i} with pattern: '{pattern}'")
                    found = True
                    break
            
            # Partial match fallback - check for individual words
            if not found:
                doc_lower = doc_text.lower()
                section_count = doc_lower.count('section')
                num_count = doc_lower.count(section_num.lower())
                
                if section_count > 0 and num_count > 0:
                    score = (section_count + num_count) / len(doc_text.split())
                    results.append({
                        'score': score,
                        'text': doc_text,
                        'metadata': data["metadatas"][i]
                    })
                    print(f"Partial match in doc {i}: 'section'({section_count}) + '{section_num}'({num_count}), score: {score:.6f}")
        
        print(f"Section search found {len(results)} matches")
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def keyword_search(self, query, top_k=3):
        data = self.doc_collection.get(include=["metadatas", "documents"])
        results = []
        query_lower = query.lower()
        query_words = query_lower.split()
        
        print(f"Keyword search for: '{query}'")
        print(f"Total documents to search: {len(data['documents'])}")
        
        # First try whole sentence search
        for i, doc_text in enumerate(data["documents"]):
            doc_lower = doc_text.lower()
            sentence_matches = doc_lower.count(query_lower)
            
            if sentence_matches > 0:
                score = sentence_matches * 10 / len(doc_text.split())  # Higher weight for sentence matches
                print(f"Sentence match in doc {i}: {sentence_matches} occurrences, score: {score:.6f}")
                results.append({
                    'score': score,
                    'text': doc_text,
                    'type': data["metadatas"][i].get('type', 'Unknown'),
                    'number': data["metadatas"][i].get('number', '')
                })
        
        # If no sentence matches, fallback to word search
        if not results:
            print("No sentence matches found, trying word search...")
            for i, doc_text in enumerate(data["documents"]):
                doc_lower = doc_text.lower()
                total_matches = 0
                
                # Count matches for each word in query
                for word in query_words:
                    total_matches += doc_lower.count(word)
                
                if total_matches > 0:
                    score = total_matches / len(doc_text.split())
                    print(f"Word match in doc {i}: {total_matches} occurrences, score: {score:.6f}")
                    results.append({
                        'score': score,
                        'text': doc_text,
                        'type': data["metadatas"][i].get('type', 'Unknown'),
                        'number': data["metadatas"][i].get('number', '')
                    })
        
        print(f"Found {len(results)} matching documents")
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
    

    

    
    def delete_file(self, file_path):
        """Delete all data for a specific file from all collections"""
        deleted_count = 0
        
        # Delete from documents collection
        doc_data = self.doc_collection.get(include=["metadatas"])
        doc_ids_to_delete = [doc_data["ids"][i] for i, metadata in enumerate(doc_data["metadatas"]) 
                            if metadata.get('file_path') == file_path]
        if doc_ids_to_delete:
            self.doc_collection.delete(ids=doc_ids_to_delete)
            deleted_count += len(doc_ids_to_delete)
        
        # Delete from status collection
        status_data = self.status_collection.get(include=["metadatas"])
        status_ids_to_delete = [status_data["ids"][i] for i, metadata in enumerate(status_data["metadatas"]) 
                               if metadata.get('file_path') == file_path]
        if status_ids_to_delete:
            self.status_collection.delete(ids=status_ids_to_delete)
            deleted_count += len(status_ids_to_delete)
        
        print(f"Deleted {deleted_count} records for file: {file_path}")
        return deleted_count > 0

    def create_structured_documents(self, structure, parent_path="", parent_content="", file_path=""):
        """Create documents with hierarchical structure and metadata"""
        documents = []
        
        for item in structure:
            # Handle both old and new structure formats
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
            if item_number:
                title = f"{item_type.title()} {item_number}"
            else:
                title = item_type.title()
            
            if item_title:
                title += f": {item_title}"
            
            # Use content if available
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
                        'has_subsections': bool(item.get('children') or item.get('parts') or item.get('divisions') or item.get('subdivisions') or item.get('sections')),
                        'file_path': file_path,
                        'file_name': os.path.basename(file_path) if file_path else ''
                    }
                )
                documents.append(doc)
            
            # Process nested elements (handle both old and new formats)
            nested_items = []
            
            # New format: children array
            if item.get('children'):
                nested_items.extend(item['children'])
            
            # Old format: specific keys
            for nested_key in ['parts', 'divisions', 'subdivisions', 'sections']:
                if item.get(nested_key):
                    nested_items.extend(item[nested_key])
            
            if nested_items:
                nested_docs = self.create_structured_documents(
                    nested_items, 
                    current_path, 
                    full_content,
                    file_path
                )
                documents.extend(nested_docs)
        
        return documents
    
    def _create_general_structured_documents(self, detection_result, original_doc):
        """Create structured documents from general document structure detection"""
        documents = []
        grouped_headers = detection_result.get('grouped_headers', [])
        text = original_doc.text
        file_path = original_doc.metadata.get('file_path', '')
        
        for group in grouped_headers:
            if group.get('type') == 'prelims':
                # Extract prelims content
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
                
                # Find content between header and first numbered item
                if items:
                    first_item_pos = items[0]['position']
                    header_content = text[header_start:first_item_pos].strip()
                else:
                    # Find next header or end of document
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
                
                # Create documents for numbered items under this header
                for i, item in enumerate(items):
                    item_start = item['position']
                    
                    # Find content for this numbered item
                    if i + 1 < len(items):
                        item_end = items[i + 1]['position']
                    else:
                        # Find next header or end of document
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

    def get_dimension_info(self, collection_name=None):
        """Get dimension information from ChromaDB collections"""
        try:
            collections = self.client.list_collections()
            
            if collection_name:
                collections = [c for c in collections if c.name == collection_name]
                if not collections:
                    return {'error': f"Collection '{collection_name}' not found"}
            
            result = {
                'db_path': os.path.abspath(self.db_path),
                'total_collections': len(collections),
                'collections': [],
                'config': {
                    'embed_model': self.embed_model,
                    'embed_url': self.embed_url
                }
            }
            
            for collection in collections:
                data = collection.get(limit=1, include=["embeddings", "metadatas"])
                
                collection_info = {
                    'name': collection.name,
                    'total_documents': collection.count()
                }
                
                if (data['embeddings'] is not None and 
                    len(data['embeddings']) > 0 and 
                    data['embeddings'][0] is not None):
                    collection_info['dimension'] = len(data['embeddings'][0])
                else:
                    collection_info['dimension'] = 'No embeddings'
                
                if data['metadatas'] and len(data['metadatas']) > 0:
                    metadata = data['metadatas'][0]
                    if 'embed_model' in metadata:
                        collection_info['stored_embed_model'] = metadata['embed_model']
                
                result['collections'].append(collection_info)
            
            return result
            
        except Exception as e:
            return {'error': str(e)}

    def display_status(self):
        """Get complete document status as structured data"""
        try:
            doc_data = self.doc_collection.get(include=["metadatas"])
            status_data = self.status_collection.get(include=["metadatas"])
            
            processed_files = self._get_processed_files()
            completed_files = self._get_completed_files()
            
            all_files = {}
            for metadata in doc_data['metadatas']:
                file_path = metadata.get('file_path')
                if file_path:
                    all_files[file_path] = {
                        'name': metadata.get('file_name', 'Unknown'),
                        'path': file_path,
                        'processed': True,
                        'completed': file_path in completed_files,
                        'metadata': metadata
                    }
            
            return {
                'db_path': os.path.abspath(self.db_path),
                'model': self.model,
                'collections': {
                    'documents': len(doc_data['ids']),
                    'status': len(status_data['ids'])
                },
                'processing_status': {
                    'total_processed_files': len(processed_files),
                    'completed_files': len(completed_files)
                },
                'files': list(all_files.values())
            }
            
        except Exception as e:
            return {'error': str(e)}
    

    

    
    def tune_parameters(self, eval_questions=None, eval_answers=None):
        """Simple parameter tuning for optimal performance"""
        if not EVALUATION_AVAILABLE:
            return {
                'error': 'Evaluation not available',
                'default_params': {'similarity_top_k': 5}
            }
        
        if not eval_questions or not eval_answers:
            # Use default test questions if none provided
            eval_questions = ["What is Section 151?", "What are the requirements?", "What is the procedure?"]
            eval_answers = ["Section 151 content", "Requirements content", "Procedure content"]
        
        print("Tuning similarity_top_k parameter...")
        evaluator = SemanticSimilarityEvaluator()
        best_score = 0
        best_k = 5
        
        for k in [3, 5, 10, 15, 20]:
            print(f"Testing similarity_top_k={k}")
            query_engine = self.index.as_query_engine(similarity_top_k=k)
            total_score = 0
            
            for question, expected_answer in zip(eval_questions, eval_answers):
                try:
                    response = query_engine.query(question)
                    eval_result = evaluator.evaluate(response=str(response), reference=expected_answer)
                    total_score += eval_result.score
                except Exception as e:
                    print(f"Error evaluating with k={k}: {e}")
                    continue
            
            avg_score = total_score / len(eval_questions)
            print(f"Average score for k={k}: {avg_score:.3f}")
            
            if avg_score > best_score:
                best_score = avg_score
                best_k = k
        
        best_params = {'similarity_top_k': best_k}
        print(f"Best parameters: {best_params} (score: {best_score:.3f})")
        return best_params
    
    def query(self, query_text, use_direct=False, similarity_top_k=10, chat_history=None):
        print(f"Query: {query_text}")
        print(f"Mode: {'Hybrid search' if not use_direct else 'Direct vector search'}")
        
        # Use hybrid search by default, direct vector only when specified
        if use_direct:
            retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
        else:
            from llama_index.retrievers.bm25 import BM25Retriever
            from llama_index.core.retrievers import QueryFusionRetriever
            
            # Get all nodes for BM25
            all_nodes = []
            doc_data = self.doc_collection.get(include=["documents", "metadatas"])
            for i, doc_text in enumerate(doc_data["documents"]):
                from llama_index.core.schema import TextNode
                node = TextNode(text=doc_text, metadata=doc_data["metadatas"][i])
                all_nodes.append(node)
            
            # Create retrievers
            vector_retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
            bm25_retriever = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=similarity_top_k)
            
            # Custom query generation prompt
            QUERY_GEN_PROMPT = (
                "You are a helpful assistant that generates multiple search queries based on a "
                "single input query. Generate {num_queries} search queries, one on each line, "
                "related to the following input query:\n"
                "Query: {query}\n"
                "Queries:\n"
            )
            
            # Combine with QueryFusionRetriever
            retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                similarity_top_k=similarity_top_k,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False,
                verbose=True
            )
        
        nodes = retriever.retrieve(query_text)
        
        if not nodes:
            print("No relevant documents found")
            return "No relevant information found for your query."
        
        print(f"\nRetrieved {len(nodes)} relevant documents:")
        for i, node in enumerate(nodes[:5], 1):  # Show top 5
            doc_name = node.metadata.get('file_name', 'N/A')
            hierarchy = node.metadata.get('hierarchy_path', 'N/A')
            print(f"{i}. Score: {node.score:.3f} | {hierarchy} | {doc_name}")
            print(f"   Preview: {node.text[:100]}...")
        
        # Combine all retrieved context with source IDs
        context_parts = []
        for i, node in enumerate(nodes, 1):
            hierarchy = node.metadata.get('hierarchy_path', '')
            file_name = node.metadata.get('file_name', f'source_{i}')
            
            if hierarchy:
                context_parts.append(f"<source_id>{file_name}</source_id>\n[{hierarchy}]\n{node.text}")
            else:
                context_parts.append(f"<source_id>{file_name}</source_id>\n{node.text}")
        
        combined_context = "\n\n---\n\n".join(context_parts)
        
        # Use RAG prompt template with chat history
        context_prompt = create_rag_prompt(combined_context, query_text, chat_history)
        
        # Debug print with context placeholder
        debug_prompt = context_prompt.replace(combined_context, "{context}")
        print(f"\n=== PROMPT TO LLM ===\n{debug_prompt}\n=== END PROMPT ===\n")
        
        context_length = len(context_prompt)
        input_tokens = context_length // 4
        input_cost = (input_tokens / 1000) * 0.00135
        
        print(f"\nGenerating response using {len(nodes)} context documents...")
        print(f"Context length: {context_length:,} characters (~{input_tokens:,} tokens)")
        print(f"Estimated input cost: ${input_cost:.6f} USD")
        
        response = Settings.llm.complete(context_prompt)
        
        output_length = len(str(response))
        output_tokens = output_length // 4
        output_cost = (output_tokens / 1000) * 0.0054
        total_cost = input_cost + output_cost
        
        print(f"\nResponse length: {output_length:,} characters (~{output_tokens:,} tokens)")
        print(f"Estimated output cost: ${output_cost:.6f} USD")
        print(f"Total estimated cost: ${total_cost:.6f} USD")
        print(f"\nContext-based response:\n{response}")
        return response