import os
import chromadb
from llama_index.core import VectorStoreIndex, Settings, StorageContext, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from docling.document_converter import DocumentConverter

try:
    from llama_index.core.evaluation import SemanticSimilarityEvaluator
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False

class DocumentIndexer:
    def __init__(self, target_dir="/mnt/c/users/flyluk/test", db_path="./chroma_db", model="gpt-oss:20b"):
        self.target_dir = target_dir
        self.db_path = db_path
        self.model = model
        self._setup_settings()
        self._setup_storage()
        
    def _setup_settings(self):
        Settings.embed_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://192.168.1.16:11434",
            embed_batch_size=1
        )
        Settings.node_parser = SentenceSplitter(chunk_size=3000, chunk_overlap=400)
        Settings.llm = Ollama(model=self.model, request_timeout=240.0, context_window=8192)
    
    def _setup_storage(self):
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.doc_collection = self.client.get_or_create_collection("documents")
        self.summary_collection = self.client.get_or_create_collection("summaries")
        self.status_collection = self.client.get_or_create_collection("status")
        
        self.doc_store = ChromaVectorStore(chroma_collection=self.doc_collection)
        self.summary_store = ChromaVectorStore(chroma_collection=self.summary_collection)
        
        self.doc_context = StorageContext.from_defaults(vector_store=self.doc_store)
        self.summary_context = StorageContext.from_defaults(vector_store=self.summary_store)
    
    def _create_summary_document(self, file_path, summary_text, chunk_count=0, original_size=0):
        """Create standardized summary document with consistent metadata"""
        return Document(
            text=f"SUMMARY: {summary_text}\nFILE: {os.path.basename(file_path)}",
            metadata={
                'file_name': os.path.basename(file_path),
                'file_path': file_path,
                'summary': "refer to text",
                'chunk_count': chunk_count,
                'original_size': original_size
            }
        )
    
    def _create_summaries(self, documents):
        summary_nodes = []
        for doc in documents:
            doc_size = len(doc.text)
            chunk_size, overlap = self._get_smart_chunk_params(doc_size)
            file_path = doc.metadata.get('file_path', '')
            
            # Use smart chunking for large documents
            if doc_size > chunk_size:
                splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
                nodes = splitter.get_nodes_from_documents([doc])
                
                # Create summary for each chunk
                summaries = []
                for i, node in enumerate(nodes):
                    summary = Settings.llm.complete(f"Summarize in 1-2 sentences:\n\n{node.text}")
                    summaries.append(str(summary))
                    
                    # Save individual chunk summary
                    chunk_summary_doc = Document(
                        text=f"CHUNK SUMMARY: {summary}\nFILE: {os.path.basename(file_path)}",
                        metadata={
                            'file_name': os.path.basename(file_path),
                            'file_path': file_path,
                            'summary': "refer to text",
                            'summary_type': 'chunk',
                            'chunk_index': i,
                            'chunk_count': len(nodes),
                            'original_size': doc_size
                        }
                    )
                    summary_nodes.append(chunk_summary_doc)
                
                # Create overall summary
                overall_summary = Settings.llm.complete(
                    f"Create comprehensive summary from:\n\n{chr(10).join(summaries)}"
                )
                chunk_count = len(nodes)
            else:
                # Simple summary for smaller documents
                overall_summary = Settings.llm.complete(
                    f"Create comprehensive summary from:\n\n{doc.text}."
                )
                chunk_count = 1
            
            # Save overall summary
            overall_summary_doc = Document(
                text=f"OVERALL SUMMARY: {overall_summary}\nFILE: {os.path.basename(file_path)}",
                metadata={
                    'file_name': os.path.basename(file_path),
                    'file_path': file_path,
                    'summary': "refer to text",
                    'summary_type': 'overall',
                    'chunk_count': chunk_count,
                    'original_size': doc_size
                }
            )
            summary_nodes.append(overall_summary_doc)
        return summary_nodes
    
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
        num_summaries = len(self.summary_collection.get()["ids"])
        print(f"ChromaDB status: {num_docs} documents, {num_summaries} summaries")
        
        if os.path.exists(self.db_path) and num_docs > 0:
            print("Loading existing index")
            self.index = VectorStoreIndex.from_vector_store(self.doc_store, storage_context=self.doc_context)
            self.summary_index = VectorStoreIndex.from_vector_store(self.summary_store, storage_context=self.summary_context)
        else:
            print("Creating new index")
            self.index = VectorStoreIndex([], storage_context=self.doc_context)
            self.summary_index = VectorStoreIndex([], storage_context=self.summary_context)
        
        print("Index loading/creation completed")
    
    def process_files(self):
        """Process files from target directory"""
        if not self.target_dir or not os.path.exists(self.target_dir):
            return
            
        documents = self._load_documents()
        if not documents:
            return
            
        processed_files = self._get_processed_files()
        completed_files = self._get_completed_files()
        remaining_docs = [doc for doc in documents if doc.metadata.get('file_path') not in completed_files]
        
        if remaining_docs:
            print(f"Processing: {len(remaining_docs)} files remaining (skipped {len(documents) - len(remaining_docs)} already processed)")
            
            for i, doc in enumerate(remaining_docs, 1):
                print(f"Processing {i}/{len(remaining_docs)}: {doc.metadata.get('file_name', 'Unknown')}")
                
                try:
                    self.index.insert(doc)
                    summary_docs = self._create_summaries([doc])
                    for summary_doc in summary_docs:
                        self.summary_index.insert(summary_doc)
                    self._mark_file_complete(doc.metadata.get('file_path'))
                except Exception as e:
                    print(f"Failed to process {doc.metadata.get('file_name', 'Unknown')}: {e}")
                    continue
        
        self._generate_missing_summaries()
    
    def _generate_missing_summaries(self):
        """Generate summaries for documents that don't have them"""
        doc_files = self._get_processed_files()
        summary_data = self.summary_collection.get()
        summary_files = {metadata.get('file_path') for metadata in summary_data.get('metadatas', []) if metadata.get('file_path')}
        missing_summary_files = doc_files - summary_files
        
        if not missing_summary_files:
            return
            
        print(f"Missing summaries for {len(missing_summary_files)} files. Generating...")
        doc_data = self.doc_collection.get(include=['metadatas', 'documents'])
        
        # Group documents by file_path
        files_docs = {}
        for i, metadata in enumerate(doc_data['metadatas']):
            file_path = metadata.get('file_path')
            if file_path in missing_summary_files:
                if file_path not in files_docs:
                    files_docs[file_path] = {'texts': [], 'metadata': metadata}
                files_docs[file_path]['texts'].append(doc_data['documents'][i])
        
        # Create summaries for grouped documents
        for file_path, file_data in files_docs.items():
            combined_text = '\n\n'.join(file_data['texts'])
            doc = Document(text=combined_text, metadata=file_data['metadata'])
            
            try:
                summary_docs = self._create_summaries([doc])
                for summary_doc in summary_docs:
                    self.summary_index.insert(summary_doc)
                self._mark_file_complete(file_path)
                print(f"Generated summary for {file_data['metadata'].get('file_name', 'Unknown')}")
            except Exception as e:
                print(f"Failed to generate summary for {file_data['metadata'].get('file_name', 'Unknown')}: {e}")
    
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
                    
                    # Add to main index
                    self.index.insert(doc)
                    
                    # Create and add summary immediately
                    summary_docs = self._create_summaries([doc])
                    for summary_doc in summary_docs:
                        self.summary_index.insert(summary_doc)
                
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
        query_words = query.lower().split()
        
        print(f"Keyword search for: '{query}'")
        print(f"Total documents to search: {len(data['documents'])}")
        
        for i, doc_text in enumerate(data["documents"]):
            doc_lower = doc_text.lower()
            total_matches = 0
            
            # Count matches for each word in query
            for word in query_words:
                total_matches += doc_lower.count(word)
            
            if total_matches > 0:
                score = total_matches / len(doc_text.split())
                print(f"Match found in doc {i}: {total_matches} occurrences, score: {score:.6f}")
                results.append({
                    'score': score,
                    'text': doc_text,
                    'type': data["metadatas"][i].get('type', 'Unknown'),
                    'number': data["metadatas"][i].get('number', '')
                })
        
        print(f"Found {len(results)} matching documents")
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
    
    def summary_search(self, query, top_k=3):
        if not hasattr(self, 'summary_index'):
            return []
            
        retriever = self.summary_index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        
        return [{
            'file_name': node.metadata.get('file_name', 'Unknown'),
            'summary': node.metadata.get('summary', ''),
            'score': node.score
        } for node in nodes]
    
    def _get_smart_chunk_params(self, doc_size):
        """Determine optimal chunking parameters based on document size"""
        if doc_size > 100000:
            return 2000, 300
        elif doc_size > 50000:
            return 1500, 200
        else:
            return 1000, 150
    
    def _is_already_processed(self, file_path):
        """Check if document is already processed"""
        try:
            existing_data = self.summary_collection.get(include=["metadatas"])
            for metadata in existing_data["metadatas"]:
                if metadata.get("file_path") == file_path:
                    return True
        except:
            pass
        return False
    
    def process_large_document(self, file_path):
        """Process large document with smart chunking"""
        if self._is_already_processed(file_path):
            print(f"Document already processed: {os.path.basename(file_path)}")
            return None
        
        text = self._convert_with_docling(file_path)
        doc = Document(
            text=text,
            metadata={
                'file_name': os.path.basename(file_path),
                'file_path': file_path
            }
        )
        
        doc_size = len(doc.text)
        print(f"Processing: {os.path.basename(file_path)} ({doc_size:,} chars)")
        
        chunk_size, overlap = self._get_smart_chunk_params(doc_size)
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        nodes = splitter.get_nodes_from_documents([doc])
        
        print(f"Created {len(nodes)} chunks (size: {chunk_size}, overlap: {overlap})")
        
        summary_nodes = nodes[:min(10, len(nodes))]
        summaries = []
        for node in summary_nodes:
            summary = Settings.llm.complete(f"Summarize in 1-2 sentences:\n\n{node.text}")
            summaries.append(str(summary))
        
        overall_summary = Settings.llm.complete(
            f"Create comprehensive summary from:\n\n{chr(10).join(summaries)}"
        )
        
        for node in nodes:
            node.metadata.update({'file_name': os.path.basename(file_path)})
        self.index.insert_nodes(nodes)
        
        summary_doc = self._create_summary_document(
            file_path, overall_summary, len(nodes), doc_size
        )
        self.summary_index.insert(summary_doc)
        
        print(f"Summary: {overall_summary}")
        return str(overall_summary)
    
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
        
        # Delete from summaries collection
        summary_data = self.summary_collection.get(include=["metadatas"])
        summary_ids_to_delete = [summary_data["ids"][i] for i, metadata in enumerate(summary_data["metadatas"]) 
                                if metadata.get('file_path') == file_path]
        if summary_ids_to_delete:
            self.summary_collection.delete(ids=summary_ids_to_delete)
            deleted_count += len(summary_ids_to_delete)
        
        # Delete from status collection
        status_data = self.status_collection.get(include=["metadatas"])
        status_ids_to_delete = [status_data["ids"][i] for i, metadata in enumerate(status_data["metadatas"]) 
                               if metadata.get('file_path') == file_path]
        if status_ids_to_delete:
            self.status_collection.delete(ids=status_ids_to_delete)
            deleted_count += len(status_ids_to_delete)
        
        print(f"Deleted {deleted_count} records for file: {file_path}")
        return deleted_count > 0

    def create_structured_documents(self, structure, parent_path="", parent_content=""):
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
                        'has_subsections': bool(item.get('children') or item.get('parts') or item.get('divisions') or item.get('subdivisions') or item.get('sections'))
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
                    full_content
                )
                documents.extend(nested_docs)
        
        return documents

    def display_all_summaries(self):
        """Display complete status and summaries in markdown format"""
        try:
            # Database status
            doc_data = self.doc_collection.get(include=["metadatas"])
            summary_data = self.summary_collection.get(include=["metadatas"])
            status_data = self.status_collection.get(include=["metadatas"])
            
            print(f"# Complete Document Status\n")
            print(f"**Database Path:** {os.path.abspath(self.db_path)}")
            print(f"**Model:** {self.model}")
            print(f"**Collections:** Documents: {len(doc_data['ids'])}, Summaries: {len(summary_data['ids'])}, Status: {len(status_data['ids'])}\n")
            
            # Processing status overview
            processed_files = self._get_processed_files()
            completed_files = self._get_completed_files()
            summary_files = {m.get('file_path') for m in summary_data['metadatas'] if m.get('file_path')}
            
            print(f"## Processing Status\n")
            print(f"- **Total processed files:** {len(processed_files)}")
            print(f"- **Completed files:** {len(completed_files)}")
            print(f"- **Files with summaries:** {len(summary_files)}")
            print(f"- **Missing summaries:** {len(processed_files - summary_files)}\n")
            
            # File status details
            if processed_files:
                print(f"## File Processing Details\n")
                all_files = {}
                
                # Collect all file info
                for metadata in doc_data['metadatas']:
                    file_path = metadata.get('file_path')
                    if file_path:
                        all_files[file_path] = {
                            'name': metadata.get('file_name', 'Unknown'),
                            'path': file_path,
                            'processed': True,
                            'completed': file_path in completed_files,
                            'has_summary': file_path in summary_files,
                            'doc_metadata': metadata
                        }
                
                # Add summary info
                for metadata in summary_data['metadatas']:
                    file_path = metadata.get('file_path')
                    if file_path and file_path in all_files:
                        all_files[file_path]['summary_metadata'] = metadata
                
                # Display file status table
                print("| File | Status | Size | Chunks | Summary |")
                print("|------|--------|------|--------|---------|")
                
                for file_path, info in sorted(all_files.items()):
                    name = info['name']
                    status = "✅ Complete" if info['completed'] else "⚠️ Partial"
                    
                    summary_meta = info.get('summary_metadata', {})
                    size = summary_meta.get('original_size', 0)
                    chunks = summary_meta.get('chunk_count', 0)
                    has_summary = "✅" if info['has_summary'] else "❌"
                    
                    size_str = f"{size:,}" if size else "N/A"
                    chunks_str = str(chunks) if chunks else "N/A"
                    
                    print(f"| {name} | {status} | {size_str} | {chunks_str} | {has_summary} |")
                
                print("\n")
            
            # Document summaries
            if not summary_data["ids"]:
                print("## No Summaries Available\n")
                print("No summaries found. Run processing to generate summaries.\n")
                return
            
            # Group summaries by document
            docs = {}
            summary_docs_data = self.summary_collection.get(include=["metadatas", "documents"])
            for i, metadata in enumerate(summary_docs_data["metadatas"]):
                file_name = metadata.get('file_name', 'Unknown')
                if file_name not in docs:
                    docs[file_name] = []
                # Add both metadata and document text
                summary_item = metadata.copy()
                summary_item['doc_text'] = summary_docs_data["documents"][i]
                docs[file_name].append(summary_item)
            
            print(f"## Document Summaries ({len(docs)} documents)\n")
            
            for file_name, summaries in sorted(docs.items()):
                print(f"### {file_name}\n")
                
                # Separate overall and chunk summaries
                overall_summaries = [s for s in summaries if s.get('summary_type') == 'overall']
                chunk_summaries = [s for s in summaries if s.get('summary_type') == 'chunk']
                
                # Display overall summary first
                for summary_data in overall_summaries:
                    summary_text = summary_data.get('doc_text', 'No summary')
                    chunk_count = summary_data.get('chunk_count', 0)
                    original_size = summary_data.get('original_size', 0)
                    file_path = summary_data.get('file_path', 'Unknown path')
                    
                    print(f"**Path:** `{file_path}`")
                    print(f"**Size:** {original_size:,} chars | **Chunks:** {chunk_count}")
                    print(f"**Type:** Overall Summary\n")
                    print(f"{summary_text}\n")
                
                # Display chunk summaries if any
                if chunk_summaries:
                    print(f"**Chunk Summaries ({len(chunk_summaries)} chunks):**\n")
                    for summary_data in sorted(chunk_summaries, key=lambda x: x.get('chunk_index', 0)):
                        summary_text = summary_data.get('doc_text', 'No summary')
                        chunk_index = summary_data.get('chunk_index', 0)
                        print(f"- **Chunk {chunk_index + 1}:** {summary_text}")
                    print()
                
                print("---\n")
                    
        except Exception as e:
            print(f"Error displaying status: {e}")
            import traceback
            traceback.print_exc()
    

    

    
    def tune_parameters(self, eval_questions=None, eval_answers=None):
        """Simple parameter tuning for optimal performance"""
        if not EVALUATION_AVAILABLE:
            print("Evaluation not available. Using default parameters.")
            return {'similarity_top_k': 5}
        
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
    
    def query(self, query_text, use_direct=False, similarity_top_k=5):
        print(f"Query: {query_text}")
        print(f"Mode: {'Direct vector' if use_direct else 'Keyword -> Summary -> Vector'}")
        
        # Check if query is asking for specific section numbers
        import re
        section_matches = re.findall(r'Section\s+(\d+)', query_text, re.IGNORECASE)
        # Also check for standalone numbers that might be section references
        standalone_numbers = re.findall(r'\b(\d{2,3})\b', query_text)
        # Combine both types of matches
        all_section_nums = list(set(section_matches + standalone_numbers))
        if all_section_nums:
            print(f"Detected section number query: {all_section_nums}")
            # Use much higher top_k for section-specific queries
            similarity_top_k = max(similarity_top_k, 50)
        
        if use_direct:
            enhanced_query = f"""{query_text}
            
            Find and return the exact content of the requested section, part, or schedule. Include the full text and any subsections."""
            
            # Get retriever to see what documents are retrieved
            retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
            nodes = retriever.retrieve(enhanced_query)
            
            # If looking for specific sections, boost nodes that contain those sections
            if all_section_nums:
                boosted_nodes = []
                for node in nodes:
                    boost_applied = False
                    for section_num in all_section_nums:
                        if f"Section {section_num}:" in node.text or f"Section {section_num} " in node.text:
                            node.score += 0.5  # Boost score significantly
                            if not boost_applied:
                                boosted_nodes.insert(0, node)  # Move to front
                                boost_applied = True
                            break
                    if not boost_applied:
                        boosted_nodes.append(node)
                nodes = boosted_nodes
            
            print(f"\nVector search retrieved {len(nodes)} documents:")
            for i, node in enumerate(nodes[:10], 1):  # Show top 10
                doc_name = node.metadata.get('source', node.metadata.get('file_name', 'N/A'))
                print(f"{i}. Score: {node.score:.3f} | Path: {node.metadata.get('hierarchy_path', 'N/A')} | Doc: {doc_name}")
                print(f"   Text preview: {node.text[:100]}...")
            
            response = self.index.as_query_engine(similarity_top_k=similarity_top_k).query(enhanced_query)
            print(f"\nDirect response:\n{response}")
            return response
        
        # Try direct section search first for section queries
        if all_section_nums:
            all_section_results = []
            for section_num in all_section_nums:
                section_results = self.section_search(section_num)
                all_section_results.extend(section_results)
            
            if all_section_results:
                print(f"\nDirect section matches found for sections: {all_section_nums}")
                combined_text = "\n\n".join([result['text'] for result in all_section_results])
                response = Settings.llm.complete(f"Based on the text: {combined_text}\n\nQuestion: {query_text}")
                print(f"\nSection-based response:\n{response}")
                return response
        
        # Try keyword search second
        keyword_results = self.keyword_search(query_text)
        if keyword_results:
            print(f"\nKeyword matches: {len(keyword_results)}")
            combined_text = "\n\n".join([result['text'] for result in keyword_results])
            print(f"Using {len(keyword_results)} documents")
            response = Settings.llm.complete(f"Based on the text: {combined_text}\n\nQuestion: {query_text}")
            print(f"\nKeyword-based response:\n{response}")
            return response
        
        # Try summary search second
        summary_results = self.summary_search(query_text)
        if summary_results:
            print(f"\nFound {len(summary_results)} relevant summaries:")
            for i, result in enumerate(summary_results):
                print(f"{i+1}. {result['file_name']} (score: {result['score']:.3f})")
                print(f"Summary: {result['summary']}")
            
            combined_context = "\n\n".join([f"File: {result['file_name']}\nSummary: {result['summary']}" for result in summary_results])
            response = Settings.llm.complete(f"Based on:\n{combined_context}\n\nQuestion: {query_text}")
            print(f"\nSummary-based response:\n{response}")
            return response
        
        # Fallback to vector search
        print("\nUsing vector search fallback")
        enhanced_query = f"""{query_text}
        
        Find and return the exact content of the requested section, part, or schedule. Include the full text and any subsections."""
        
        # Check if query is asking for a specific section number
        import re
        section_match = re.search(r'Section\s+(\d+)', query_text, re.IGNORECASE)
        if section_match:
            section_num = section_match.group(1)
            print(f"Detected section number query: {section_num}")
            # Use much higher top_k for section-specific queries
            similarity_top_k = max(similarity_top_k, 50)
        
        # Get retriever to see what documents are retrieved
        retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
        nodes = retriever.retrieve(enhanced_query)
        
        # If looking for specific section, boost nodes that contain that section
        if section_match:
            section_num = section_match.group(1)
            boosted_nodes = []
            for node in nodes:
                if f"Section {section_num}:" in node.text or f"Section {section_num} " in node.text:
                    node.score += 0.5  # Boost score significantly
                    boosted_nodes.insert(0, node)  # Move to front
                else:
                    boosted_nodes.append(node)
            nodes = boosted_nodes
        
        print(f"\nVector search retrieved {len(nodes)} documents:")
        for i, node in enumerate(nodes[:10], 1):  # Show top 10
            doc_name = node.metadata.get('source', node.metadata.get('file_name', 'N/A'))
            print(f"{i}. Score: {node.score:.3f} | Path: {node.metadata.get('hierarchy_path', 'N/A')} | Doc: {doc_name}")
            print(f"   Text preview: {node.text[:100]}...")
        
        response = self.index.as_query_engine(similarity_top_k=similarity_top_k).query(enhanced_query)
        print(f"\nVector response:\n{response}")
        return response