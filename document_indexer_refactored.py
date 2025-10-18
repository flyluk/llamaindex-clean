"""Refactored DocumentIndexer using modular components."""

import os
import site
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from llm_factory import LLMFactory
from storage_manager import StorageManager
from document_processor import DocumentProcessor
from structure_detect_n_extract import extract_structure
from rag_prompt_template import create_rag_prompt

# Setup TensorRT environment
site_packages = site.getsitepackages()[0]
tensorrt_lib_path = os.path.join(site_packages, 'tensorrt_cu13_libs')
if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['LD_LIBRARY_PATH'] = f"{tensorrt_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
else:
    os.environ['LD_LIBRARY_PATH'] = tensorrt_lib_path


class DocumentIndexer:
    """Main document indexing and querying class."""
    
    def __init__(self, target_dir="", db_path="./chroma_db", model="deepseek-r1:7b", 
                 base_url="http://localhost:11434", embed_url=None, api_key=None, 
                 embed_model="nomic-embed-text", context_length=32768, use_sentence_splitter=False):
        self.target_dir = target_dir
        self.model = model
        self.base_url = base_url
        self.embed_url = embed_url or base_url
        self.api_key = api_key
        self.embed_model = embed_model
        self.context_length = context_length
        self.use_sentence_splitter = use_sentence_splitter
        
        # Setup components
        Settings.node_parser = SentenceSplitter(chunk_size=3000, chunk_overlap=300)
        LLMFactory.setup_settings(base_url, model, self.embed_url, embed_model, api_key, context_length)
        self.storage = StorageManager(db_path)
        self.index = None
    
    def load_or_create_index(self):
        """Load existing index or create empty index structure."""
        num_docs = len(self.storage.doc_collection.get()["ids"])
        print(f"ChromaDB status: {num_docs} documents (DB: {self.storage.db_path})")
        
        if os.path.exists(self.storage.db_path) and num_docs > 0:
            print("Loading existing index")
            self.index = VectorStoreIndex.from_vector_store(
                self.storage.doc_store, 
                storage_context=self.storage.storage_context
            )
        else:
            print("Creating new index")
            self.index = VectorStoreIndex([], storage_context=self.storage.storage_context)
        
        print("Index loading/creation completed")
    
    def process_single_file(self, file_path):
        """Process a single file."""
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
        
        print(f"Processing single file: {os.path.basename(file_path)}")
        
        try:
            doc, structure_info = DocumentProcessor.process_document(file_path, self.use_sentence_splitter)
            
            if self.use_sentence_splitter or not structure_info:
                print(f"📝 Using sentence splitter")
                self.index.insert(doc)
            else:
                self._process_with_structure(doc, structure_info)
            
            self.storage.mark_file_complete(file_path)
            print(f"✅ Successfully processed: {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"❌ Failed to process {os.path.basename(file_path)}: {e}")
    
    def _process_with_structure(self, doc, structure_info):
        """Process document with structure detection."""
        if structure_info['is_legal']:
            print(f"📋 Legal document detected")
            structure = extract_structure(doc.text)
            if structure:
                print(f"✅ Extracted {len(structure)} legal structural elements")
                structured_documents = DocumentProcessor.create_structured_documents(
                    structure, file_path=doc.metadata.get('file_path')
                )
                print(f"➡️ Created {len(structured_documents)} structured documents")
                for struct_doc in structured_documents:
                    self.index.insert(struct_doc)
        elif structure_info['has_structure']:
            print(f"📄 Structured document detected")
            structured_documents = DocumentProcessor.create_general_structured_documents(
                structure_info['detection_result'], doc
            )
            print(f"✅ Created {len(structured_documents)} general structured documents")
            for struct_doc in structured_documents:
                self.index.insert(struct_doc)
        else:
            print(f"📝 Plain document")
            self.index.insert(doc)
    
    def process_files(self, target_path=None):
        """Process files from target directory."""
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
                    
        completed_files = self.storage.get_completed_files()
        remaining_docs = [doc for doc in documents if doc.metadata.get('file_path') not in completed_files]
        
        if remaining_docs:
            print(f"Processing: {len(remaining_docs)} files remaining (skipped {len(documents) - len(remaining_docs)} already processed)")
            
            for i, doc in enumerate(remaining_docs, 1):
                print(f"Processing {i}/{len(remaining_docs)}: {doc.metadata.get('file_name', 'Unknown')}")
                
                try:
                    if self.use_sentence_splitter:
                        print(f"📝 Using sentence splitter: {doc.metadata.get('file_name')}")
                        self.index.insert(doc)
                    else:
                        _, structure_info = DocumentProcessor.process_document(
                            doc.metadata.get('file_path'), 
                            self.use_sentence_splitter
                        )
                        if structure_info:
                            self._process_with_structure(doc, structure_info)
                        else:
                            self.index.insert(doc)
                        
                    self.storage.mark_file_complete(doc.metadata.get('file_path'))
                except Exception as e:
                    print(f"Failed to process {doc.metadata.get('file_name', 'Unknown')}: {e}")
                    continue
    
    def _load_documents(self):
        """Load documents from target directory."""
        from llama_index.core import Document
        documents = []
        
        for root, _, files in os.walk(self.target_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Handle markdown files
                if file.endswith('.md'):
                    base_path = os.path.splitext(file_path)[0]
                    original_exists = any(os.path.exists(f"{base_path}.{ext}") 
                                        for ext in ['pdf', 'docx', 'txt'])
                    if original_exists:
                        continue
                    else:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text = f.read()
                            doc = Document(
                                text=text,
                                metadata={'file_name': file, 'file_path': file_path}
                            )
                            documents.append(doc)
                        except Exception as e:
                            print(f"Error processing {file}: {e}")
                        continue
                
                # Process non-markdown files
                md_path = os.path.splitext(file_path)[0] + '.md'
                try:
                    if os.path.exists(md_path):
                        with open(md_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                    else:
                        text = DocumentProcessor.convert_with_docling(file_path)
                        with open(md_path, 'w', encoding='utf-8') as f:
                            f.write(text)
                    
                    doc = Document(
                        text=text,
                        metadata={'file_name': file, 'file_path': file_path}
                    )
                    documents.append(doc)
                except Exception as e:
                    print(f"Error processing {file}: {e}")
        
        return documents
    
    def query(self, query_text, use_direct=False, similarity_top_k=10, chat_history=None):
        """Query the index with optional hybrid search."""
        print(f"Query: {query_text}")
        print(f"Mode: {'Hybrid search' if not use_direct else 'Direct vector search'}")
        
        if use_direct:
            retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
        else:
            retriever = self._create_hybrid_retriever(similarity_top_k)
        
        nodes = retriever.retrieve(query_text)
        
        if not nodes:
            print("No relevant documents found")
            return "No relevant information found for your query."
        
        self._print_retrieval_results(nodes)
        combined_context = self._build_context(nodes)
        context_prompt = create_rag_prompt(combined_context, query_text, chat_history)
        
        self._print_cost_estimate(context_prompt)
        response = Settings.llm.complete(context_prompt)
        self._print_response_cost(response)
        
        print(f"\nContext-based response:\n{response}")
        return response
    
    def _create_hybrid_retriever(self, similarity_top_k):
        """Create hybrid retriever with BM25 and vector search."""
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.core.retrievers import QueryFusionRetriever
        
        # Get all nodes for BM25
        all_nodes = []
        doc_data = self.storage.doc_collection.get(include=["documents", "metadatas"])
        for i, doc_text in enumerate(doc_data["documents"]):
            node = TextNode(text=doc_text, metadata=doc_data["metadatas"][i])
            all_nodes.append(node)
        
        # Create retrievers
        vector_retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
        bm25_retriever = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=similarity_top_k)
        
        # Combine with QueryFusionRetriever
        return QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=similarity_top_k,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
            verbose=True
        )
    
    def _build_context(self, nodes):
        """Build context string from retrieved nodes."""
        context_parts = []
        for i, node in enumerate(nodes, 1):
            hierarchy = node.metadata.get('hierarchy_path', '')
            file_name = node.metadata.get('file_name', f'source_{i}')
            
            if hierarchy:
                context_parts.append(f"<source_id>{file_name}</source_id>\n[{hierarchy}]\n{node.text}")
            else:
                context_parts.append(f"<source_id>{file_name}</source_id>\n{node.text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _print_retrieval_results(self, nodes):
        """Print retrieval results."""
        print(f"\nRetrieved {len(nodes)} relevant documents:")
        for i, node in enumerate(nodes[:5], 1):
            doc_name = node.metadata.get('file_name', 'N/A')
            hierarchy = node.metadata.get('hierarchy_path', 'N/A')
            print(f"{i}. Score: {node.score:.3f} | {hierarchy} | {doc_name}")
            print(f"   Preview: {node.text[:100]}...")
    
    def _print_cost_estimate(self, context_prompt):
        """Print cost estimate for input."""
        context_length = len(context_prompt)
        input_tokens = context_length // 4
        input_cost = (input_tokens / 1000) * 0.00135
        
        print(f"\nGenerating response...")
        print(f"Context length: {context_length:,} characters (~{input_tokens:,} tokens)")
        print(f"Estimated input cost: ${input_cost:.6f} USD")
    
    def _print_response_cost(self, response):
        """Print cost estimate for output."""
        output_length = len(str(response))
        output_tokens = output_length // 4
        output_cost = (output_tokens / 1000) * 0.0054
        
        print(f"\nResponse length: {output_length:,} characters (~{output_tokens:,} tokens)")
        print(f"Estimated output cost: ${output_cost:.6f} USD")
    
    # Delegate methods to storage manager
    def delete_file(self, file_path):
        return self.storage.delete_file(file_path)
    
    def display_status(self):
        return self.storage.get_status(self.model)
    
    def init_and_process_files(self):
        """Initialize index and process files if target_dir is set."""
        self.load_or_create_index()
        if self.target_dir:
            self.process_files()
    
    def sync_new_files(self):
        """Sync new files that aren't in the index yet."""
        # Implementation similar to original if needed
        pass
    
    def get_dimension_info(self, collection_name=None):
        """Get dimension information from ChromaDB collections."""
        try:
            collections = self.storage.client.list_collections()
            
            if collection_name:
                collections = [c for c in collections if c.name == collection_name]
                if not collections:
                    return {'error': f"Collection '{collection_name}' not found"}
            
            result = {
                'db_path': os.path.abspath(self.storage.db_path),
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
