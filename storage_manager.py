"""ChromaDB storage management."""

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext


class StorageManager:
    """Manages ChromaDB collections and storage context."""
    
    def __init__(self, db_path="./chroma_db"):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self._setup_collections()
    
    def _setup_collections(self):
        """Setup document and status collections."""
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
        self.storage_context = StorageContext.from_defaults(vector_store=self.doc_store)
    
    def get_processed_files(self):
        """Get set of already processed file paths."""
        doc_data = self.doc_collection.get()
        return {metadata.get('file_path') for metadata in doc_data.get('metadatas', []) 
                if metadata.get('file_path')}
    
    def get_completed_files(self):
        """Get set of files marked as fully completed."""
        try:
            status_data = self.status_collection.get()
            return {metadata.get('file_path') for metadata in status_data.get('metadatas', []) 
                    if metadata.get('file_path')}
        except:
            return set()
    
    def mark_file_complete(self, file_path):
        """Mark individual file as complete."""
        import os
        file_id = f"complete_{hash(file_path)}"
        self.status_collection.upsert(
            ids=[file_id],
            documents=[f"File processing completed: {os.path.basename(file_path)}"],
            metadatas=[{"file_path": file_path, "status": "complete"}]
        )
    
    def delete_file(self, file_path):
        """Delete all data for a specific file from all collections."""
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
    
    def get_status(self, model):
        """Get complete document status as structured data."""
        import os
        try:
            doc_data = self.doc_collection.get(include=["metadatas"])
            status_data = self.status_collection.get(include=["metadatas"])
            
            processed_files = self.get_processed_files()
            completed_files = self.get_completed_files()
            
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
                'model': model,
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
