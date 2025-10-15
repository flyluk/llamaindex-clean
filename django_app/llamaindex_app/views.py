import json
import os
import re
import requests
import tempfile
import shutil
import chromadb
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

# Import your existing classes
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, parent_dir)
try:
    from document_indexer import DocumentIndexer
    print("DEBUG: DocumentIndexer imported successfully")
except ImportError as e:
    print(f"DEBUG: Import error: {e}")
    DocumentIndexer = None

def load_config():
    config_file = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
    default_config = {
        "base_url": "http://localhost:11434",
        "embed_url": "http://localhost:11434",
        "api_key": "",
        "default_model": "deepseek-r1:14b",
        "embed_model": "nomic-embed-text",
        "default_kb": "default",
        "context_length": 32768,
        "ollama_library": [
            "llama3.2:3b", "llama3.2:1b", "llama3.1:8b", "llama3.1:70b",
            "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "qwen2.5:72b",
            "deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:8b", "deepseek-r1:14b", "deepseek-r1:32b",
            "mistral:7b", "mixtral:8x7b", "codellama:7b", "codellama:13b",
            "phi3:3.8b", "phi3:14b", "gemma2:2b", "gemma2:9b", "gemma2:27b",
            "nomic-embed-text", "mxbai-embed-large", "all-minilm"
        ]
    }
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
            # Ensure embed_library exists in loaded config
            if 'embed_library' not in loaded_config:
                loaded_config['embed_library'] = default_config['embed_library']
            return {**default_config, **loaded_config}
    return default_config

def is_azure_endpoint(config):
    """Check if the configuration uses Azure endpoints"""
    return "azure" in config.get("base_url", "").lower() or "azure" in config.get("embed_url", "").lower()

def get_ollama_models(ollama_url="http://localhost:11434"):
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            print(f"DEBUG: Retrieved {len(models)} models from Ollama: {models}")
            return models if models else []
        else:
            print(f"DEBUG: Ollama API returned status {response.status_code}")
    except Exception as e:
        print(f"DEBUG: Failed to connect to Ollama at {ollama_url}: {e}")
    return []

def get_ollama_library():
    config = load_config()
    return config.get('ollama_library', [
        "llama3.2:3b", "llama3.2:1b", "llama3.1:8b", "llama3.1:70b",
        "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "qwen2.5:72b",
        "deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:8b", "deepseek-r1:14b", "deepseek-r1:32b",
        "mistral:7b", "mixtral:8x7b", "codellama:7b", "codellama:13b",
        "phi3:3.8b", "phi3:14b", "gemma2:2b", "gemma2:9b", "gemma2:27b",
        "nomic-embed-text", "mxbai-embed-large", "all-minilm"
    ])

def get_knowledge_bases():
    import glob
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    db_dirs = glob.glob(os.path.join(parent_dir, '*_db'))
    kb_options = {}
    for db_dir in db_dirs:
        clean_name = os.path.basename(db_dir).replace('_db', '')
        if clean_name == 'chroma':
            clean_name = 'default'
        kb_options[clean_name] = db_dir
    
    # Create default chroma_db if no databases exist
    if not kb_options and DocumentIndexer:
        try:
            config = load_config()
            default_db_path = get_db_path('default')
            indexer = DocumentIndexer(
                target_dir="",
                db_path=default_db_path,
                model=config["default_model"],
                base_url=config["base_url"],
                embed_url=config.get("embed_url", config["base_url"]),
                api_key=config.get("api_key"),
                embed_model=config.get("embed_model", "nomic-embed-text")
            )
            indexer.load_or_create_index()
            kb_options['default'] = default_db_path
            print(f"Created default knowledge base at {default_db_path}")
        except Exception as e:
            print(f"Failed to create default knowledge base: {e}")
    
    return kb_options

def get_db_path(kb_name):
    """Get correct database path, handling default -> chroma_db mapping"""
    if kb_name == 'default':
        return os.path.join(os.path.dirname(__file__), '..', '..', 'chroma_db')
    else:
        return os.path.join(os.path.dirname(__file__), '..', '..', f'{kb_name}_db')

def format_think_tags(text):
    """Format <think></think> tags as collapsible HTML details"""
    text = str(text)
    # Replace HTML entities
    text = text.replace('&lt;think&gt;', '<think>')
    text = text.replace('&lt;/think&gt;', '</think>')
    
    # Check if there are any think tags
    if '<think>' not in text or '</think>' not in text:
        return text
    
    # Convert think tags to collapsible HTML details
    text = re.sub(r'<think>', '<details><summary>🤔 Thinking</summary>\n\n', text)
    text = re.sub(r'</think>', '\n</details>\n\n---\n\n', text)
    
    # Clean up extra newlines
    text = re.sub(r'\n\n\n+', '\n\n', text)
    text = text.strip()
    
    return text

def index(request):
    config = load_config()
    is_azure = is_azure_endpoint(config)
    
    # Only get Ollama models if not using Azure
    installed_models = [] if is_azure else get_ollama_models(config.get("base_url", "http://localhost:11434"))
    knowledge_bases = get_knowledge_bases()
    
    # Get knowledge status
    kb_name = config.get('default_kb', 'default')
    db_path = get_db_path(kb_name)
    knowledge_status = None
    
    try:
        indexer = DocumentIndexer(
            target_dir="",
            db_path=db_path,
            model=config["default_model"],
            base_url=config["base_url"],
            embed_url=config.get("embed_url", config["base_url"]),
            api_key=config.get("api_key"),
            embed_model=config.get("embed_model", "nomic-embed-text")
        )
        indexer.load_or_create_index()
        knowledge_status = get_status_data(indexer)
    except Exception as e:
        knowledge_status = {'error': str(e)}
    
    context = {
        'config': config,
        'installed_models': installed_models,
        'knowledge_bases': knowledge_bases,
        'knowledge_status': knowledge_status,
        'is_azure': is_azure
    }
    return render(request, 'index.html', context)

def search(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        use_vector_only = request.POST.get('use_direct_vector') == 'on'
        top_k = int(request.POST.get('top_k', 10))
        context_length = int(request.POST.get('context_length', 32768))
        chat_history = request.POST.get('chat_history', '')
        print(f"DEBUG: Search query: {query}, use_vector_only: {use_vector_only}, top_k: {top_k}, context_length: {context_length}")
        
        if not DocumentIndexer:
            print("DEBUG: DocumentIndexer not available")
            return JsonResponse({'success': False, 'error': 'DocumentIndexer not available'})
        
        config = load_config()
        print(f"DEBUG: Config loaded: {config}")
        kb_name = config.get('default_kb', 'default')
        db_path = get_db_path(kb_name)
        print(f"DEBUG: DB path: {db_path}")
        
        try:
            # Initialize indexer
            print("DEBUG: Initializing DocumentIndexer")
            indexer = DocumentIndexer(
                target_dir="",
                db_path=db_path,
                model=config["default_model"],
                base_url=config["base_url"],
                embed_url=config.get("embed_url", config["base_url"]),
                api_key=config.get("api_key"),
                embed_model=config.get("embed_model", "nomic-embed-text"),
                context_length=context_length
            )
            print("DEBUG: Loading index")
            indexer.load_or_create_index()
            print("DEBUG: Querying")
            result = indexer.query(query, use_vector_only, top_k, chat_history if chat_history else None)
            formatted_result = format_think_tags(result)
            print(f"DEBUG: Query result: {formatted_result}")
            return JsonResponse({'success': True, 'result': formatted_result})
        except Exception as e:
            import traceback
            print(f"DEBUG: Exception in search: {e}")
            print(f"DEBUG: Stack trace: {traceback.format_exc()}")
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request'})

def upload(request):
    if request.method == 'POST':
        files = request.FILES.getlist('files')
        use_sentence_splitter = request.POST.get('use_sentence_splitter') == 'true'
        config = load_config()
        
        kb_name = config.get('default_kb', 'default')
        db_path = get_db_path(kb_name)
        
        indexer = DocumentIndexer(
            target_dir="",
            db_path=db_path,
            model=config["default_model"],
            base_url=config["base_url"],
            embed_url=config.get("embed_url", config["base_url"]),
            api_key=config.get("api_key"),
            embed_model=config.get("embed_model", "nomic-embed-text"),
            use_sentence_splitter=use_sentence_splitter
        )
        indexer.load_or_create_index()
        
        temp_dir = tempfile.mkdtemp()
        try:
            # Check for existing files and delete them from database
            for file in files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, 'wb') as f:
                    for chunk in file.chunks():
                        f.write(chunk)
                
                # Check if file exists in database and delete it
                if indexer.delete_file(file_path):
                    print(f"Deleted existing file from database: {file.name}")
            
            indexer.process_files(temp_dir)
            messages.success(request, f'Successfully processed {len(files)} files!')
            
        except Exception as e:
            messages.error(request, f'Error processing files: {str(e)}')
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    return redirect('index')

def admin_panel(request):
    if request.method == 'POST':
        kb_name = request.POST.get('kb_name')
        if kb_name:
            config = load_config()
            db_path = get_db_path(kb_name)
            
            try:
                indexer = DocumentIndexer(
                    target_dir="",
                    db_path=db_path,
                    model=config["default_model"],
                    base_url=config["base_url"],
                    embed_url=config.get("embed_url", config["base_url"]),
                    api_key=config.get("api_key"),
                    embed_model=config.get("embed_model", "nomic-embed-text")
                )
                indexer.load_or_create_index()
                messages.success(request, f'Knowledge base "{kb_name}" created successfully!')
            except Exception as e:
                messages.error(request, f'Error creating knowledge base: {str(e)}')
        
        return redirect('admin_panel')
    
    config = load_config()
    is_azure = is_azure_endpoint(config)
    installed_models = [] if is_azure else get_ollama_models(config.get("base_url", "http://localhost:11434"))
    knowledge_bases = get_knowledge_bases()
    context = {
        'config': config,
        'installed_models': installed_models,
        'knowledge_bases': knowledge_bases
    }
    return render(request, 'admin_panel.html', context)

def models_view(request):
    config = load_config()
    library_models = get_ollama_library()
    is_azure = is_azure_endpoint(config)
    installed_models = [] if is_azure else get_ollama_models(config.get("base_url", "http://localhost:11434"))
    knowledge_bases = get_knowledge_bases()
    
    context = {
        'library_models': library_models,
        'installed_models': installed_models,
        'knowledge_bases': knowledge_bases,
        'config': config
    }
    return render(request, 'models.html', context)

from django.http import StreamingHttpResponse

@csrf_exempt
@require_POST
def pull_model(request):
    model_name = request.POST.get('model_name')
    config = load_config()
    
    try:
        response = requests.post(f"{config['base_url']}/api/pull", 
                               json={"name": model_name}, timeout=300)
        if response.status_code == 200:
            try:
                result = response.json()
                if result.get('status') == 'success' or 'error' not in result:
                    return JsonResponse({'success': True, 'message': f'Successfully pulled {model_name}'})
                else:
                    return JsonResponse({'success': False, 'error': result.get('error', 'Unknown error')})
            except:
                return JsonResponse({'success': True, 'message': f'Successfully pulled {model_name}'})
        else:
            return JsonResponse({'success': False, 'error': f'Failed to pull {model_name}'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@csrf_exempt
@require_POST
def delete_model(request):
    model_name = request.POST.get('model_name')
    config = load_config()
    
    try:
        response = requests.delete(f"{config['base_url']}/api/delete", 
                                 json={"name": model_name}, timeout=30)
        if response.status_code == 200:
            return JsonResponse({'success': True, 'message': f'Successfully deleted {model_name}'})
        else:
            return JsonResponse({'success': False, 'error': f'Failed to delete {model_name}'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@csrf_exempt
@require_POST
def generate_followup(request):
    query = request.POST.get('query')
    response = request.POST.get('response')
    config = load_config()
    
    try:
        prompt = f"""Based on this question and answer, generate exactly 3 short follow-up questions that a user might want to ask next. Return only the questions, one per line, without numbering or bullet points.

Original Question: {query}

Answer: {response[:500]}...

Generate 3 follow-up questions:"""
        
        # Detect service type and use appropriate LLM
        base_is_azure = "azure" in config["base_url"].lower()
        
        if base_is_azure:
            from llama_index.llms.azure_openai import AzureOpenAI
            llm = AzureOpenAI(
                api_key=config.get("api_key"),
                azure_endpoint=config["base_url"],
                engine=config["default_model"],
                api_version="2024-02-01"
            )
        elif config.get("api_key"):
            from llama_index.llms.openai import OpenAI
            llm = OpenAI(
                api_key=config["api_key"],
                model=config["default_model"],
                api_base=config["base_url"]
            )
        else:
            from llama_index.llms.ollama import Ollama
            llm = Ollama(
                model=config["default_model"],
                base_url=config["base_url"],
                request_timeout=30.0
            )
        
        result = llm.complete(prompt)
        prompts = [line.strip() for line in str(result).strip().split('\n') if line.strip() and not line.strip().startswith(('#', '-', '*'))]
        prompts = prompts[:3]  # Take only first 3
        
        return JsonResponse({'success': True, 'prompts': prompts})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@csrf_exempt
@require_POST
def pull_custom_model(request):
    model_name = request.POST.get('model_name')
    config = load_config()
    config_file = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
    
    try:
        # Try to pull the model
        response = requests.post(f"{config['base_url']}/api/pull", 
                               json={"name": model_name}, timeout=300)
        if response.status_code == 200:
            try:
                result = response.json()
                if result.get('status') == 'success' or 'error' not in result:
                    # Add to ollama_library if not already present
                    if model_name not in config.get('ollama_library', []):
                        config.setdefault('ollama_library', []).append(model_name)
                        with open(config_file, 'w') as f:
                            json.dump(config, f, indent=2)
                    return JsonResponse({'success': True, 'message': f'Successfully pulled {model_name}'})
                else:
                    return JsonResponse({'success': False, 'error': result.get('error', 'Model not found or failed to pull')})
            except:
                # Add to ollama_library if not already present
                if model_name not in config.get('ollama_library', []):
                    config.setdefault('ollama_library', []).append(model_name)
                    with open(config_file, 'w') as f:
                        json.dump(config, f, indent=2)
                return JsonResponse({'success': True, 'message': f'Successfully pulled {model_name}'})
        else:
            return JsonResponse({'success': False, 'error': f'Model {model_name} not found or failed to pull'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@csrf_exempt
@require_POST
def delete_file(request):
    file_path = request.POST.get('file_path')
    config = load_config()
    kb_name = config.get('default_kb', 'default')
    db_path = get_db_path(kb_name)
    
    try:
        indexer = DocumentIndexer(
            target_dir="",
            db_path=db_path,
            model=config["default_model"],
            base_url=config["base_url"],
            embed_url=config.get("embed_url", config["base_url"]),
            api_key=config.get("api_key"),
            embed_model=config.get("embed_model", "nomic-embed-text")
        )
        indexer.load_or_create_index()
        
        if indexer.delete_file(file_path):
            return JsonResponse({'success': True, 'message': f'Successfully deleted file'})
        else:
            return JsonResponse({'success': False, 'error': 'File not found in database'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@csrf_exempt
@require_POST
def change_model(request):
    model = request.POST.get('model')
    config_file = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
    
    config = load_config()
    config['default_model'] = model
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    return JsonResponse({'success': True})

@csrf_exempt
@require_POST
def update_config(request):
    config_file = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
    config = load_config()
    
    if 'base_url' in request.POST:
        config['base_url'] = request.POST.get('base_url')
    if 'embed_url' in request.POST:
        config['embed_url'] = request.POST.get('embed_url')
    if 'embed_model' in request.POST:
        config['embed_model'] = request.POST.get('embed_model')
    if 'context_length' in request.POST:
        config['context_length'] = int(request.POST.get('context_length'))
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    return JsonResponse({'success': True})

@csrf_exempt
@require_POST
def get_stats(request):
    config = load_config()
    kb_name = config.get('default_kb', 'default')
    db_path = get_db_path(kb_name)
    
    try:
        indexer = DocumentIndexer(
            target_dir="",
            db_path=db_path,
            model=config["default_model"],
            base_url=config["base_url"],
            embed_url=config.get("embed_url", config["base_url"]),
            api_key=config.get("api_key"),
            embed_model=config.get("embed_model", "nomic-embed-text")
        )
        indexer.load_or_create_index()
        
        doc_count = len(indexer.doc_collection.get()["ids"])
        
        return JsonResponse({
            'success': True,
            'doc_count': doc_count,
            'db_path': db_path,
            'kb_name': kb_name
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@csrf_exempt
@require_POST
def change_kb(request):
    kb_name = request.POST.get('kb_name')
    config_file = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
    
    config = load_config()
    config['default_kb'] = kb_name
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    return JsonResponse({'success': True})

@csrf_exempt
@require_POST
def get_status(request):
    config = load_config()
    kb_name = config.get('default_kb', 'default')
    db_path = get_db_path(kb_name)
    
    try:
        indexer = DocumentIndexer(
            target_dir="",
            db_path=db_path,
            model=config["default_model"],
            base_url=config["base_url"],
            embed_url=config.get("embed_url", config["base_url"]),
            api_key=config.get("api_key"),
            embed_model=config.get("embed_model", "nomic-embed-text")
        )
        indexer.load_or_create_index()
        
        status_data = get_status_data(indexer)
        
        return JsonResponse({
            'success': True,
            'status_data': status_data
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@csrf_exempt
@require_POST
def delete_db(request):
    config = load_config()
    kb_name = config.get('default_kb', 'default')
    db_path = get_db_path(kb_name)
    
    try:
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            return JsonResponse({'success': True, 'message': f'Database "{kb_name}" deleted successfully'})
        else:
            return JsonResponse({'success': False, 'error': 'Database not found'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@csrf_exempt
@require_POST
def save_history(request):
    history_file = os.path.join(os.path.dirname(__file__), '..', '..', 'history.json')
    try:
        history = json.loads(request.POST.get('history', '[]'))
        with open(history_file, 'w') as f:
            json.dump(history, f)
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

def load_history(request):
    history_file = os.path.join(os.path.dirname(__file__), '..', '..', 'history.json')
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        return JsonResponse({'success': True, 'history': history})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

def chunks_view(request):
    config = load_config()
    knowledge_bases = get_knowledge_bases()
    context = {
        'config': config,
        'knowledge_bases': knowledge_bases
    }
    return render(request, 'chunks.html', context)

@csrf_exempt
def get_documents(request):
    config = load_config()
    kb_name = config.get('default_kb', 'default')
    db_path = get_db_path(kb_name)
    
    try:
        indexer = DocumentIndexer(
            target_dir="",
            db_path=db_path,
            model=config["default_model"],
            base_url=config["base_url"],
            embed_url=config.get("embed_url", config["base_url"]),
            api_key=config.get("api_key"),
            embed_model=config.get("embed_model", "nomic-embed-text")
        )
        indexer.load_or_create_index()
        
        doc_data = indexer.doc_collection.get(include=["metadatas"])
        documents = {}
        
        for metadata in doc_data['metadatas']:
            file_path = metadata.get('file_path')
            if file_path:
                file_name = metadata.get('file_name', os.path.basename(file_path))
                if file_path not in documents:
                    documents[file_path] = file_name
        
        return JsonResponse({'success': True, 'documents': documents})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@csrf_exempt
def get_chunks(request):
    file_path = request.GET.get('file_path')
    config = load_config()
    kb_name = config.get('default_kb', 'default')
    db_path = get_db_path(kb_name)
    
    try:
        indexer = DocumentIndexer(
            target_dir="",
            db_path=db_path,
            model=config["default_model"],
            base_url=config["base_url"],
            embed_url=config.get("embed_url", config["base_url"]),
            api_key=config.get("api_key"),
            embed_model=config.get("embed_model", "nomic-embed-text")
        )
        indexer.load_or_create_index()
        
        doc_data = indexer.doc_collection.get(include=["documents", "metadatas"])
        chunks = []
        
        for i, metadata in enumerate(doc_data['metadatas']):
            if metadata.get('file_path') == file_path:
                chunk_text = doc_data['documents'][i]
                chunk_id = doc_data['ids'][i]
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'metadata': metadata
                })
        
        return JsonResponse({'success': True, 'chunks': chunks})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

def get_vector_info(indexer):
    """Get vector database info using indexer's get_dimension_info method"""
    try:
        # Use the indexer's get_dimension_info method
        dim_info = indexer.get_dimension_info(indexer.doc_collection.name)
        
        if 'error' in dim_info:
            return dim_info
        
        # Extract info for the current collection
        current_collection = None
        for collection in dim_info['collections']:
            if collection['name'] == indexer.doc_collection.name:
                current_collection = collection
                break
        
        if not current_collection:
            return {'error': 'Current collection not found'}
        
        return {
            'dimension': current_collection['dimension'],
            'total_documents': current_collection['total_documents'],
            'collection_name': current_collection['name'],
            'embed_model': dim_info['config']['embed_model'],
            'embed_url': dim_info['config']['embed_url'],
            'client_type': type(indexer.client).__name__,
            'chroma_version': getattr(chromadb, '__version__', 'Unknown')
        }
    except Exception as e:
        return {'error': str(e)}

def get_status_data(indexer):
    """Get status data as JSON instead of printing"""
    try:
        doc_data = indexer.doc_collection.get(include=["metadatas"])
        status_data = indexer.status_collection.get(include=["metadatas"])
        
        processed_files = {metadata.get('file_path') for metadata in doc_data.get('metadatas', []) if metadata.get('file_path')}
        completed_files = {metadata.get('file_path') for metadata in status_data.get('metadatas', []) if metadata.get('file_path')}
        
        all_files = {}
        for metadata in doc_data['metadatas']:
            file_path = metadata.get('file_path')
            if file_path:
                all_files[file_path] = {
                    'name': metadata.get('file_name', 'Unknown'),
                    'path': file_path,
                    'processed': True,
                    'completed': file_path in completed_files
                }
        
        # Get vector database technical info using indexer's get_dimension_info method
        vector_info = get_vector_info(indexer)
        
        return {
            'db_path': os.path.abspath(indexer.db_path),
            'model': indexer.model,
            'doc_count': len(doc_data['ids']),
            'status_count': len(status_data['ids']),
            'processed_files_count': len(processed_files),
            'completed_files_count': len(completed_files),
            'vector_info': vector_info,
            'files': list(all_files.values())
        }
    except Exception as e:
        return {'error': str(e)}