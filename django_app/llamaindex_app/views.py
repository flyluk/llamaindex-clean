import json
import os
import requests
import tempfile
import shutil
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
        "embed_base_url": "http://localhost:11434",
        "ollama_base_url": "http://localhost:11434",
        "default_model": "deepseek-r1:14b",
        "default_kb": "default"
    }
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return {**default_config, **json.load(f)}
    return default_config

def get_ollama_models(ollama_url="http://localhost:11434"):
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            return models if models else ["deepseek-r1:14b"]
    except:
        pass
    return ["deepseek-r1:14b", "gpt-oss:20b", "llama3.2:3b", "llama3.1:8b", "qwen2.5:7b", "deepseek-r1:8b"]

def get_ollama_library():
    return [
        "llama3.2:3b", "llama3.2:1b", "llama3.1:8b", "llama3.1:70b",
        "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "qwen2.5:72b",
        "deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:8b", "deepseek-r1:14b", "deepseek-r1:32b",
        "mistral:7b", "mixtral:8x7b", "codellama:7b", "codellama:13b",
        "phi3:3.8b", "phi3:14b", "gemma2:2b", "gemma2:9b", "gemma2:27b",
        "nomic-embed-text", "mxbai-embed-large", "all-minilm"
    ]

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
                embed_base_url=config["embed_base_url"],
                ollama_base_url=config["ollama_base_url"]
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

def index(request):
    config = load_config()
    installed_models = get_ollama_models(config["ollama_base_url"])
    knowledge_bases = get_knowledge_bases()
    context = {
        'config': config,
        'installed_models': installed_models,
        'knowledge_bases': knowledge_bases
    }
    return render(request, 'index.html', context)

def search(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        direct_vector = request.POST.get('direct_vector') == 'true'
        print(f"DEBUG: Search query: {query}, direct_vector: {direct_vector}")
        
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
                embed_base_url=config["embed_base_url"],
                ollama_base_url=config["ollama_base_url"]
            )
            print("DEBUG: Loading index")
            indexer.load_or_create_index()
            print("DEBUG: Querying")
            result = indexer.query(query, direct_vector)
            print(f"DEBUG: Query result: {result}")
            return JsonResponse({'success': True, 'result': str(result)})
        except Exception as e:
            import traceback
            print(f"DEBUG: Exception in search: {e}")
            print(f"DEBUG: Stack trace: {traceback.format_exc()}")
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request'})

def upload(request):
    if request.method == 'POST':
        files = request.FILES.getlist('files')
        config = load_config()
        
        kb_name = config.get('default_kb', 'default')
        db_path = get_db_path(kb_name)
        
        indexer = DocumentIndexer(
            target_dir="", 
            db_path=db_path, 
            model=config["default_model"],
            embed_base_url=config["embed_base_url"],
            ollama_base_url=config["ollama_base_url"]
        )
        indexer.load_or_create_index()
        
        temp_dir = tempfile.mkdtemp()
        try:
            for file in files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, 'wb') as f:
                    for chunk in file.chunks():
                        f.write(chunk)
            
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
                    embed_base_url=config["embed_base_url"],
                    ollama_base_url=config["ollama_base_url"]
                )
                indexer.load_or_create_index()
                messages.success(request, f'Knowledge base "{kb_name}" created successfully!')
            except Exception as e:
                messages.error(request, f'Error creating knowledge base: {str(e)}')
        
        return redirect('admin_panel')
    
    config = load_config()
    installed_models = get_ollama_models(config["ollama_base_url"])
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
    installed_models = get_ollama_models(config["ollama_base_url"])
    knowledge_bases = get_knowledge_bases()
    
    context = {
        'library_models': library_models,
        'installed_models': installed_models,
        'knowledge_bases': knowledge_bases,
        'config': config
    }
    return render(request, 'models.html', context)

@csrf_exempt
@require_POST
def pull_model(request):
    model_name = request.POST.get('model_name')
    config = load_config()
    
    try:
        response = requests.post(f"{config['ollama_base_url']}/api/pull", 
                               json={"name": model_name}, timeout=300)
        if response.status_code == 200:
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
        response = requests.delete(f"{config['ollama_base_url']}/api/delete", 
                                 json={"name": model_name}, timeout=30)
        if response.status_code == 200:
            return JsonResponse({'success': True, 'message': f'Successfully deleted {model_name}'})
        else:
            return JsonResponse({'success': False, 'error': f'Failed to delete {model_name}'})
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
    
    # Initialize DocumentIndexer with new model
    kb_name = config.get('default_kb', 'default')
    db_path = get_db_path(kb_name)
    
    indexer = DocumentIndexer(
        target_dir="",
        db_path=db_path,
        model=model,
        embed_base_url=config["embed_base_url"],
        ollama_base_url=config["ollama_base_url"]
    )
    indexer.load_or_create_index()
    
    return JsonResponse({'success': True})

@csrf_exempt
@require_POST
def update_config(request):
    config_file = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
    config = load_config()
    
    if 'embed_base_url' in request.POST:
        config['embed_base_url'] = request.POST.get('embed_base_url')
    if 'ollama_base_url' in request.POST:
        config['ollama_base_url'] = request.POST.get('ollama_base_url')
    
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
            embed_base_url=config["embed_base_url"],
            ollama_base_url=config["ollama_base_url"]
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
            embed_base_url=config["embed_base_url"],
            ollama_base_url=config["ollama_base_url"]
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
        
        return {
            'db_path': os.path.abspath(indexer.db_path),
            'model': indexer.model,
            'doc_count': len(doc_data['ids']),
            'status_count': len(status_data['ids']),
            'processed_files_count': len(processed_files),
            'completed_files_count': len(completed_files),
            'files': list(all_files.values())
        }
    except Exception as e:
        return {'error': str(e)}