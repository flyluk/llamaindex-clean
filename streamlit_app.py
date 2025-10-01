import streamlit as st
import os
import glob
import requests
import tempfile
import json
from datetime import datetime
from document_indexer import DocumentIndexer
import io
import sys

st.set_page_config(page_title="LlamaIndex Document Search", layout="wide")

def load_search_history():
    history_file = "search_history.json"
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return []

def save_search_history(history):
    with open("search_history.json", 'w') as f:
        json.dump(history, f, indent=2)

def load_config():
    config_file = "config.json"
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

def save_config(config):
    with open("config.json", 'w') as f:
        json.dump(config, f, indent=2)

def add_to_history(query, result, kb, model):
    history = load_search_history()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "result": str(result),
        "kb": kb,
        "model": model
    }
    history.insert(0, entry)
    save_search_history(history)

@st.cache_resource
def get_ollama_models(ollama_url="http://localhost:11434"):
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            return models if models else ["deepseek-r1:14b"]
    except:
        pass
    return ["deepseek-r1:14b", "gpt-oss:20b", "llama3.2:3b", "llama3.1:8b", "qwen2.5:7b", "deepseek-r1:8b"]

@st.cache_resource
def get_indexer(db_path, model, embed_base_url="http://localhost:11434", ollama_base_url="http://localhost:11434"):
    indexer = DocumentIndexer(target_dir="", db_path=db_path, model=model, embed_base_url=embed_base_url, ollama_base_url=ollama_base_url)
    indexer.load_or_create_index()
    return indexer

def capture_output_live(func, *args):
    """Capture and display output line by line"""
    output_container = st.empty()
    output_lines = []
    
    class StreamCapture:
        def write(self, text):
            if text.strip():
                output_lines.append(text.strip())
                output_container.text_area("Processing:", "\n".join(output_lines), height=200)
        def flush(self):
            pass
    
    old_stdout = sys.stdout
    sys.stdout = StreamCapture()
    try:
        result = func(*args)
        return result, "\n".join(output_lines)
    finally:
        sys.stdout = old_stdout



# Load config first
config = load_config()

st.title("🦙 LlamaIndex Document Search")

# Show last action notification
if st.session_state.get('last_action'):
    st.info(st.session_state.last_action)
    if st.button("Clear notification"):
        st.session_state.last_action = None
        st.rerun()

# Display Python version for verification
st.sidebar.markdown(f"**Python:** {sys.version.split()[0]}")

# Sidebar
st.sidebar.header("Knowledge Base")

# Find all *_db directories and create tree structure
db_dirs = glob.glob("*_db")
if not db_dirs:
    db_dirs = ["./chroma_db"]  # fallback

# Create clean names by stripping _db suffix
kb_options = {}
for db_dir in db_dirs:
    clean_name = db_dir.replace("_db", "").replace("./", "")
    if clean_name == "chroma":
        clean_name = "default"
    kb_options[clean_name] = db_dir

# Display as expandable tree
st.sidebar.subheader("📚 Available Knowledge Bases")

# Set KB from config with fallback
config_kb = config.get("default_kb", "default")
kb_index = 0
if config_kb in kb_options:
    kb_index = list(kb_options.keys()).index(config_kb)

selected_kb = st.sidebar.radio(
    "Select Knowledge Base:",
    options=list(kb_options.keys()),
    format_func=lambda x: f"📁 {x.title()}",
    index=kb_index
)

db_path = kb_options[selected_kb]
st.sidebar.caption(f"Path: {db_path}")

# Set model from config with fallback
available_models = get_ollama_models(config["ollama_base_url"])
config_model = config.get("default_model", "deepseek-r1:14b")
model_index = 0
if config_model in available_models:
    model_index = available_models.index(config_model)

model = st.sidebar.selectbox("Model", available_models, index=model_index)

# Server configuration
st.sidebar.markdown("---")

with st.sidebar.expander("🔧 Server Configuration", expanded=False):
    change_urls = st.button("Change URLs")
    edit_mode = st.session_state.get('edit_urls', False) or change_urls
    
    if change_urls:
        st.session_state.edit_urls = True
    
    embed_base_url = st.text_input("Embed Base URL:", value=config["embed_base_url"], disabled=not edit_mode)
    ollama_base_url = st.text_input("Ollama Base URL:", value=config["ollama_base_url"], disabled=not edit_mode)
    
    if edit_mode and st.button("Save & Reload"):
        config["embed_base_url"] = embed_base_url
        config["ollama_base_url"] = ollama_base_url
        config["default_kb"] = selected_kb
        config["default_model"] = model
        save_config(config)
        st.session_state.edit_urls = False
        st.cache_resource.clear()
        st.rerun()

indexer = get_indexer(db_path, model, config["embed_base_url"], config["ollama_base_url"])

# Search History in Sidebar
st.sidebar.markdown("---")

history = load_search_history()
with st.sidebar.expander(f"🕒 Search History ({len(history)} queries)", expanded=False):
    if history:
        from collections import defaultdict
        grouped = defaultdict(list)
        for entry in history:
            date = datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d')
            grouped[date].append(entry)
        
        for date in sorted(grouped.keys(), reverse=True):
            with st.expander(f"📅 {date} ({len(grouped[date])} queries)", expanded=False):
                for entry in grouped[date]:
                    time = datetime.fromisoformat(entry['timestamp']).strftime('%H:%M')
                    kb_info = entry.get('kb', 'unknown')
                    model_info = entry.get('model', 'unknown')
                    if st.button(f"{time}: {entry['query'][:30]}...", key=entry['timestamp']):
                        st.session_state.current_query = entry['query']
                        st.session_state.current_result = entry['result']
                        st.session_state.current_kb = kb_info
                        st.session_state.current_model = model_info
                        st.rerun()
    else:
        st.info("No search history yet")

# Initialize active tab in session state
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

# Main interface
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📁 Upload", "⚙️ Admin", "🤖 Models"])

with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("Search Documents")
    with col2:
        if st.button("New Search", type="secondary"):
            st.session_state.current_query = ""
            st.session_state.current_result = None
            st.rerun()
    
    query = st.text_input("Enter your query:", value=st.session_state.get('current_query', ''), placeholder="What are you looking for?")
    col1, col2 = st.columns(2)
    
    with col1:
        use_direct = st.checkbox("Direct vector search", help="Skip keyword search")
    
    if st.button("Search", type="primary") and query:
        with st.spinner("Searching..."):
            result, output = capture_output_live(indexer.query, query, use_direct)
            
            if output:
                st.text_area("Search Process:", output, height=200)
            
            if result:
                add_to_history(query, result, selected_kb, model)
                st.session_state.current_result = str(result)
                st.success("Search completed!")
                st.info(f"📚 Knowledge Base: **{selected_kb.title()}** | 🤖 Model: **{model}**")
                st.markdown("### Response")
                st.markdown(str(result))
    
    # Display cached result from history
    elif st.session_state.get('current_result'):
        st.success("Result from history")
        hist_kb = st.session_state.get('current_kb', 'unknown')
        hist_model = st.session_state.get('current_model', 'unknown')
        st.info(f"📚 Knowledge Base: **{hist_kb.title()}** | 🤖 Model: **{hist_model}**")
        st.markdown("### Response")
        st.markdown(st.session_state.current_result)
    


with tab2:
    st.header("Upload Documents")
    
    # Option selection
    option = st.radio("Choose input method:", ["Upload Files", "Select Local Folder"])
    
    if option == "Upload Files":
        uploaded_files = st.file_uploader(
            "Choose files to upload", 
            accept_multiple_files=True,
            type=['pdf', 'txt', 'md', 'docx', 'doc']
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} files")
            
            if st.button("Process Files", type="primary"):
                with st.spinner("Processing files..."):
                    temp_dir = tempfile.mkdtemp()
                    try:
                        for uploaded_file in uploaded_files:
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                        
                        _, output = capture_output_live(indexer.process_files, temp_dir)
                        
                        if output:
                            st.text_area("Processing Log:", output, height=200)
                        
                        st.success(f"Successfully processed {len(uploaded_files)} files!")
                        st.cache_resource.clear()
                        
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")
                    finally:
                        import shutil
                        shutil.rmtree(temp_dir, ignore_errors=True)
    
    else:  # Select Local Folder
        folder_path = st.text_input("Enter local folder path:", placeholder="/path/to/documents")
        
        if folder_path and os.path.exists(folder_path):
            file_count = sum(len(files) for _, _, files in os.walk(folder_path))
            st.write(f"Folder contains {file_count} files")
            
            if st.button("Process Folder", type="primary"):
                with st.spinner("Processing folder..."):
                    try:
                        _, output = capture_output_live(indexer.process_files, folder_path)
                        
                        if output:
                            st.text_area("Processing Log:", output, height=200)
                        
                        st.success(f"Successfully processed folder: {folder_path}")
                        st.cache_resource.clear()
                        
                    except Exception as e:
                        st.error(f"Error processing folder: {str(e)}")
        
        elif folder_path:
            st.error("Folder path does not exist")

with tab3:
    st.header("Database Administration")
    
    # Create new knowledge base section
    st.subheader("📚 Create New Knowledge Base")
    new_kb_name = st.text_input("Knowledge Base Name:", placeholder="e.g., legal, medical, research")
    
    if st.button("Create Knowledge Base", type="primary") and new_kb_name:
        new_db_path = f"{new_kb_name}_db"
        if os.path.exists(new_db_path):
            st.error(f"Knowledge base '{new_kb_name}' already exists!")
        else:
            try:
                # Create new indexer to initialize the database
                new_indexer = DocumentIndexer(target_dir="", db_path=new_db_path, model=model, embed_base_url=embed_base_url, ollama_base_url=ollama_base_url)
                new_indexer.load_or_create_index()
                st.success(f"Created knowledge base: {new_kb_name}")
                st.cache_resource.clear()
            except Exception as e:
                st.error(f"Error creating knowledge base: {str(e)}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Show Statistics"):
            doc_count = len(indexer.doc_collection.get()["ids"])
            
            st.metric("Documents", doc_count)
            st.info(f"Database: {os.path.abspath(db_path)}")
    
    with col2:
        if st.button("Delete Database", type="secondary"):
            if st.session_state.get('confirm_clear'):
                import shutil
                if os.path.exists(db_path):
                    shutil.rmtree(db_path)
                    st.success("Database deleted!")
                    st.cache_resource.clear()
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm deletion")
    
    if st.button("Show Status"):
        with st.spinner("Loading status..."):
            _, output = capture_output_live(indexer.display_status)
            if output:
                st.markdown(output)
            else:
                st.info("No status information found")
            

def get_ollama_library():
    return [
        "llama3.2:3b", "llama3.2:1b", "llama3.1:8b", "llama3.1:70b",
        "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "qwen2.5:72b",
        "deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:8b", "deepseek-r1:14b", "deepseek-r1:32b",
        "mistral:7b", "mixtral:8x7b", "codellama:7b", "codellama:13b",
        "phi3:3.8b", "phi3:14b", "gemma2:2b", "gemma2:9b", "gemma2:27b",
        "nomic-embed-text", "mxbai-embed-large", "all-minilm"
    ]

def pull_model_with_progress(model_name, ollama_url, progress_bar, status_text):
    try:
        response = requests.post(f"{ollama_url}/api/pull", json={"name": model_name}, stream=True, timeout=300)
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if 'status' in data:
                        status_text.text(data['status'])
                    if 'completed' in data and 'total' in data:
                        progress = data['completed'] / data['total']
                        progress_bar.progress(progress)
        return response
    except Exception as e:
        return None

def pull_model(model_name, ollama_url):
    try:
        response = requests.post(f"{ollama_url}/api/pull", json={"name": model_name}, stream=True, timeout=300)
        return response
    except Exception as e:
        return None

def delete_model(model_name, ollama_url):
    try:
        response = requests.delete(f"{ollama_url}/api/delete", json={"name": model_name}, timeout=30)
        return response
    except Exception as e:
        return None
with tab4:
    st.header("Ollama Models")
    
    # Global refresh button at top
    if st.button("🔄 Refresh Models", type="primary"):
        st.cache_resource.clear()
        st.rerun()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Available Models")
        library_models = get_ollama_library()
        installed_models = get_ollama_models(config["ollama_base_url"])
        
        for model in library_models:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                status = "✅ Installed" if model in installed_models else "⬇️ Available"
                st.write(f"{model} - {status}")
            with col_b:
                if model not in installed_models:
                    if st.button("Pull", key=f"pull_{model}"):
                        with st.spinner(f"Pulling {model}..."):
                            response = pull_model(model, config["ollama_base_url"])
                            if response and response.status_code == 200:
                                st.success(f"Successfully pulled {model}")
                                st.session_state.last_action = f"Pulled {model} - Check Models tab"
                            else:
                                st.error(f"Failed to pull {model}")
    
    with col2:
        st.subheader("📋 Installed Models")
        
        for model in installed_models:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.write(f"✅ {model}")
            with col_b:
                confirm_key = f"confirm_delete_{model}"
                if st.session_state.get(confirm_key):
                    col_c, col_d = st.columns(2)
                    with col_c:
                        if st.button("✓", key=f"confirm_{model}", type="primary", help="Confirm delete"):
                            response = delete_model(model, config["ollama_base_url"])
                            if response and response.status_code == 200:
                                st.success(f"Deleted {model}")
                                st.session_state.last_action = f"Deleted {model} - Check Models tab"
                            else:
                                st.error(f"Failed to delete {model}")
                            st.session_state[confirm_key] = False
                    with col_d:
                        if st.button("✗", key=f"cancel_{model}", help="Cancel delete"):
                            st.session_state[confirm_key] = False
                            st.rerun()
                else:
                    if st.button("Delete", key=f"delete_{model}", type="secondary"):
                        st.session_state[confirm_key] = True
                        st.rerun()
                        