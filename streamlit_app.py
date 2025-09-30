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
def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            return models if models else ["deepseek-r1:14b"]
    except:
        pass
    return ["deepseek-r1:14b", "gpt-oss:20b", "llama3.2:3b", "llama3.1:8b", "qwen2.5:7b", "deepseek-r1:8b"]

@st.cache_resource
def get_indexer(db_path, model):
    indexer = DocumentIndexer(target_dir="", db_path=db_path, model=model)
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



st.title("🦙 LlamaIndex Document Search")

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
selected_kb = st.sidebar.radio(
    "Select Knowledge Base:",
    options=list(kb_options.keys()),
    format_func=lambda x: f"📁 {x.title()}",
    index=0
)

db_path = kb_options[selected_kb]
st.sidebar.caption(f"Path: {db_path}")
available_models = get_ollama_models()
default_idx = available_models.index("deepseek-r1:14b") if "deepseek-r1:14b" in available_models else 0
model = st.sidebar.selectbox("Model", available_models, index=default_idx)
indexer = get_indexer(db_path, model)

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

# Main interface
tab1, tab2, tab3 = st.tabs(["🔍 Search", "📁 Upload", "⚙️ Admin"])

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
                new_indexer = DocumentIndexer(target_dir="", db_path=new_db_path, model=model)
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
            _, output = capture_output(indexer.display_status)
            if output:
                st.markdown(output)
            else:
                st.info("No status information found")
            