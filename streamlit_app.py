import streamlit as st
import os
from document_indexer import DocumentIndexer
import io
import sys

st.set_page_config(page_title="LlamaIndex Document Search", layout="wide")

@st.cache_resource
def get_indexer(db_path, model):
    indexer = DocumentIndexer(target_dir="", db_path=db_path, model=model)
    indexer.load_or_create_index()
    return indexer

def capture_output(func, *args):
    """Capture print output from functions"""
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        result = func(*args)
        output = buffer.getvalue()
        return result, output
    finally:
        sys.stdout = old_stdout

st.title("🦙 LlamaIndex Document Search")

# Sidebar
st.sidebar.header("Configuration")
db_path = st.sidebar.text_input("Database Path", value="./chroma_db")
model = st.sidebar.selectbox("Model", ["gpt-oss:20b", "llama3.2:3b", "llama3.1:8b", "qwen2.5:7b","deepseek-r1:14b","deepseek-r1:8b"], index=0)
indexer = get_indexer(db_path, model)

# Main interface
tab1, tab2, tab3 = st.tabs(["🔍 Search", "📄 Process Document", "⚙️ Admin"])

with tab1:
    st.header("Search Documents")
    
    query = st.text_input("Enter your query:", placeholder="What are you looking for?")
    col1, col2 = st.columns(2)
    
    with col1:
        use_direct = st.checkbox("Direct vector search", help="Skip summary/keyword search")
    
    if st.button("Search", type="primary") and query:
        with st.spinner("Searching..."):
            result, output = capture_output(indexer.query, query, use_direct)
            
            if output:
                st.text_area("Search Process:", output, height=200)
            
            if result:
                st.success("Search completed!")
                st.markdown("### Response")
                st.markdown(str(result))

with tab2:
    st.header("Process Large Document")
    
    uploaded_file = st.file_uploader("Upload document", type=['pdf', 'docx', 'txt'])
    
    if uploaded_file:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing document..."):
                try:
                    summary = indexer.process_large_document(temp_path)
                    if summary:
                        st.success("Document processed successfully!")
                        st.markdown("### Summary")
                        st.markdown(summary)
                    else:
                        st.info("Document already processed")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    os.remove(temp_path)

with tab3:
    st.header("Database Administration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Show Statistics"):
            doc_count = len(indexer.doc_collection.get()["ids"])
            summary_count = len(indexer.summary_collection.get()["ids"])
            
            st.metric("Documents", doc_count)
            st.metric("Summaries", summary_count)
            st.info(f"Database: {os.path.abspath(db_path)}")
    
    with col2:
        if st.button("Clear Database", type="secondary"):
            if st.session_state.get('confirm_clear'):
                import shutil
                if os.path.exists(db_path):
                    shutil.rmtree(db_path)
                    st.success("Database cleared!")
                    st.cache_resource.clear()
                    st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm deletion")
    
    if st.button("Show All Summaries"):
        with st.spinner("Loading summaries..."):
            _, output = capture_output(indexer.display_all_summaries)
            if output:
                st.markdown(output)
            else:
                st.info("No summaries found")