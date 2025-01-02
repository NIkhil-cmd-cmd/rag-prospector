import streamlit as st
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.openai import OpenAIEmbedding
from googleapiclient.discovery import build
from google.oauth2 import service_account
import os
from io import BytesIO
import pandas as pd
from docx import Document as DocxDocument
from llama_index.core.storage import StorageContext
import pickle
import hashlib
import json
from pathlib import Path
import logging
import openai  # Import OpenAI SDK
from googlesearch import search
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPEN_AI_KEY"]

# Load service account credentials from Streamlit secrets
service_account_info = json.loads(st.secrets["gcp"]["service_account_json"])

# Create credentials using the service account info
creds = service_account.Credentials.from_service_account_info(service_account_info)

# Build the Google Drive service
service = build('drive', 'v3', credentials=creds)

FOLDER_ID = '1H6PgbGSvDlTvc-Zip3VW0XiHjedDKrR9'

@st.cache_resource
def get_drive_service():
    return build('drive', 'v3', credentials=creds, cache_discovery=False)

def get_file_content(service, file_id, mime_type):
    try:
        if mime_type == 'application/vnd.google-apps.document':
            content = service.files().export(
                fileId=file_id,
                mimeType='text/plain'
            ).execute()
            return content.decode('utf-8')
        
        elif mime_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            # Handle XLSX files
            content = service.files().get_media(fileId=file_id).execute()
            excel_file = BytesIO(content)
            
            # Read all sheets
            all_sheets = pd.read_excel(excel_file, sheet_name=None)
            formatted_content = []
            
            # Process each sheet
            for sheet_name, df in all_sheets.items():
                # Clean up the dataframe
                df = df.fillna('')  # Replace NaN with empty string
                
                # Add sheet name as context
                formatted_content.append(f"\nSheet: {sheet_name}\n")
                
                # Convert column names to string and make them lowercase for better searching
                df.columns = df.columns.astype(str).str.lower()
                
                # Process each row with column headers as context
                for idx, row in df.iterrows():
                    row_content = []
                    for col in df.columns:
                        if row[col] != '':  # Only include non-empty cells
                            row_content.append(f"{col}: {row[col]}")
                    formatted_content.append(" | ".join(row_content))
        
            return "\n".join(formatted_content)
            
        elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            # Handle DOCX files
            content = service.files().get_media(fileId=file_id).execute()
            doc_file = BytesIO(content)
            doc = DocxDocument(doc_file)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            
        elif mime_type == 'text/plain':
            content = service.files().get_media(fileId=file_id).execute()
            return content.decode('utf-8')
            
        elif mime_type == 'application/vnd.google-apps.folder':
            # Skip folders
            return None
            
        else:
            st.write(f"Unsupported file type: {mime_type}")
            return None
            
    except Exception as e:
        st.write(f"Error reading file {file_id}: {str(e)}")
        return None

def get_folder_contents(service, folder_id, documents=None):
    if documents is None:
        documents = []
    
    query = f"'{folder_id}' in parents"
    results = service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name, mimeType, webViewLink)',
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()
    
    files = results.get('files', [])
    
    for file in files:
        if file['mimeType'] == 'application/vnd.google-apps.folder':
            # Recursively get contents of subfolder
            get_folder_contents(service, file['id'], documents)
        else:
            content = get_file_content(service, file['id'], file['mimeType'])
            if content:
                doc = Document(
                    text=content,
                    metadata={
                        "filename": file['name'],
                        "link": file.get('webViewLink', ''),
                        "filetype": file['mimeType']
                    }
                )
                documents.append(doc)
    
    return documents

def get_files_hash():
    """Get a hash of all files and their last modified times"""
    service = get_drive_service()
    files = []
    
    def collect_files(folder_id):
        query = f"'{folder_id}' in parents"
        results = service.files().list(
            q=query,
            fields='files(id, name, mimeType, modifiedTime)',
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        
        for file in results.get('files', []):
            if file['mimeType'] == 'application/vnd.google-apps.folder':
                collect_files(file['id'])
            else:
                files.append({
                    'id': file['id'],
                    'modified': file['modifiedTime']
                })
    
    collect_files(FOLDER_ID)
    return hashlib.md5(json.dumps(files, sort_keys=True).encode()).hexdigest()

def save_index(index, doc_states):
    """Save index and document states to persistent storage"""
    # Create storage directory if it doesn't exist
    storage_dir = Path("storage")
    storage_dir.mkdir(exist_ok=True)
    
    storage_context = index._storage_context
    
    # Save all index components
    try:
        with open(storage_dir / 'docstore.pkl', 'wb') as f:
            pickle.dump(storage_context.docstore, f)
        with open(storage_dir / 'index_store.pkl', 'wb') as f:
            pickle.dump(storage_context.index_store, f)
        with open(storage_dir / 'vector_store.pkl', 'wb') as f:
            pickle.dump(storage_context.vector_store, f)
        with open(storage_dir / 'doc_states.json', 'w') as f:
            json.dump(doc_states, f)
            
        # Save current hash
        current_hash = get_files_hash()
        with open(storage_dir / 'files_hash.txt', 'w') as f:
            f.write(current_hash)
            
        return True
    except Exception as e:
        st.error(f"Error saving index: {str(e)}")
        return False

def load_index():
    """Load index from persistent storage"""
    storage_dir = Path("storage")
    
    try:
        # Verify hash hasn't changed
        current_hash = get_files_hash()
        hash_path = storage_dir / 'files_hash.txt'
        
        if hash_path.exists():
            stored_hash = hash_path.read_text().strip()
            if current_hash != stored_hash:
                logger.info("Hash mismatch. Rebuilding index.")
                return None
                
        # Load all components
        with open(storage_dir / 'docstore.pkl', 'rb') as f:
            docstore = pickle.load(f)
        with open(storage_dir / 'index_store.pkl', 'rb') as f:
            index_store = pickle.load(f)
        with open(storage_dir / 'vector_store.pkl', 'rb') as f:
            vector_store = pickle.load(f)
            
        # Create storage context
        storage_context = StorageContext.from_defaults(
            docstore=docstore,
            index_store=index_store,
            vector_store=vector_store
        )
        
        # Reconstruct index
        return VectorStoreIndex(
            nodes=list(docstore.docs.values()),
            storage_context=storage_context,
            show_progress=True
        )
        
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading index: {str(e)}")
        return None

@st.cache_resource
def create_index():
    """Create or load index with improved document processing"""
    try:
        # Try loading existing index first
        index = load_index()
        if index is not None:
            logger.info("Using existing index")
            return index
        
        # If no index exists, create new one
        logger.info("Creating new index...")
        documents = get_drive_documents()
        
        # Process documents with improved chunking
        processed_docs = []
        for doc in documents:
            chunks = get_document_chunks(doc)
            if isinstance(chunks, str):
                doc = Document(text=chunks, metadata=doc.metadata)
            else:
                doc = Document(text="\n\n".join(chunks), metadata=doc.metadata)
            processed_docs.append(doc)
        
        # Create new index
        index = VectorStoreIndex.from_documents(
            processed_docs,
            show_progress=True
        )
        
        # Save index and document states
        doc_states = get_document_states()
        if save_index(index, doc_states):
            logger.info("Index created and saved successfully")
        
        return index
        
    except Exception as e:
        st.error(f"Error creating index: {str(e)}")
        return None

def generate_response(index, user_input):
    if not index:
        return {
            "answer": "No documents found in the drive to search through.",
            "primary_citations": [],
            "secondary_citations": []
        }
    
    # Pre-prompt for semantic search
    pre_prompt = "You are searching for a journalistic topic. Please provide relevant details."
    user_input = f"{pre_prompt} {user_input}"

    retriever = VectorIndexRetriever(index=index, similarity_top_k=7)
    postprocessor = SimilarityPostprocessor(similarity_cutoff=0.77)
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever, node_postprocessors=[postprocessor], response_mode="tree_summarize"
    )

    try:
        response = query_engine.query(user_input)
        if not response or not response.source_nodes:
            return {
                "answer": "I couldn't find any relevant information based on your query.",
                "primary_citations": [],
                "secondary_citations": []
            }
        
        source_docs = sorted(response.source_nodes, key=lambda x: getattr(x, 'score', 0), reverse=True)
        primary_citations = [doc for doc in source_docs if getattr(doc, 'score', 0) > 0.85][:2]
        secondary_citations = [doc for doc in source_docs if 0.77 <= getattr(doc, 'score', 0) <= 0.85][:3]
        
        # Use the updated OpenAI call to generate a response
        openai_response = generate_openai_response(user_input)
        
        formatted_response = {
            "answer": openai_response,
            "primary_citations": [format_citation(doc) for doc in primary_citations],
            "secondary_citations": [format_citation(doc) for doc in secondary_citations]
        }
        return formatted_response

    except Exception as e:
        return {
            "answer": f"An error occurred while processing your request: {str(e)}",
            "primary_citations": [],
            "secondary_citations": []
        }

def format_citation(doc):
    metadata = doc.metadata
    score = getattr(doc, 'score', None)
    snippet = doc.node.text[:150] + "..." if len(doc.node.text) > 150 else doc.node.text
    return {
        "filename": metadata.get('filename', 'Untitled'),
        "score": f"{score:.2f}" if score else 'N/A',
        "snippet": snippet,
        "link": metadata.get('link', '#')
    }

def get_drive_documents():
    service = get_drive_service()
    logger.info("Fetching files from Google Drive...")
    documents = get_folder_contents(service, FOLDER_ID)
    logger.info(f"Successfully created {len(documents)} document objects")
    return documents

def get_document_states():
    """Get current state of all documents in Drive"""
    service = get_drive_service()
    doc_states = {}
    
    def collect_file_states(folder_id):
        query = f"'{folder_id}' in parents"
        results = service.files().list(
            q=query,
            fields='files(id, name, mimeType, modifiedTime)',
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        
        for file in results.get('files', []):
            if file['mimeType'] == 'application/vnd.google-apps.folder':
                collect_file_states(file['id'])
            else:
                doc_states[file['id']] = {
                    'name': file['name'],
                    'modified': file['modifiedTime']
                }
    
    collect_file_states(FOLDER_ID)
    return doc_states

def process_spreadsheet(content, metadata):
    """Process spreadsheet content to maintain cell context"""
    chunks = []
    current_sheet = ""
    
    for line in content.split('\n'):
        if line.startswith("Sheet:"):
            current_sheet = line
            chunks.append(current_sheet)
        else:
            chunks.append(f"{current_sheet} {line}")
    
    return chunks

def get_document_chunks(doc):
    """Chunk documents based on type and maintain context"""
    if doc.metadata.get('filetype', '').endswith('.xlsx'):
        return process_spreadsheet(doc.text, doc.metadata)
    else:
        # For other documents, use sentence splitter with larger chunks
        splitter = SentenceSplitter(
            chunk_size=1024,  # Larger chunks for better context
            chunk_overlap=200  # More overlap to maintain context
        )
        return splitter.split_text(doc.text)

def render_citation(citation):
    """Render a citation card with relevance-based styling"""
    score = citation.get('score', 'N/A')
    # Set default background for N/A or calculate based on score
    if score == 'N/A':
        bg_color = "hsla(200, 70%, 30%, 0.2)"  # Default blue-ish background
    else:
        score = float(score)
        hue = min(120, max(0, (score - 0.77) * 500))
        bg_color = f"hsla({hue}, 70%, 30%, 0.2)"
    
    html = f"""
    <div class="doc-card" style="background: {bg_color}; border-radius: 12px; padding: 16px; margin: 12px 0; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);">
        <div class="doc-title" style="font-weight: bold; color: #fff;">{citation['filename']}</div>
        <div class="doc-relevance" style="color: #aaa;">Relevance Score: {citation['score']}</div>
        <div class="doc-snippet" style="color: #ddd;">{citation['snippet']}</div>
        <div class="doc-link">
            <a href="{citation['link']}" target="_blank" style="color: #4CAF50; text-decoration: none;">View Document →</a>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def create_logging_sidebar():
    container = st.sidebar.container()
    container.markdown("""
        <style>
        .status-item {
            display: flex;
            align-items: center;
            margin: 8px 0;
            padding: 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 6px;
        }
        .status-icon {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 12px;
        }
        .status-success { background: #4CAF50; }
        .status-error { background: #F44336; }
        .status-loading { background: #FFC107; }
        .status-text { color: #fff; }
        </style>
    """, unsafe_allow_html=True)
    
    return container

def create_header():
    """Create modern header with logo"""
    capybara_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="24" height="24" style="fill: currentColor;">
        <path d="M374 74.47c-7.1.26-10.8 6.79-4.3 15.89l24-3.41c-6.5-9.11-14.1-12.69-19.7-12.48zm-38 9.1c-3.5 0-6.6 1.01-9 2.73c-7.1 5.1-7.6 16.8 7.9 28c-8.9 15.9-29.8 45.8-60.2 43.2l32.1 9.8c-2.7 1.6-5.7 3.1-9.2 4.5C118.7 119.4 29.29 275.1 29.29 275.1c51.1 69.9 4.1 98.9 4.1 98.9l7.81 63h28.81l3.19-41s32.5-3 62.8-63.3c29 9.8 71 9.1 102.6 3.3l-4.1 7.1l-37.4 11.1c31.2 2.8 58.5-2.3 78.7-8.5c-3.4-15.1-4.5-31.5 3.5-52.8L307.2 437h25.9s-4.6-75 34.4-143.5c5-7.8 9.4-15.1 13.1-23.7l2 11.1l-10.5 23.2s39-15.7 29.2-96c23 3.9 45.6 1.7 66.6-4.6c5.3-1.7 9.5-5.8 11.2-11c5-15.6 9.5-32.5 10.4-47.3l-9.7.8c-.2-15.3-21.2-13.1-14.9.8l-10.5.5l-4.9-15.5s16.9-12.3 38.4-7.1c-.9-3.2-2.2-6-3.9-8.6c-13.8-20.8-54.3-27.8-122.4-15.6c-8-12.24-17.8-16.96-25.6-16.93zm49.9 33.83c12.4 1.4 21.9 4.3 30.2 9.6h-15.9c-1.6 4.8-7.5 8.4-14.5 8.4s-12.9-3.6-14.5-8.4h-15.5c4.2-3 15.3-9.7 30.2-9.6zm9.6 181.6c-15.2 30.3-34.5 33.8-34.5 33.8c-13.4 37.7-10.4 71.8 1.8 103.9H385c-3.8-44.7-3.2-78.4 10.5-137.7zm-251.1 50.3L126.6 376l27.2 25.1l13.9 35.6h29.9l-20.1-81.8z"/>
    </svg>
    """
    
    st.markdown(f"""
        <style>
        .header-container {{
            display: flex;
            align-items: center;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .logo {{
            display: flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            margin-right: 16px;
            color: #ffffff;
        }}
        .title {{
            font-size: 24px;
            font-weight: 500;
            color: #ffffff;
            margin: 0;
        }}
        .subtitle {{
            font-size: 14px;
            color: #888;
            margin: 4px 0 0 0;
        }}
        </style>
        <div class="header-container">
            <div class="logo">
                {capybara_svg}
            </div>
            <div>
                <h1 class="title">Prospector</h1>
                <p class="subtitle">An AI Assistant for Our Drive Folder</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

def format_response(response_dict):
    """Format the chatbot response in a modern style"""
    citations_html = '\n'.join([
        f"""
        <div class="citation-card">
            <div class="citation-title">{citation['filename']}</div>
            <div class="citation-score">Relevance: {citation['score']}</div>
            <div class="citation-snippet">{citation['snippet']}</div>
            <a href="{citation['link']}" class="citation-link" target="_blank">View Document →</a>
        </div>
        """
        for citation in response_dict.get("primary_citations", []) + response_dict.get("secondary_citations", [])
    ])
    
    st.markdown("""
        <style>
        .response-container {
            font-family: 'Inter', sans-serif;
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .answer {
            font-size: 16px;
            line-height: 1.6;
            color: #333;
            margin-bottom: 20px;
        }
        .citations {
            border-top: 1px solid #eee;
            padding-top: 16px;
        }
        .citation-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
        }
        .citation-title {
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 4px;
        }
        .citation-score {
            font-size: 12px;
            color: #666;
            margin-bottom: 8px;
        }
        .citation-snippet {
            font-size: 14px;
            color: #444;
            margin-bottom: 8px;
        }
        .citation-link {
            font-size: 12px;
            color: #007bff;
        }
        </style>
        <div class="response-container">
            <div class="answer">{}</div>
            <div class="citations">
                <h3>Sources</h3>
                {}
            </div>
        </div>
    """.format(
        response_dict['answer'],
        citations_html
    ), unsafe_allow_html=True)

def create_chat_interface():
    st.markdown("""
        <style>
        .chat-container {font-family: 'Inter', sans-serif; max-width: 800px; margin: 0 auto;}
        .stTextInput > div > div > input {font-family: 'Inter', sans-serif; border-radius: 8px !important;}
        .stChatMessage {background: rgba(255, 255, 255, 0.05) !important; border-radius: 8px; margin: 8px 0;}
        .stChatInput {border-color: rgba(255, 255, 255, 0.1) !important; background: rgba(255, 255, 255, 0.05) !important;}
        </style>
    """, unsafe_allow_html=True)

def set_dark_theme():
    st.markdown("""
        <style>
        /* Dark theme base styles */
        [data-testid="stAppViewContainer"] {
            background-color: #000000;
            color: #ffffff;
        }
        [data-testid="stSidebar"] {
            background-color: #0a0a0a;
            border-right: 1px solid #1a1a1a;
        }
        .stTextInput > div > div > input {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
            padding: 12px !important;
        }
        .stChatMessage {
            background-color: #1a1a1a !important;
            border: none !important;
            padding: 16px !important;
            margin: 16px 0 !important;
        }
        /* Streamlit icon modifications */
        [data-testid="stIcon"] {
            opacity: 0.7;
            transition: opacity 0.3s;
        }
        [data-testid="stIcon"]:hover {
            opacity: 1;
        }
        /* Animation for sidebar elements */
        .sidebar-container {
            animation: slideIn 0.5s ease-out;
        }
        @keyframes slideIn {
            from { transform: translateX(-100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        </style>
    """, unsafe_allow_html=True)

def search_web(topic):
    try:
        # Perform Google search
        search_results = list(search(topic, num_results=5))
        
        # Generate analysis using LLM
        llm_prompt = f"""
        Analyze this topic as a potential news story: "{topic}"
        Provide a brief analysis of its journalistic value, including key strengths and challenges.
        Keep the analysis concise and professional.
        """
        
        analysis = generate_openai_response(llm_prompt)
        
        # Format search results
        raw_results = []
        for result in search_results:
            try:
                response = requests.get(result, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                title = soup.title.string.strip() if soup.title and soup.title.string else "No Title"
                summary = ' '.join(soup.get_text().split())[:200] + "..."
                raw_results.append({
                    "title": title,
                    "link": result,
                    "summary": summary
                })
            except Exception:
                continue

        # Clean up results with LLM
        results_prompt = f"""
        Clean up and summarize these search results about "{topic}":
        {raw_results}
        
        For each result, provide:
        1. A clear, concise title
        2. A 1-2 sentence summary of the key points
        Keep the language professional and journalistic.
        """
        
        cleaned_results = generate_openai_response(results_prompt)
        
        return analysis, cleaned_results
    except Exception as e:
        return f"Error: {str(e)}", []

def generate_openai_response(prompt):
    try:
        client = openai.OpenAI(api_key=st.secrets["OPEN_AI_KEY"])  # Initialize with API key
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred while generating a response: {str(e)}"

def main():
    # Set up Streamlit page configuration
    st.set_page_config(
        page_title="Prospector",
        page_icon="🦫",
        layout="wide"
    )
    
    # Apply dark theme
    set_dark_theme()
    
    # Initialize sidebar with logging and data source selection
    sidebar_container = create_logging_sidebar()
    
    # Add data source selection to sidebar
    if 'data_source' not in st.session_state:
        st.session_state.data_source = 'Drive'
    
    data_source = st.sidebar.selectbox("Select Data Source", ['Drive', 'Google'], key='data_source_select')
    
    # Reset chat when data source changes
    if 'prev_data_source' not in st.session_state:
        st.session_state.prev_data_source = data_source

    if data_source != st.session_state.prev_data_source:
        st.session_state.messages = []
        st.session_state.prev_data_source = data_source
        st.rerun()
    
    # Create header
    create_header()
    
    # Create chat interface styles
    create_chat_interface()
    
    try:
        # Load or create the document index with a spinner
        with st.spinner("Loading document index..."):
            if data_source == 'Drive':
                index = create_index()
                if index:
                    sidebar_container.markdown('<div class="status-item"><div class="status-icon status-success"></div><div class="status-text">✓ Index Created Successfully</div></div>', unsafe_allow_html=True)
                else:
                    sidebar_container.markdown('<div class="status-item"><div class="status-icon status-error"></div><div class="status-text">❌ Failed to create or load index.</div></div>', unsafe_allow_html=True)
            elif data_source == 'Google':
                index = None
                sidebar_container.markdown('<div class="status-item"><div class="status-icon status-success"></div><div class="status-text">🔍 Google Search Ready</div></div>', unsafe_allow_html=True)
                
                if prompt := st.chat_input("Enter a topic to research", key="google_search"):
                    with st.spinner("Searching and analyzing..."):
                        analysis, search_results = search_web(prompt)
                        
                        # Display analysis and results
                        st.write(analysis)
                        st.write("\nRelated Articles:")
                        st.write(search_results)
    except Exception as e:
        st.error(f"Error creating index: {str(e)}")
        return

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": {
                "answer": "Hello! I can help you find information in your Google Drive documents. Ask me anything about their contents!",
                "primary_citations": [],
                "secondary_citations": []
            }
        })

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], dict):
                st.markdown(message["content"]["answer"])
                if message["content"].get("primary_citations") or message["content"].get("secondary_citations"):
                    st.markdown("### Sources")
                    for citation in message["content"].get("primary_citations", []):
                        render_citation(citation)
                    for citation in message["content"].get("secondary_citations", []):
                        render_citation(citation)
            else:
                st.markdown(message["content"])

    # Handle user input based on data source
    if data_source == 'Drive':
        if prompt := st.chat_input("Ask me about your documents", key="drive_chat"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            try:
                with st.spinner("Thinking..."):
                    response = generate_response(index, prompt)
                
                if not response.get("answer"):
                    response["answer"] = "I'm sorry, I couldn't find an answer to that based on the available documents."
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
                with st.chat_message("assistant"):
                    st.markdown(response["answer"])
                    if response.get("primary_citations") or response.get("secondary_citations"):
                        st.markdown("### Sources")
                        for citation in response.get("primary_citations", []):
                            render_citation(citation)
                        for citation in response.get("secondary_citations", []):
                            render_citation(citation)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    elif data_source == 'Google':
        if prompt := st.chat_input("Enter a topic to research", key="google_search"):
            with st.spinner("Searching and analyzing..."):
                analysis, search_results = search_web(prompt)
                
                # Display analysis and results
                st.write(analysis)
                st.write("\nRelated Articles:")
                st.write(search_results)

if __name__ == "__main__":
    main()