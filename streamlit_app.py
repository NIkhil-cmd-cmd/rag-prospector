import streamlit as st
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI
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
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import openai

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
    pre_prompt = "Find topics related to the following:"
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
        
        formatted_response = {
            "answer": str(response),
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
        splitter = SentenceSplitter()
        return splitter.split(doc.text, max_length=500)

def search_web(topic):
    """Search Google for the given topic and return a summary and links with LLM-generated analysis."""
    try:
        # Perform Google search
        search_results = list(search(topic, num_results=5))
        
        # Use LLM to analyze the topic and generate pros/cons
        llm_prompt = f"""
        As a journalism expert, analyze this potential story topic: "{topic}"
        Provide:
        1. Two key reasons why this would make a compelling journalism story
        2. Two main challenges or considerations for covering this story
        Keep each point brief and focused on journalistic value.
        """
        
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=llm_prompt,
            max_tokens=150,
            temperature=0.7
        )
        
        analysis = response.choices[0].text.strip()
        pros_cons = analysis.split("\n")
        overall_pros = "\n".join([p for p in pros_cons if any(word in p.lower() for word in ["compelling", "good", "advantage", "reason"])])
        overall_cons = "\n".join([c for c in pros_cons if any(word in c.lower() for word in ["challenge", "consideration", "limitation"])])
        
        formatted_results = []
        for result in search_results:
            try:
                response = requests.get(result, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                title = soup.title.string.strip() if soup.title and soup.title.string else "No Title"
                summary = ' '.join(soup.get_text().split())[:150] + "..."  # Shorter summary
                
                formatted_results.append({
                    "filename": title,
                    "link": result,
                    "snippet": summary
                })
            except Exception as e:
                continue

        return "**Journalism Story Analysis:**", overall_pros, overall_cons, formatted_results
    except Exception as e:
        return f"An error occurred while searching: {str(e)}", "", "", []

def main():
    # Set up Streamlit page configuration
    st.set_page_config(
        page_title="Prospector",
        page_icon="🦫",
        layout="wide"
    )
    
    # Apply dark theme
    set_dark_theme()
    
    # Initialize sidebar with logging and search type
    sidebar_container = create_logging_sidebar()
    
    # Create header
    create_header()
    
    # Create chat interface styles
    create_chat_interface()
    
    # Dropdown for selecting search type in the sidebar
    search_type = sidebar_container.selectbox("Select Search Type", ["Search Drive for Past Pitches", "Search Web for Topic"])
    
    if search_type == "Search Web for Topic":
        with st.form(key='web_search_form'):
            topic = st.text_input("Enter the topic you want to research:", key="search_topic")
            submitted = st.form_submit_button("Search")
        
        if submitted:
            if topic:
                overall_summary, overall_pros, overall_cons, results = search_web(topic)
                if isinstance(overall_summary, str) and overall_cons == "" and overall_pros == "" and not results:
                    st.error(overall_summary)
                else:
                    st.markdown(overall_summary)
                    st.markdown("### Story Potential")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**✔️ Advantages**")
                        st.markdown(overall_pros)
                    with col2:
                        st.markdown("**❌ Challenges**")
                        st.markdown(overall_cons)
                    
                    st.markdown("### Related Sources")
                    for result in results:
                        render_web_citation({
                            'filename': result['filename'],
                            'snippet': result['snippet'],
                            'link': result['link']
                        })
            else:
                st.error("Please enter a topic to search.")
    
    else:
        # Existing functionality to search the drive
        try:
            # Load or create the document index with a spinner
            with st.spinner("Loading document index..."):
                index = create_index()
                if index:
                    sidebar_container.markdown('<div class="status-item"><div class="status-icon status-success"></div><div class="status-text">✓ Index Loaded Successfully</div></div>', unsafe_allow_html=True)
                else:
                    sidebar_container.markdown('<div class="status-item"><div class="status-icon status-error"></div><div class="status-text">❌ Failed to load index.</div></div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading index: {str(e)}")
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
                            render_drive_citation(citation)
                        for citation in message["content"].get("secondary_citations", []):
                            render_drive_citation(citation)
                else:
                    st.markdown(message["content"])

        # Handle user input with Enter key using form
        with st.form(key='drive_chat_form', clear_on_submit=True):
            prompt = st.text_input("Ask me about your documents")
            submit_button = st.form_submit_button(label="Send")
        
        if submit_button and prompt:
            # Append user message to session state
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            try:
                # Generate assistant response with a spinner
                with st.spinner("Thinking..."):
                    response = generate_response(index, prompt)
            
                # Ensure response is not empty
                if not response.get("answer"):
                    response["answer"] = "I'm sorry, I couldn't find an answer to that based on the available documents."
            
                # Append assistant response to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
            
                # Display assistant response
                with st.chat_message("assistant"):
                    format_response(response)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
