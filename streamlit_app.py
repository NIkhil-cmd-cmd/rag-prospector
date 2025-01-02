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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your OpenAI API key from Streamlit secrets
OpenAI.api_key = st.secrets["OPEN_AI_KEY"]

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
            
    except Exception as e:
        logger.error(f"Error fetching file content: {e}")
        return ""

def get_drive_documents():
    """Fetch documents from Google Drive folder."""
    try:
        results = service.files().list(
            q=f"'{FOLDER_ID}' in parents",
            pageSize=100,
            fields="nextPageToken, files(id, name, mimeType)"
        ).execute()
        items = results.get('files', [])
        documents = []
        for item in items:
            content = get_file_content(service, item['id'], item['mimeType'])
            if content:
                documents.append(Document(text=content, metadata={'filename': item['name'], 'filetype': item['mimeType']}))
        return documents
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        return []

def create_index():
    """Create or load the document index."""
    try:
        documents = get_drive_documents()
        if not documents:
            logger.warning("No documents found in the Drive folder.")
            return None
        
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
        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex.from_documents(processed_docs, storage_context=storage_context)
        return index
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        return None

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

def process_spreadsheet(content, metadata):
    """Process spreadsheet content into chunks."""
    chunks = []
    current_sheet = ""
    
    for line in content.split('\n'):
        if line.startswith('Sheet:'):
            current_sheet = line.replace('Sheet:', '').strip()
        elif line.strip():
            # Combine sheet name, cell reference and content
            chunk = f"{current_sheet}: {line}"
            chunks.append(chunk)
    
    return "\n\n".join(chunks)

def render_drive_citation(citation):
    """Render a citation card with relevance-based styling for Drive search"""
    score = float(citation['score']) if citation.get('score') and citation['score'] != 'N/A' else 0
    hue = min(120, max(0, (score - 0.77) * 500))  # Maps 0.77-1.0 to 0-120 (red to green)
    bg_color = f"hsla({hue}, 70%, 30%, 0.2)"
    
    html = f"""
    <div class="doc-card" style="background: {bg_color}; border-radius: 12px; padding: 16px; margin: 12px 0; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);">
        <div class="doc-title" style="font-weight: bold; color: #fff;">{citation['filename']}</div>
        <div class="doc-relevance" style="color: #aaa;">Relevance Score: {citation.get('score', 'N/A')}</div>
        <div class="doc-snippet" style="color: #ddd;">{citation['snippet']}</div>
        <div class="doc-link">
            <a href="{citation['link']}" target="_blank" style="color: #4CAF50; text-decoration: none;">View Document →</a>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_web_citation(citation):
    """Render a citation card for web search results"""
    html = f"""
    <div class="doc-card" style="background: hsla(200, 70%, 30%, 0.2); border-radius: 12px; padding: 16px; margin: 12px 0; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);">
        <div class="doc-title" style="font-weight: bold; color: #fff;">{citation['filename']}</div>
        <div class="doc-snippet" style="color: #ddd;">{citation['snippet']}</div>
        <div class="doc-link">
            <a href="{citation['link']}" target="_blank" style="color: #4CAF50; text-decoration: none;">View Source →</a>
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
            color: #333;
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
            background-color: #121212;
            color: #ffffff;
        }
        [data-testid="stSidebar"] {
            background-color: #1e1e1e;
            border-right: 1px solid #333333;
        }
        .stTextInput > div > div > input {
            background-color: #2c2c2c !important;
            color: #ffffff !important;
            padding: 12px !important;
        }
        .stChatMessage {
            background-color: #2c2c2c !important;
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
    """Search Google for the given topic and return a summary and links with LLM-generated analysis."""
    if not is_journalistic_topic(topic):
        return "Please enter a valid journalistic topic for research.", "", "", []
    
    try:
        # Perform Google search
        search_results = list(search(topic, num_results=5))
        
        if not search_results:
            return "No results found for the given topic.", "", "", []
        
        # Use LLM to analyze the topic and generate pros/cons
        llm_prompt = f"""
        As a journalism expert, analyze this potential story topic: "{topic}"
        Provide:
        1. Two key reasons why this would make a compelling journalism story
        2. Two main challenges or considerations for covering this story
        Keep each point brief and focused on journalistic value.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": llm_prompt}],
            temperature=0.7
        )
        
        analysis = response.choices[0].message.content.strip()
        pros_cons = analysis.split('\n')
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
                logger.error(f"Error fetching web page: {e}")
                continue

        return "**Journalism Story Analysis:**", overall_pros, overall_cons, formatted_results
    except Exception as e:
        logger.error(f"Error during web search: {e}")
        return f"An error occurred while searching: {str(e)}", "", "", []

def is_journalistic_topic(topic):
    """Check if the topic is related to journalism."""
    # Simple check for common journalistic keywords
    journalistic_keywords = ["news", "report", "investigate", "journalism", "story", "article", "pitch", "analysis", "feature"]
    return any(keyword in topic.lower() for keyword in journalistic_keywords)

def generate_response(index, user_input):
    """Generate a response based on user input using the document index."""
    if not index:
        return {
            "answer": "No documents found in the drive to search through.",
            "primary_citations": [],
            "secondary_citations": []
        }

    # Pre-prompt for semantic search
    user_input = f"You are searching for a journalistic topic. Please provide relevant details. {user_input}"

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
        logger.error(f"Error generating response: {e}")
        return {
            "answer": f"An error occurred while generating the response: {str(e)}",
            "primary_citations": [],
            "secondary_citations": []
        }

def format_citation(doc):
    """Format a document citation."""
    return {
        'filename': doc.metadata.get('filename', 'Unknown Document'),
        'score': f"{doc.score:.2f}",
        'snippet': doc.text[:150] + '...' if len(doc.text) > 150 else doc.text,
        'link': doc.metadata.get('link', '#')
    }

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
        # Use a form to handle search submission via Enter key
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