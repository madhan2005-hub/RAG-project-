

import streamlit as st
from QueryProcessor import process_user_query
from pdfreader import read_pdf
from chunker import chunk_pages
from embedder import embed_chunks
from vectorstore import store_in_pinecone
import tempfile
import os

# Page configuration
st.set_page_config(page_title="HR Assistant Chatbot", page_icon="🤖", layout="wide")

# Custom CSS for chat UI
st.markdown("""
<style>
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #e8f4fd;
        border-left: 4px solid #2196F3;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4CAF50;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Sidebar for document upload
with st.sidebar:
    st.title("📄 Document Upload")
    st.markdown("Upload PDF documents to add to the knowledge base:")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], key="sidebar_uploader")
    
    if uploaded_file is not None:
        st.info(f"📎 File: {uploaded_file.name}")
        if st.button("Upload & Index", key="index_btn"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            with st.spinner("Processing and indexing document..."):
                try:
                    pages = read_pdf(tmp_path)
                    chunks = chunk_pages(pages)
                    embeddings = embed_chunks(chunks)
                    store_in_pinecone(chunks, embeddings)
                    st.success("✅ Document uploaded and indexed!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            os.remove(tmp_path)
    
    st.markdown("---")
    st.markdown("### 📊 Chat Statistics")
    st.write(f"Total messages: {len(st.session_state['chat_history'])}")
    
    if st.button("Clear Chat History"):
        st.session_state["chat_history"] = []
        st.rerun()

# Main chat area
st.title("🤖 HR Assistant Chatbot")
st.markdown("Ask any question and get instant answers from your documents!")

# Chat container
chat_container = st.container()

with chat_container:
    # Display chat history
    for i, (q, a) in enumerate(st.session_state.chat_history):
        # User message
        with st.container():
            col1, col2 = st.columns([1, 8])
            with col1:
                st.markdown("👤")
            with col2:
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong><br>{q}</div>', unsafe_allow_html=True)
        
        # Bot response
        with st.container():
            col1, col2 = st.columns([1, 8])
            with col1:
                st.markdown("🤖")
            with col2:
                st.markdown(f'<div class="chat-message bot-message"><strong>HR Assistant:</strong><br>{a}</div>', unsafe_allow_html=True)
        
        st.markdown("")  # Spacing

# Input area at bottom
st.markdown("---")
input_col1, input_col2 = st.columns([6, 1])

with input_col1:
    user_input = st.text_input("Type your question:", "", key="input", placeholder="Ask a question about your documents...")

with input_col2:
    send_btn = st.button("Send", type="primary")

# Process user input
if send_btn or (user_input and st.session_state.get("last_input") != user_input):
    if user_input.strip():
        with st.spinner("Thinking..."):
            import io
            import sys
            buffer = io.StringIO()
            sys.stdout = buffer
            answer = process_user_query(user_input)
            sys.stdout = sys.__stdout__
            answer = buffer.getvalue().strip()
            
            if not answer:
                answer = "I couldn't find a relevant answer in the uploaded documents. Please try a different question or upload more documents."
        
        st.session_state.chat_history.append((user_input, answer))
        st.session_state["last_input"] = user_input
        st.rerun()

# Hide Streamlit footer
st.markdown("<style>footer {visibility: hidden;}</style>", unsafe_allow_html=True)
