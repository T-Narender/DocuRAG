import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.exceptions import ModelRateLimitError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Document Knowledge Assistant", page_icon="📂", layout="centered")

# Load the local fallback before reading GEMINI_API_KEY.
load_dotenv(Path(__file__).resolve().parent / ".env")
FALLBACK_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_DEFAULT_FALLBACK_API_KEY")
VECTORSTORE_DIR = Path("chroma_db_upload")
SUPPORTED_TYPES = ["pdf", "txt", "docx", "doc", "pptx", "ppt"]


def _load_documents(file_path: str, suffix: str):
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".ppt": UnstructuredPowerPointLoader,
    }
    loader_class = loaders.get(suffix.lower())
    if loader_class is None:
        raise ValueError(f"Unsupported file type: {suffix}")
    return loader_class(file_path).load()


def process_and_index_file(file_bytes: bytes, file_name: str):
    """Processes, chunks, and indexes a file cleanly into a fresh vector directory."""
    suffix = Path(file_name).suffix.lower()
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_file.write(file_bytes)
            temporary_path = temporary_file.name

        raw_documents = _load_documents(temporary_path, suffix)
        
        # Inject our optimized 500/100 character window layout
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(raw_documents)
        if not chunks:
            raise ValueError("The uploaded file did not contain extractable text.")

        # Force a hard clear on the directory disk block to keep data isolated
        if VECTORSTORE_DIR.exists():
            try:
                shutil.rmtree(VECTORSTORE_DIR)
            except PermissionError:
                # Fallback path modification if Windows locks the collection folder
                pass
                
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(VECTORSTORE_DIR),
        )
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)


def format_docs(documents):
    return "\n\n".join(document.page_content for document in documents)


def retry_on_rate_limit(rag_chain, question, max_retries=2, wait_seconds=35):
    """Invoke the chain, auto-waiting through temporary Gemini free-tier rate limits."""
    try:
        return rag_chain.invoke(question)
    except ModelRateLimitError:
        for attempt in range(1, max_retries + 1):
            st.warning(f"⏳ Gemini is rate-limited. Waiting {wait_seconds}s before retrying ({attempt}/{max_retries})...")
            time.sleep(wait_seconds)
            try:
                return rag_chain.invoke(question)
            except ModelRateLimitError:
                continue
        raise


def response_instructions(depth: str) -> str:
    instructions = {
        "Short": "Answer with at most 2-3 precise bullet points or sentences.",
        "Medium": "Answer with a clear standard overview paragraph and only the key supporting details.",
        "Detailed": "Give a structured breakdown with relevant context and comprehensive details, derived strictly from the retrieved chunks.",
    }
    return instructions[depth]


# --- SIDEBAR UI NAVIGATION CONFIGURATION ---
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader(
        "Upload a policy document",
        type=SUPPORTED_TYPES,
        help="Supported formats: PDF, TXT, DOCX, DOC, PPTX, and PPT.",
    )
    custom_api_key = st.text_input("Gemini API key", type="password")
    response_depth = st.selectbox(
        "Select desired response depth",
        options=["Short", "Medium", "Detailed"],
        index=1,
    )
    
    # Establish cascading fallback architecture for security keys
    api_key = custom_api_key.strip() or FALLBACK_API_KEY
    if custom_api_key.strip():
        st.caption("🟢 Using custom key for optimal performance")
    else:
        st.caption("🔵 Using shared system fallback key")

if api_key == "YOUR_DEFAULT_FALLBACK_API_KEY":
    st.warning(
        "Enter a valid Gemini API key in the sidebar, or set GEMINI_API_KEY before starting Streamlit."
    )

st.title("DocuRAG — Local PDF Intelligence Assistant")
st.write("Upload any supported file format to extract insights and ask questions about its contents.")

# Initialize conversation history persistence layers
if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file is None:
    st.info("👋 Upload a PDF, text, Word, or PowerPoint file in the sidebar to begin.")
    st.stop()

# --- FILE SWITCH DETECTION PIPELINE ---
file_bytes = uploaded_file.getvalue()
file_signature = hashlib.sha256(file_bytes).hexdigest()

# If the document changes, drop the active retriever and history arrays immediately
if st.session_state.get("file_signature") != file_signature:
    st.session_state.file_signature = file_signature
    st.session_state.messages = []
    if "retriever" in st.session_state:
        del st.session_state["retriever"]

# Build or reinitialize system indices securely outside volatile global caching loops
if "retriever" not in st.session_state:
    try:
        with st.spinner("Extracting, embedding, and indexing the uploaded document..."):
            st.session_state.retriever = process_and_index_file(file_bytes, uploaded_file.name)
    except Exception as error:
        st.error(f"Could not process the uploaded file: {error}")
        st.stop()

# --- INITIALIZE CHAT ENGINE COMPONENTS ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0,
)

prompt = ChatPromptTemplate.from_template(
    """You are a helpful and precise document analysis assistant. 
Use the provided retrieved context and the conversation history to answer the user's question accurately.

Conversation History:
{chat_history}

Retrieved context:
{context}

Question: {question}
Answer:"""
)

def format_chat_history(messages):
    formatted = []
    # Only send the last 4-6 messages so we don't overload the context window
    for msg in messages[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Strip out the source citation metadata string from the assistant's history so it stays clean
        content = msg["content"].split("\n\n📌")[0] if role == "Assistant" else msg["content"]
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


chat_history_text = format_chat_history(st.session_state.get("messages", []))

rag_chain = (
    {
        "context": st.session_state.retriever | format_docs,
        "chat_history": lambda _: chat_history_text,
        "question": RunnablePassthrough(),
        "depth_instruction": lambda _: response_instructions(response_depth),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Render historical context arrays
for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- EXECUTE USER QUERY ACTIONS ---
if user_question := st.chat_input("Ask a question about the uploaded document..."):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching document context..."):
            source_documents = st.session_state.retriever.invoke(user_question)
            try:
                answer = retry_on_rate_limit(rag_chain, user_question)
            except ModelRateLimitError as error:
                st.error(
                    "Your Gemini API key has hit its rate limit (free-tier quota). "
                    "Wait a minute and retry, or switch to a paid/billable key in the sidebar or GEMINI_API_KEY. "
                    f"({type(error).__name__})"
                )
                st.session_state.messages.pop()
                st.stop()
            except Exception as error:
                st.error(
                    f"Request failed ({type(error).__name__}: {error}). "
                    "Check that your API key is valid and enabled for Gemini."
                )
                st.session_state.messages.pop()
                st.stop()
            
            # Map dynamic source files or pages matching citation lists
            pages = sorted(list(set([doc.metadata.get('page', 0) + 1 for doc in source_documents])))
            page_string = ", ".join(f"Page {p}" for p in pages)
            
            full_response = f"{answer}\n\n📌 *Sources: {uploaded_file.name} ({page_string})*"
            st.markdown(full_response)
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})
