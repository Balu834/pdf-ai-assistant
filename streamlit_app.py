import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
import tempfile

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📄",
    layout="wide"
)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📄 PDF AI Assistant")
st.sidebar.write("Upload a PDF and chat with it.")
st.sidebar.markdown("---")

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.messages = []

# -------------------------------
# Session State
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    with st.spinner("Processing PDF... ⏳"):

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector Store
        db = FAISS.from_documents(docs, embeddings)

        # Lightweight Model (Cloud Safe)
        pipe = pipeline(
            task="text2text-generation",
            model="google/flan-t5-small",
            max_new_tokens=256
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        # QA Chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever()
        )

        st.session_state.qa_chain = qa

    st.success("PDF processed successfully! ✅")

# -------------------------------
# Chat Interface
# -------------------------------
if st.session_state.qa_chain:

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask something about your PDF..."):

        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤖"):
                response = st.session_state.qa_chain.run(prompt)
                st.markdown(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
