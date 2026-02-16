import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
import tempfile

st.set_page_config(page_title="PDF AI Assistant", page_icon="📄", layout="wide")

# Sidebar
st.sidebar.title("📄 PDF AI Assistant")
st.sidebar.write("Upload a PDF and chat with it.")
st.sidebar.markdown("---")
if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.messages = []

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    with st.spinner("Processing PDF... ⏳"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = FAISS.from_documents(docs, embeddings)

        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=512,
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever()
        )

    st.success("PDF processed successfully! ✅")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if prompt := st.chat_input("Ask something about your PDF..."):

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤖"):
                response = qa.run(prompt)
                st.markdown(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
