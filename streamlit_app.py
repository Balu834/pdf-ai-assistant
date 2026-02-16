import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="PDF AI Assistant", page_icon="📄")

st.sidebar.title("📄 PDF AI Assistant")
st.sidebar.write("Upload a PDF and chat with it.")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ------------------------
# Get OpenAI Key from Secrets
# ------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# ------------------------
# File Upload
# ------------------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    with st.spinner("Processing PDF..."):

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        docs = splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY
        )

        db = FAISS.from_documents(docs, embeddings)

        llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo",
            openai_api_key=OPENAI_API_KEY
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever()
        )

        st.session_state.qa_chain = qa

    st.success("PDF processed successfully! ✅")

# ------------------------
# Chat Interface
# ------------------------
if st.session_state.qa_chain:

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask something about your PDF..."):

        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.run(prompt)
                st.markdown(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
