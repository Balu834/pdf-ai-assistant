import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
import tempfile
import os

st.set_page_config(page_title="PDF AI Assistant")
st.title("📄 PDF AI Assistant")
st.write("Upload a PDF and chat with it (100% Free Version)")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:

    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("PDF uploaded successfully ✅")

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)

    # Free Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(docs, embeddings)

    # FREE HuggingFace LLM
    pipe = pipeline(
        "text-generation",
        model="google/flan-t5-small",
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever()
    )

    query = st.text_input("Ask a question about your PDF")

    if query:
        result = qa.run(query)
        st.write("### Answer:")
        st.write(result)

    os.remove(tmp_path)
