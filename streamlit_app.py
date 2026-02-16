import streamlit as st
import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


# ------------------ PAGE CONFIG ------------------

st.set_page_config(page_title="PDF AI Assistant", layout="wide")

st.title("📄 PDF AI Assistant")
st.write("Upload a PDF and chat with it.")

st.sidebar.markdown("### Built by Balu 🚀")

# ------------------ OPENAI KEY ------------------

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("OpenAI API key not found. Please add it in Streamlit Secrets.")
    st.stop()

# ------------------ FILE UPLOAD ------------------

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully ✅")

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = OpenAIEmbeddings()

    # Vector DB
    db = FAISS.from_documents(docs, embeddings)

    # LLM
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo"
    )

    # QA Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )

    # ------------------ CHAT ------------------

    query = st.text_input("Ask something about your PDF")

    if query:
        with st.spinner("Thinking..."):
            result = qa.run(query)
        st.write("### Answer:")
        st.write(result)
