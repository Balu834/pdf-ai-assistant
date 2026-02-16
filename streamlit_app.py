import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

st.set_page_config(page_title="PDF AI Assistant", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.title("📄 PDF AI Assistant")
st.sidebar.write("Upload a PDF and chat with it.")
st.sidebar.write("Built by Balu 🚀")

st.title("💬 Chat with your PDF")

# ---------------- Load LLM ----------------
@st.cache_resource
def load_llm():
    text_pipe = pipeline(
        task="text-generation",
        model="google/flan-t5-small",
        max_new_tokens=150,
        do_sample=False
    )
    return HuggingFacePipeline(pipeline=text_pipe)

llm = load_llm()

# ---------------- Upload PDF ----------------
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # ---------------- Process PDF ----------------
    @st.cache_resource
    def process_pdf():
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        docs = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(docs, embeddings)

        # Custom clean prompt
        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Answer the question using only the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            chain_type_kwargs={"prompt": custom_prompt}
        )

        return qa_chain

    with st.spinner("Processing PDF... ⏳"):
        qa = process_pdf()

    st.success("PDF processed successfully! ✅")

    # ---------------- Chat Memory ----------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show old messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask something about your PDF...")

    if user_input:
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking... 🤔"):
            response = qa.invoke({"query": user_input})
            answer = response["result"]

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.markdown(answer)
