import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
import pyperclip

load_dotenv()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

# Path to the Chroma DB
chroma_db_path = "/home/daisy/Desktop/tenx/ContractAdvisorRAG/scripts/db1"

# Load Chroma DB vector store
vectorstore = Chroma(
    persist_directory=chroma_db_path,
    embedding_function=OpenAIEmbeddings()
)

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), llm=llm
)

# Set up the RAG chain
rag_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

# Streamlit UI
st.title("Contract Q&A System")

if "history" not in st.session_state:
    st.session_state.history = []

def rag_qa(query):
    response = rag_chain.invoke(query)
    return response['result']

with st.sidebar:
    st.header("Chat History")
    search_query = st.text_input("Search chat history")
    if st.button("Search"):
        search_results = [(i, q, a) for i, (q, a) in enumerate(st.session_state.history) if search_query.lower() in q.lower() or search_query.lower() in a.lower()]
        for i, query, answer in search_results:
            st.write(f"Q{i+1}: {query}")
            st.write(f"A{i+1}: {answer}")
            st.write("---")
    for i, (query, answer) in enumerate(st.session_state.history):
        st.write(f"Q{i+1}: {query}")
        st.write(f"A{i+1}: {answer}")
        st.write("---")

query = st.text_input("Enter your query about the contract")
if st.button("Submit"):
    if query:
        answer = rag_qa(query)
        st.session_state.history.append((query, answer))
        st.write("### Answer")
        st.write(answer)
        if st.button("📋"):
            st.experimental_set_query_params(answer=answer)
            st.success("Answer copied to clipboard!")

st.write("Disclaimer: Please verify any important information as the system may make mistakes.")

st.sidebar.write("Developed by D.C")
