import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
#from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import clipboard

load_dotenv()

embedding_function = OpenAIEmbeddings()
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db3.as_retriever(), llm=llm
)

rag_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever_from_llm)

# Streamlit UI
st.title("Contract Q&A System")

if "history" not in st.session_state:
    st.session_state.history = []

if "copied" not in st.session_state:
    st.session_state.copied = []

def rag_qa(query):
    response = rag_chain.invoke(query)
    return response['result']

def on_copy_click(text):
    st.session_state.copied.append(text)
    clipboard.copy(text)

# Sidebar
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
        st.button("📋", on_click=on_copy_click, args=(answer,))
        st.toast(f"Copied to clipboard", icon='✅')

st.write("Disclaimer! Please verify any important information as the system may make mistakes.")
st.sidebar.write("Developed by Daisy Cherono")
