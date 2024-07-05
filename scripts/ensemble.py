import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
#from langchain.retrievers.multi_query import MultiQueryRetriever
from datasets import Dataset
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

load_dotenv()

llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
        max_tokens=800,
        model_kwargs={"top_p": 0, "frequency_penalty": 0, "presence_penalty": 0},
    )
embedding = OpenAIEmbeddings()

loader = TextLoader("../data/Raptor.txt")
data = loader.load()
loader = TextLoader("../data/Robinson.txt")
data2 = loader.load()

data = data + data2

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(data)

#ensemble retriever
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 2

chroma_vectorstore = Chroma.from_documents(docs, embedding)
chroma_retriever = chroma_vectorstore.as_retriever()

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
)

# Set up the RAG chain
rag_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=ensemble_retriever)

def rag_qa(query):
    response = rag_chain.invoke(query)
    return response

query = "What is the termination notice?"
answer = rag_qa(query)
print(answer)
