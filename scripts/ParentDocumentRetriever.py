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
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

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

docs = data + data2

child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

vectorstore = Chroma(
    collection_name="full_documents", embedding_function=embedding
)
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)
retriever.add_documents(docs, ids=None)

rag_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

def rag_qa(query):
    response = rag_chain.invoke(query)
    return response

query = "What is the termination notice?"
answer = rag_qa(query)
print(answer)