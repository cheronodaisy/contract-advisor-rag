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
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

load_dotenv()

llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
        max_tokens=800,
        model_kwargs={"top_p": 0, "frequency_penalty": 0, "presence_penalty": 0},
    )
embedding = OpenAIEmbeddings()

# Path to the Chroma DB
chroma_db_path = "/home/daisy/Desktop/tenx/ContractAdvisorRAG/scripts/db1"
# Load Chroma DB vector store
vectorstore = Chroma(
    persist_directory=chroma_db_path,
    embedding_function=OpenAIEmbeddings()
)
# Initialize retriever from the vector store
retriever = vectorstore.as_retriever()

#compressor = LLMChainExtractor.from_llm(llm)
#compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

#embeddings_filter = EmbeddingsFilter(embeddings=embedding, similarity_threshold=0.85)
#compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator=". ")
redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding)
relevant_filter = EmbeddingsFilter(embeddings=embedding, similarity_threshold=0.76)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)

compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)

rag_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=compression_retriever)

def rag_qa(query):
    response = rag_chain.invoke(query)
    return response

query = "What are the payments to the Advisor under the Agreement?"
answer = rag_qa(query)
print(answer)