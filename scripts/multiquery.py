import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_openai import ChatOpenAI
#from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

# Path to the Chroma DB
chroma_db_path = "/home/daisy/Desktop/tenx/ContractAdvisorRAG/scripts/db1"

# Load Chroma DB vector store
vectorstore = Chroma(
    persist_directory=chroma_db_path,
    embedding_function=OpenAIEmbeddings()
)

# Initialize retriever from the vector store
#retriever = vectorstore.as_retriever()

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), llm=llm
)

# Set up the RAG chain
rag_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

def rag_qa(query):
    response = rag_chain.invoke(query)
    return response

query = "Whose consent is required for the assignment of the Agreement by the Buyer?"
answer = rag_qa(query)
print(answer)