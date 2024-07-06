import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

pdf_text_1 = extract_text_from_pdf("../data/Raptor.pdf")
pdf_text_2 = extract_text_from_pdf("../data/robinson.pdf")

texts = [pdf_text_1, pdf_text_2]

chunk_size = 500
chunk_overlap = 50

text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

chunks = []
for text in texts:
    chunks.extend(text_splitter.split_text(text))

persist_directory = 'db1'
embeddings_file = os.path.join(persist_directory, 'embeddings.bin')

if not os.path.exists(embeddings_file):
    embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    documents = [Document(page_content=chunk) for chunk in chunks]
    vectordb = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=persist_directory)
else:
    vectordb = Chroma.load(persist_directory)

retriever = vectordb.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))

rag_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

def rag_qa(query):
    response = rag_chain.invoke(query)
    return response

query = " In which street does the Advisor live?"
answer = rag_qa(query)
print(answer)
