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

#def rag_qa(query):
    #response = rag_chain.invoke(query)
    #return response

#query = "What is the termination notice?"
#answer = rag_qa(query)
#print(answer)

#RAGAS
questions = [
    "Who are the parties to the Agreement and what are their defined names?", 
    "What is the termination notice?",
    "In which street does the Advisor live?",
    "How much is the escrow amount?",
    "Does any of the Sellers provide a representation with respect to any Tax matters related to the Company?",
    "Whose consent is required for the assignment of the Agreement by the Buyer?",
]
ground_truths = [
    "Cloud Investments Ltd. (“Company”) and Jack Robinson (“Advisor”)",
    "According to section 4:14 days for convenience by both parties. The Company may terminate without notice if the Advisor refuses or cannot perform the Services or is in breach of any provision of this Agreement.",
    "1 Rabin st, Tel Aviv, Israel",
    "The escrow amount is equal to $1,000,000.",
    "No. Only the Company provides such a representation. ",
    "If the assignment is to an Affiliate or purchaser of all of the Buyer assets, no consent is required. Otherwise, the consent of the Company and the Seller Representative is required."
]

answers = []
contexts = []

# Inference
for query in questions:
    result = rag_chain.invoke(query)
    answer = result['result'] if 'result' in result else result  # Ensure result is in the expected format
    relevant_docs = ensemble_retriever.invoke(query)
    answers.append(answer)
    contexts.append([doc.page_content for doc in relevant_docs])

# To dict
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}
# Convert dict to dataset
dataset = Dataset.from_dict(data)

# Print dataset structure for debugging
#print(dataset[2])

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

result = evaluate(
    dataset = dataset, 
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

df = result.to_pandas()

#print(df.head())
heatmap_data = df[['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']]

cmap = LinearSegmentedColormap.from_list('green_red', ['red', 'green'])

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", linewidths=.5, cmap=cmap)

plt.yticks(ticks=range(len(df['question'])), labels=df['question'], rotation=0)

plt.show()
