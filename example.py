#!/usr/bin/env python3

# https://kleiber.me/blog/2024/08/04/demystifying-vector-stores-lanchain-vector-store-from-scratch/
# https://www.datacamp.com/tutorial/llama-3-1-rag

print("Downloading nltk data...")
import nltk

nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")

import os
import sys
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_turbopuffer import TurbopufferVectorStore
from langchain_community.document_loaders import WebBaseLoader

if sys.argv[1] == "--skip-load":
    vectorstore = TurbopufferVectorStore(
        embedding=OllamaEmbeddings(model="mxbai-embed-large"),
        namespace="langchain-turbopuffer-test",
        api_key=os.getenv("TURBOPUFFER_API_KEY"),
    )
else:
    print("Loading documents...")

    # List of URLs to load documents from
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    # Load documents from the URLs
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    print("Creating embeddings...")
    vectorstore = TurbopufferVectorStore.from_documents(
        documents=doc_splits,
        embedding=OllamaEmbeddings(model="mxbai-embed-large"),
        namespace="langchain-turbopuffer-test",
        api_key=os.getenv("TURBOPUFFER_API_KEY"),
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

print("Creating model...")

prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
    Use the following documents to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise:
    Question: {question}
    Documents: {documents}
    Answer:
    """,
    input_variables=["question", "documents"],
)

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

rag_chain = prompt | llm | StrOutputParser()


class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain

    def run(self, question):
        # Retrieve relevant documents
        print("retrieving facts from vector store...")
        documents = self.retriever.invoke(question)
        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        # Get the answer from the language model
        print("answering question...")
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer


rag_application = RAGApplication(retriever, rag_chain)

while True:
    question = input("Enter a question: ")
    answer = rag_application.run(question)
    print("Answer:", answer)
