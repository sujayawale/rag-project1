import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def create_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def load_vectorstore():
    embeddings=create_embeddings()

    vectorstore=FAISS.load_local(
        "vectorestore/company_faiss",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def create_retriever():
    vectorstore=load_vectorstore()
    retiever=vectorstore.as_retriever(
        search_kwargs={"k":3}
    )
    return retiever