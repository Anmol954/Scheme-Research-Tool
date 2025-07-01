import os
import pickle
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def create_faiss_store(docs, pickle_path="faiss_store_openai.pkl"):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    with open(pickle_path, "wb") as f:
        pickle.dump(db, f)

    print(f"FAISS store saved to {pickle_path}")

def load_faiss_store(pickle_path="faiss_store_openai.pkl"):
    with open(pickle_path, "rb") as f:
        db = pickle.load(f)
    return db
