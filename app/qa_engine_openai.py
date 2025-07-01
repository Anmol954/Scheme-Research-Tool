from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import openai
from openai._exceptions import RateLimitError, AuthenticationError, OpenAIError

def create_faiss_index(content):
    if not content.strip():
        raise ValueError("Content is empty — cannot create FAISS index.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(content)

    if not texts:
        raise ValueError("Text splitting resulted in no chunks — check document content.")

    try:
        embeddings = OpenAIEmbeddings()
        faiss_index = FAISS.from_texts(texts, embeddings)
        return faiss_index

    except RateLimitError:
        raise RuntimeError("OpenAI API quota exceeded while creating embeddings.")

    except AuthenticationError:
        raise RuntimeError("Invalid OpenAI API Key. Please check your config.")

    except OpenAIError as e:
        raise RuntimeError(f"OpenAI error occurred while embedding: {str(e)}")

def run_qa(faiss_index, query):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=faiss_index.as_retriever())
    result = qa.run(query)
    return result
