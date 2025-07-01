import streamlit as st
from app.pdf_loader import download_pdf, load_pdf_content
from app.summarizer_openai import generate_summary
from app.qa_engine_openai import create_faiss_index, run_qa
from app.utils import delete_temp_file, load_openai_api_key
from app.token_utils import count_tokens
import pickle
import os

api_key = load_openai_api_key(".config")
os.environ["OPENAI_API_KEY"] = api_key

st.title("Scheme Research Tool")

url = st.text_input("Enter PDF URL (leave blank if uploading a file)")
uploaded_file = st.file_uploader("Or upload a PDF document", type="pdf")
pdf_path = None

if st.button("Fetch & Summarize"):
    with st.spinner("Processing PDF..."):
        try:
            if uploaded_file is not None:
                pdf_path = f"temp_uploaded_{uploaded_file.name}"
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.read())
                st.success("File uploaded successfully!")

            elif url:
                pdf_path = download_pdf(url)
                st.success("PDF downloaded successfully!")

            else:
                st.error("Please either upload a PDF file or enter a URL.")
                st.stop()

            content = load_pdf_content(pdf_path)
            st.write("Content length:", len(content))
            st.write("First 500 characters:", content[:500])

            token_count = count_tokens(content, model="gpt-3.5-turbo")
            st.write(f"Total Tokens in Document: {token_count}")

            st.subheader("Extracted Content (First 1000 characters)")
            st.write(content[:1000] + "...")

            with st.spinner("Generating summary via OpenAI..."):
                summary = generate_summary(content)
                st.subheader("Scheme Summary")
                st.write(summary)

            with st.spinner("Creating FAISS index..."):
                faiss_index = create_faiss_index(content)
                with open("faiss_store_openai.pkl", "wb") as f:
                    pickle.dump(faiss_index, f)

            st.session_state['content'] = content
            st.success("Document processed and FAISS index created!")

        finally:
            if pdf_path:
                delete_temp_file(pdf_path)

if 'content' in st.session_state:
    question = st.text_input("Ask a question about the scheme:")
    if st.button("Get Answer"):
        if os.path.exists("faiss_store_openai.pkl"):
            with open("faiss_store_openai.pkl", "rb") as f:
                faiss_index = pickle.load(f)
            answer = run_qa(faiss_index, question)
            st.subheader("Answer")
            st.write(answer)
        else:
            st.error("FAISS index not found. Process a document first.")
