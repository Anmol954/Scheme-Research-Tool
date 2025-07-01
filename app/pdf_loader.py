import requests
import os
import pytesseract
from langchain_community.document_loaders import UnstructuredPDFLoader

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def download_pdf(url, save_path="temp_download.pdf"):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    return save_path

def load_pdf_content(file_path, ocr_enabled=True):
    if ocr_enabled:
        os.environ["UNSTRUCTURED_OCR_AGENT"] = "pytesseract"
        loader = UnstructuredPDFLoader(
            file_path,
            unstructured_kwargs={
                "ocr_languages": "hin+eng"
            }
        )
    else:
        loader = UnstructuredPDFLoader(file_path, unstructured_kwargs={"ocr": False})

    docs = loader.load()
    content = "\n".join([doc.page_content for doc in docs])
    return content