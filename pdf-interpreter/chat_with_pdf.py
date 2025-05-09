from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)
    return docs

def summarize_intro(llm, docs, chunk_count=3):
    intro_text = " ".join([doc.page_content for doc in docs[:chunk_count]])
    prompt = f"Summarize the following medical report in plain English:\n\n{intro_text}\n\nSummary:"
    return llm.invoke(prompt)

def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(docs, embeddings)

def extract_glossary(llm, doc_text):
    prompt = f"""
You are a helpful medical assistant. From the following medical report text, extract up to 10 medical terms or abbreviations, and explain each in simple language suitable for a patient.

Text:
{doc_text}

Return the result in this format:
Term: Definition
"""
    return llm.invoke(prompt)

