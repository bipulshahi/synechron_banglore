import streamlit as st
import PyPDF2
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import os

os.environ['OPENAI_API_KEY'] = ''
os.environ['GOOGLE_API_KEY'] = ''

def extract_data_from_pdf(pdf_path):
    with open(pdf_path , 'rb') as file:
        pdfreader = PyPDF2.PdfReader(file)
        full_text = ''
        for page in pdfreader.pages:
            full_text += page.extract_text()
    return full_text

#RecursiveCharacterTextSplitter
def split_text(text):
  splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
  docs = splitter.create_documents([text])
  return docs

#FAISS
def create_vector_store(docs):
  embeddings = OpenAIEmbeddings()
  vectorstore = FAISS.from_documents(docs , embeddings)
  return vectorstore

def setup_rag_qa(vectorstore):
  retriever = vectorstore.as_retriever(search_type = 'similarity')
  #llm = ChatOpenAI(model = "gpt-4.1-nano")
  llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
  rag_chain = RetrievalQA.from_chain_type(llm=llm , retriever=retriever)
  return rag_chain

#import google.generativeai as genai
#genai.configure(api_key = "")

st.title("User Interface for RAG...")

prompt = st.text_area("Whats in your mind today?")

data_path = 'company_manual.pdf'

if st.button("Generate Output..."):
    extracted_text = extract_data_from_pdf(data_path)
    docs = split_text(extracted_text)
    embeddings = create_vector_store(docs)
    rag_app = setup_rag_qa(embeddings)
    response = rag_app(prompt)
    st.text(f"Here is your response:- {response['result']}")