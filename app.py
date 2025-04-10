import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import (
    ChatPromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
 
load_dotenv()
 
# Load Groq and Google API Key from the .env file
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
 
st.title("Gemma Model Document Q&A")
 
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")
 
template_string = """
Answer the question based on the provided context only.
Please provide the most accurate response based on the question.
<Context>
{context}
<context>
Question: {input}
"""
 
prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(template_string)
])
 
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./urpdfs")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
 
prompt1 = st.text_input("What do you want to ask from the documents")
 
if st.button("Create Vector Store"):
    vector_embedding()
    st.write("Vector Store Created")
 
import time
 
if prompt1:
    retriever = st.session_state.vectors.as_retriever()
    retrieval_qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
 
    start = time.process_time()
    response = retrieval_qa_chain({"query": prompt1})
    st.write(response["result"])