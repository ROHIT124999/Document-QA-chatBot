# 
import streamlit as smt
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

## load the GROQ And OpenAI API KEY 
groq_api_key=os.getenv('GROQ_API_KEY')
#Alternate groq_api_key = os.environ.get('GROQ_API_KEY)
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

smt.title("Document Q&A")

llm=ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the basis of provided pdf context only. Please provide the most accurate and a consise response less than 1000 words, based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

# def vector_embedding():

#     if "vectors" not in stm.session_state:

#         stm.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
#         stm.session_state.loader=PyPDFDirectoryLoader("./Rohit's__resume.pdf") ## Data Ingestion
#         stm.session_state.docs=stm.session_state.loader.load() ## Document Loading
#         stm.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
#         stm.session_state.final_documents=stm.session_state.text_splitter.split_documents(stm.session_state.docs[:20]) #splitting
#         stm.session_state.vectors=FAISS.from_documents(stm.session_state.final_documents,stm.session_state.embeddings) #vector OpenAI embeddings
def vector_embedding():
    if "vectors" not in smt.session_state:
        smt.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        smt.session_state.loader = PyPDFDirectoryLoader("./pdf")
        smt.session_state.docs = smt.session_state.loader.load()
        
        if not smt.session_state.docs:
            smt.error("No documents were loaded. Please check if the PDF file exists and is readable.")
            return
        
        smt.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        smt.session_state.final_documents = smt.session_state.text_splitter.split_documents(smt.session_state.docs)
        
        if not smt.session_state.final_documents:
            smt.error("No text could be extracted from the documents. Please check the content of your PDF file.")
            return
        
        try:
            smt.session_state.vectors = FAISS.from_documents(smt.session_state.final_documents, smt.session_state.embeddings)
            smt.success("Vector Store DB Is Ready")
        except Exception as e:
            smt.error(f"An error occurred while creating the vector store: {str(e)}")


if smt.button("Documents Embedding"):
    vector_embedding()
    smt.write("Vector Store DB Is Ready")

prompt1=smt.text_input("Enter Your Question From Doduments")




import time



if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=smt.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    smt.write(response['answer'])

    # With a streamlit expander
    with smt.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            smt.write(doc.page_content)
            smt.write("--------------------------------")