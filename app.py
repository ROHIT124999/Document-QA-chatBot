import os
import streamlit as slt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
import pypdf
from groq import Groq


# client = Groq(
#     api_key= os.environ.get("GROQ_API_KEY")
# )

print(groq_api)