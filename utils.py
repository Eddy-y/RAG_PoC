
import os
import streamlit as st
import pandas as pd

import pinecone
from pinecone import Pinecone, PodSpec
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
#from langchain.vectorstores import Pinecone as Pinecone
from langchain_pinecone import PineconeVectorStore
#from langchain.embeddings import HuggingFaceEmbeddings 
from langchain_community.embeddings import HuggingFaceEmbeddings


# Set the PINECONE_API_KEY environment variable
os.environ["PINECONE_API_KEY"] = "1302c513-c9c7-4ee1-a15f-a6c9e00f9d7a"

FILE_LIST = "archivos.txt"
INDEX_NAME = 'rag'

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def save_name_files(path, new_files):

    old_files = load_name_files(path)

    with open(path, "a") as file:
        for item in new_files:
            if item not in old_files:
                file.write(item + "\n")
                old_files.append(item)
    
    return old_files


def load_name_files(path):

    archivos = []
    with open(path, "r") as file:
        for line in file:
            archivos.append(line.strip())

    return archivos


def clean_files(path):
    with open(path, "w") as file:
        pass
    
    delete_by_index(INDEX_NAME)

    return True

def delete_by_nameSpace(nameSpace):
    #TO DELETE FROM A NAMESPACE (ID, METADATA, NAMESPACE)
    index = pc.Index(INDEX_NAME)
    index.delete(delete_all=True, namespace=nameSpace)

def delete_by_index(indexName):
    pc.delete_index(indexName)

    pc.create_index(
        name=indexName,
        dimension=384,
        metric="cosine",
        spec=PodSpec(environment="gcp-starter")
    )


def text_to_pinecone2(file):

    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, file.name)
    with open(temp_filepath, "wb") as f:
        f.write(file.getvalue())

    if file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        # Read Excel file into a Pandas DataFrame
        df = pd.read_excel(temp_filepath)
        
        # Extract text data from the DataFrame
        text = ""
        for column in df.columns:
            text += " ".join(str(cell) for cell in df[column])
    else:
        loader = PyPDFLoader(temp_filepath)
        text = loader.load()

    with st.spinner(f'Creando embedding fichero: {file.name}'):
        create_embeddings(file.name, text)

    return True

def text_to_pinecone(pdf):

    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, pdf.name)
    with open(temp_filepath, "wb") as f:
        f.write(pdf.getvalue())

    loader = PyPDFLoader(temp_filepath)
    text = loader.load()

    with st.spinner(f'Creando embedding fichero: {pdf.name}'):
        create_embeddings(pdf.name, text)

    return True


def create_embeddings(file_name, text):
    print(f"Creando embeddings del archivo: {file_name}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
        )        
    
    chunks = text_splitter.split_documents(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    
    PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=INDEX_NAME)
        
    return True