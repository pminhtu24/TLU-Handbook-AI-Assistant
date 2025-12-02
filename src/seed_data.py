from genericpath import isfile
from typing import  Union, Any
from uuid import uuid4
from dotenv import load_dotenv
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from utils import create_documents, load_data_from_directory, load_data_from_local, load_data_from_upload
import streamlit as st
import os

load_dotenv()


@st.cache_resource
def seed_milvus(URI_link: str, 
                collection_name: str, 
                file_source: Union[str, Any], 
                use_vihuggingface: bool = False, 
                is_local: bool = False ) -> Milvus:
    """
    Function to create and save vector embeddings to Milvus from local data
    Args:
        URI_link (str): link to Milvus
        collection_name (str): Name of the collection in Milvus to connect to
        file_source (Union[str, Any]): source of data to upload
        use_ollama (bool): Use Ollama embeddings instead of OpenAI
        is_local (bool): Use data load from local
    
    Returns:
        Milvus: Vector store has been seeded
    """

    if use_vihuggingface:
        embeddings = HuggingFaceEmbeddings(
            model_name = "huyydangg/DEk21_hcmute_embedding",
        )
        print(f"----> Using viHuggingFace Embeddings: {embeddings.model_name}")
    else:
       embeddings = OpenAIEmbeddings(
            model = "text-embedding-3-large",
       )
       print(f"----> Using OpenAI Embeddings: {embeddings.model}")
    

    all_documents = []
    try:
        if is_local:
            # Load add files from directory
            if os.path.isdir(file_source):
                results = load_data_from_directory(file_source) # (markdown_text, doc_name))
                print(f"----> Loaded {len(results)} files from directory: {file_source}")
                for markdown_text, doc_name in results:
                    document = create_documents(markdown_text, doc_name)
                    all_documents.append(document)
            else:
                # Single file
                markdown_text, doc_name = load_data_from_local(file_source)
                print(f"----> Data loaded from local file: {doc_name}")
                document = create_documents(markdown_text, doc_name)
                all_documents.append(document)

        else:
            # Uploadd file
            markdown_text, doc_name = load_data_from_upload(file_source)
            print(f"----> Data loaded from file upload: {doc_name}")
            document = create_documents(markdown_text, doc_name)
    except Exception as e:
        raise Exception(f"Error when load file: {e}")
    

    # Text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,  
        length_function=len,
        separators=[
            "\n## ",  # Markdown heading level 2
            "\n### ", # Markdown heading level 3
            "\n\n",   # Paragraph break
            "\n",     # Line break
            ". ",     # Sentence
            " ",      # Word
            ""
        ]
    )
    split_documents = text_splitter.split_documents(all_documents)
    print(f"----> Split into {len(split_documents)} chunks")


    # UUID for each document
    uuids = [str(uuid4()) for _ in range(len(split_documents))]

    try:
        vectorstore = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": URI_link},
            collection_name=collection_name,
            drop_old=True 
        )
        print(f"=====Connected to Milvus=====")
        
    except Exception as e:
        raise Exception(f"Error connecting to Milvus: {e}")
    
    try:
        vectorstore.add_documents(documents=split_documents, ids=uuids)
        print(f"Added {len(split_documents)} chunks to collection '{collection_name}'")
    except Exception as e:
        raise Exception(f"Error adding documents: {str(e)}")
    
    return vectorstore


def connect_to_milvus(URI_link: str, collection_name: str):
    """
    Args:
        URI_link (str): link to Milvus
        collection_name (str): Name of the collection in Milvus to connect to
    Returns:
        Milvus: The Milvus object is connected and ready to query.
        
    """
    embeddings = HuggingFaceEmbeddings(
        model_name = "huyydangg/DEk21_hcmute_embedding",
    )
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args = {"uri": URI_link},
        collection_name=collection_name
    )
    return vectorstore 
        
