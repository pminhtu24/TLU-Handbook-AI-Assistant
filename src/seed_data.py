from typing import  Union, Any
from uuid import uuid4
from dotenv import load_dotenv
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from utils import load_data_from_upload
import streamlit as st
load_dotenv()


@st.cache_resource
def seed_milvus(URI_link: str, collection_name: str, file_source: Union[str, Any], use_vihuggingface: bool = False ) -> Milvus:
    """
    Hàm tạo và lưu vector embeddings vào Milvus từ dữ liệu local
    Args:
        URI_link (str): Đường dẫn kết nối đến Milvus
        collection_name (str): Tên collection trong Milvus để lưu dữ liệu
        file_source (Union[str, Any]): Tên file JSON HOẶC file object (UploadedFile)
        use_ollama (bool): Sử dụng Ollama embeddings thay vì OpenAI
    
    Returns:
        Milvus: Vector store đã được seed
    """

    if use_vihuggingface:
        embeddings = HuggingFaceEmbeddings(
            model_name = "huyydangg/DEk21_hcmute_embedding",
        )
        print(f"-> Sử dụng viHuggingFace Embeddings: {embeddings.model_name}")
    else:
       embeddings = OpenAIEmbeddings(
            model = "text-embedding-3-large",
       )
       print(f"-> Sử dụng OpenAI Embeddings: {embeddings.model}")
    

    try:
        markdown_text, doc_name = load_data_from_upload(file_source)
        print(f"-> Dữ liệu được load từ file: {doc_name}")

    except Exception as e:
        raise Exception(f"Lỗi khi load dữ liệu: {e}")
    
    doc_metadata = {
        'source': doc_name,
        'content_type': 'text/markdown',
        'title': doc_name,
        'description': f'Markdown document converted from PDF: {doc_name}',
        'language': 'vi',
        'doc_name': doc_name
    }

    document = Document(
        page_content = markdown_text,
        metadata = doc_metadata
    )
    print(f"-> Đã tạo Document với metadata: {doc_metadata}")


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
    split_documents = text_splitter.split_documents([document])
    print(f"-> Đã chia thành {len(split_documents)} chunks sau khi split")


    # UUID cho mỗi document
    uuids = [str(uuid4()) for _ in range(len(split_documents))]

    try:
        vectorstore = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": URI_link},
            collection_name=collection_name,
            drop_old= True 
        )
        print(f"=====Đã kết nối đến Milvus=====")
        
    except Exception as e:
        raise Exception(f"Lỗi khi kết nối Milvus: {e}")
    
    try:
        vectorstore.add_documents(documents=split_documents, ids=uuids)
        print(f"Đã thêm {len(split_documents)} chunks vào collection '{collection_name}'")
    except Exception as e:
        raise Exception(f" Lỗi khi thêm documents: {str(e)}")
    
    return vectorstore


def connect_to_milvus(URI_link: str, collection_name: str):
    """
    Hàm kết nối đến collection có sẵn trong Milvus 
    Args:
        URI_link (str): Đường dẫn kết nối đến Milvus
        collection_name (str): Tên collection trong Milvus cần kết nối 
    Returns:
        Milvus: Đối tượng Milvus đã được connect, sẵn sàng query
        
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
        
