import streamlit as st
from dotenv import load_dotenv
from seed_data import seed_milvus
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from agent import (get_retriever as get_openai_retriever, 
                   get_llm_and_agent as get_openai_agent)
from ollama_local import get_qa_chain, get_retriever
import os
import gc
import torch

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def release_memory():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()

def setup_page():
    st.set_page_config(
        page_title="AI Assistant", 
        page_icon="💬",  
        layout="wide"  
    )

def initialize_app():
    load_dotenv()
    setup_page()

def setup_sidebar():
    with st.sidebar:
        st.title("Configuration")

        # Chose embedding model
        st.header("Embeddings Model")
        embedding_choice = st.radio(
            "Select Embedding Model:",
            ["viHuggingFace Embeddings", "OpenAI Embeddings"]
        )

        use_vihuggingface_embeddings = ( embedding_choice == "viHuggingFace Embeddings" )

        # Config data
        data_source = st.selectbox(
            "Select Data Source:",
            ("Upload Handbook", "File local")
        )

        if data_source == "Upload Handbook":
            handle_upload_file(use_vihuggingface_embeddings)
        else:
            handle_local_file(use_vihuggingface_embeddings)

        st.header("Collection to query")
        collection_to_query = st.text_input(
            "Enter collection name stored Milvus: ",
            "student_handbook",
            help="Nhập tên collection bạn muốn sử dụng để tìm kiếm thông tin"
        )

        # Reranking config
        st.header("Reranking Settings")
        use_rerank = st.checkbox(
            "Use Reranking",
            value=True,
            help="Sử dụng model ViRanker để rerank kết quả tìm kiếm, cải thiện độ chính xác"

        )
        rerank_top_n = 3
        if use_rerank:
            rerank_top_n = st.slider(
                "Nums of documents after reranking",
                min_value=1,
                max_value=10,
                value=3
            )

        # Model to answer 
        st.header("Model AI")
        model_choice = st.radio(
            "AI model to answer:",
            ["Qwen3-4B-Instruct (local)", "OpenAI GPT-5-nano"]
        )
        return model_choice, collection_to_query, use_rerank, rerank_top_n

def handle_upload_file(use_vihuggingface_embeddings: bool):
    collection_name = st.text_input(
        "Collection name to save in Milvus: ",
        "student_handbook",
        help="Nhập tên collection để lưu trữ dữ liệu trong Milvus",
    )

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload file",
        type=["json", "pdf"],
        help="Tải lên file handbook định dạng JSON, PDF"
    )

    if uploaded_file:
        st.success(f"Upload file {uploaded_file.name} sucess!")
        file_details = {
            "filename": uploaded_file.name,
            "filetype": uploaded_file.type,
            "filesize": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.json(file_details)

        if st.button("Process and store in Milvus", type="primary"):
            if not collection_name:
                st.error("Enter collection name")
                return
            with st.spinner("Processing..."):
                try:
                    seed_milvus(
                        'http://localhost:19530',
                        collection_name,
                        uploaded_file,
                        use_vihuggingface=use_vihuggingface_embeddings
                    )
                    st.success(f"Saved to collection successfully '{collection_name}'!")

                except Exception as e:
                    st.error(f"Error: {e}")
    return collection_name

def handle_local_file(use_vihuggingface_embeddings: bool):
    st.subheader("Load Data from local")
    collection_name = st.text_input(
        "Collection name to save in Milvus: ",
        "student_handbook",
        help="Nhập tên collection để lưu trữ dữ liệu trong Milvus",
    )

    directory_path = st.text_input("Directory or single file path: ",
                                    help="Nhập đường dẫn đến thư mục hoặc file",
                                    key="directory_path")
    if directory_path:
        if os.path.exists(directory_path):
            if os.path.isdir(directory_path):
                # Its a directory
                pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
                if pdf_files:
                    st.success(f"Found {len(pdf_files)} PDF files")

                    with st.expander(f"List of {len(pdf_files)} PDF files", expanded=True):
                        for i, pdf_file in enumerate(pdf_files, 1):
                            file_path = os.path.join(directory_path, pdf_file)
                            file_size = os.path.getsize(file_path) / 1024
                            st.write(f"{i}. **{pdf_file}** - {file_size:.2f} KB")
                else:
                    st.warning("The folder does not contain any supported files !")
            
            elif directory_path.lower().endswith('.pdf'):
                # Its a single file
                st.success("---->Valid File PDF")
                file_size = os.path.getsize(directory_path) / 1024
                st.json({
                    "filename": os.path.basename(directory_path),
                    "filesize": f"{file_size:.2f} KB",
                    "full_path": directory_path
                })
            else:
                st.warning("The path is not a folder or PDF file")
        else:
            st.warning("The path do es not exist !")

    # Process button
    if st.button("Process and store in Milvus", type="primary", key="local_btn"):
        if not collection_name:
            st.error("Enter collection name")
            return
        
        if not directory_path or not os.path.exists(directory_path):
            st.error("Invalid path")
            return
        
        with st.spinner("Processing..."):
            try:
                seed_milvus(
                    'http://localhost:19530',
                    collection_name,
                    directory_path,  # Can be file or folder
                    use_vihuggingface=use_vihuggingface_embeddings,
                    is_local=True
                )
                st.success(f"Saved to collection successfully '{collection_name}'!")
                st.balloons()
            except Exception as e:
                st.error(f"Error: {e}")

def setup_chat_interface(model_choice: str, use_rerank: bool, rerank_top_n: int):
    st.title("AI Assistant")

    caption_parts = []
    if model_choice == "Qwen3-4B-Instruct (local)":
        caption_parts.append("Qwen3-4B-Instruct (local)")
    else:
        caption_parts.append("OpenAI GPT-5-nano")
    
    if use_rerank:
        caption_parts.append(f"ViRanker (top {rerank_top_n})")
    
    st.caption(" | ".join(caption_parts))

    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Chào bạn! Mình có thể giúp gì cho bạn ?"}
        ]
    
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])

    return msgs

def handle_user_input(msgs, qa_chain):
    prompt = st.chat_input("Hãy hỏi tôi bất cứ điều gì về sổ tay sinh viên Thuỷ Lợi !")
    
    if prompt:  
        st.session_state.messages.append({"role": "human", "content": prompt})
        with st.chat_message("human"):
            st.write(prompt)
        msgs.add_user_message(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Đang tìm kiếm và trả lời..."):
                for chunk in qa_chain.stream(prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            msgs.add_ai_message(full_response)
            

            release_memory()

#  === MAIN FUNCTION ===
def main():
    initialize_app()
    model_choice, collections_to_query, use_rerank, rerank_top_n = setup_sidebar()
    msgs = setup_chat_interface(model_choice, use_rerank, rerank_top_n)

    if model_choice == "OpenAI GPT-5-nano":
        retriever = get_openai_retriever(
            collections_to_query,
            use_rerank =use_rerank,
            top_n=rerank_top_n)
        agent_executor = get_openai_agent(retriever)

    else:
        qa_chain = get_qa_chain(
            collection_name=collections_to_query,
            use_rerank=use_rerank,
            top_n=rerank_top_n
        )
        handle_user_input(msgs, qa_chain)

if __name__ == '__main__':
    main()