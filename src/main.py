import streamlit as st
from dotenv import load_dotenv
from seed_data import seed_milvus
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_classic.memory import StreamlitChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from agent import (get_retriever as get_openai_retriever, 
                   get_llm_and_agent as get_openai_agent)
from ollama_local import (get_retriever as get_ollama_retriever, 
                          get_llm_and_agent as get_ollama_agent)
import os
import gc
import torch

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

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

        # Model to answer 
        st.header("Model AI")
        model_choice = st.radio(
            "AI model to answer:",
            ["Qwen3-4B-Instruct (local)", "OpenAI GPT-5-nano"]
        )
        return model_choice, collection_to_query

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
        st.success(f"Đã tải file {uploaded_file.name} thành công !")
        file_details = {
            "filename": uploaded_file.name,
            "filetype": uploaded_file.type,
            "filesize": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.json(file_details)

        if st.button("Xử lý và lưu dữ liệu vào Milvus", type="primary"):
            if not collection_name:
                st.error("Vui lòng nhập tên collection trước khi tiếp tục.")
                return
            with st.spinner("Đang xử lý và lưu dữ liệu vào Milvus..."):
                try:
                    seed_milvus(
                        'http://localhost:19530',
                        collection_name,
                        uploaded_file,
                        use_vihuggingface=use_vihuggingface_embeddings
                    )
                    st.success(f"Đã tải dữ liệu thành công vào collection '{collection_name}'!")

                except Exception as e:
                    st.error(f"Lỗi khi lưu dữ liệu vào Milvus: {e}")
    return collection_name

def handle_local_file(use_vihuggingface_embeddings: bool):
    st.info("HEHE chua lam xong ham nay")

def setup_chat_interface(model_choice: str):
    st.title("AI Assistant")

    if model_choice == "Qwen3-4B-Instruct (local)":
        st.caption("Đang sử dụng mô hình Qwen3-4B-Instruct chạy local.")
    else:
        st.caption("Đang sử dụng mô hình OpenAI GPT-5-nano.")

    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Chào bạn! Mình có thể giúp gì cho bạn ?"}
        ]
    
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])

    return msgs

def handle_user_input(msgs, agent_executor):
    prompt = st.chat_input("Hãy hỏi tôi bất cứ điều gì về sổ tay sinh viên Thuỷ Lợi !")
    
    if prompt:  
        st.session_state.messages.append({"role": "human", "content": prompt})
        with st.chat_message("human"):
            st.write(prompt)
        msgs.add_user_message(prompt)

        with st.chat_message("assistant"):
            # Show thinking process in an expander
            with st.expander(">> Xem quá trình xử lý", expanded=False):
                st_callback = StreamlitCallbackHandler(st.container())
                
                chat_history = msgs.messages[:-1]
                response = agent_executor.invoke(
                    {"input": prompt, "chat_history": chat_history},
                    {"callbacks": [st_callback]}
                )
            
            answer = response["output"]
            st.session_state.messages.append({"role": "assistant", "content": answer})
            msgs.add_ai_message(answer)
            st.write(answer)
            release_memory()

#  === MAIN FUNCTION ===
def main():
    initialize_app()
    model_choice, collections_to_query = setup_sidebar()
    msgs = setup_chat_interface(model_choice)

    if model_choice == "OpenAI GPT-5-nano":
        retriever = get_openai_retriever(collections_to_query)
        agent_executor = get_openai_agent(retriever)
    else:
        retriever = get_ollama_retriever(collections_to_query)
        agent_executor = get_ollama_agent(retriever)

    handle_user_input(msgs, agent_executor)

if __name__ == '__main__':
    main()