from langchain_classic.tools.retriever import create_retriever_tool
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.documents import Document 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.retrievers import BM25Retriever
from seed_data import connect_to_milvus
from reranker import ViRanker, CustomRetrieverWithReranker
from dotenv import load_dotenv
from pydantic import SecretStr
import os


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

api_key = SecretStr(OPENAI_API_KEY)

def get_retriever(
        collection_name: str = "student_handbook", 
        use_rerank: bool = True,
        top_n: int = 3
    ):
    """
    Tạo một ensemble retriever kết hợp vector search (Milvus) và BM25
    Args:
        collection_name (str): Tên collection trong Milvus để truy vấn
    """
    try:
        vectorstore = connect_to_milvus(
            'http://localhost:19530',
            collection_name
        )
        milvus_retriever = vectorstore.as_retriever(
            search_type = 'similarity',
            search_kwargs = {"k": 4}
        )

        # Create BM25 retriever from whole documents
        documents = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in vectorstore.similarity_search("",k=100)
        ]

        if not documents:
            raise ValueError(f"Không tìm thấy documents trong collection '{collection_name}'")
        
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 4

        # Gather 2 retriever with weights
        ensemble_retriever = EnsembleRetriever(
            retrievers = [milvus_retriever, bm25_retriever],
            weights = [0.7, 0.3]
        )

        if use_rerank:
            reranker = ViRanker(
                model_name="namdp-ptit/ViRanker",
                top_n = top_n,
                use_fp16=True,
                normalize=True
            )

            final_retriever = CustomRetrieverWithReranker(
                base_retriever=ensemble_retriever,
                reranker=reranker
            )
            return final_retriever

        return ensemble_retriever
    except Exception as e:
        print(f"Lỗi khi khởi tạo retriever: {str(e)}")
        # Return back to default retriever with document if it errors
        default_doc = [
            Document(
                page_content="Có lỗi xảy ra khi kết nối database. Vui lòng thử lại sau.",
                metadata = {"source": "error"}
            )
        ]
        return BM25Retriever.from_documents(default_doc)
    

# Create search tool for agent
tool = create_retriever_tool(
    get_retriever(),
    "find_documents",
    "Search for information of Student Handbook"
)

def get_llm_and_agent( _retriever, model_choice="gpt-5-nano"):
    
    """
    Khởi tạo Language Model và Agent với cấu hình cụ thể
    Args:
        _retriever: Retriever đã được cấu hình để tìm kiếm thông tin
        model_choice: Lựa chọn model ("gpt5" hoặc "ollama (local)")
    """

    if model_choice == "gpt-5-nano":
        llm = ChatOpenAI(
            model = 'gpt-5-nano',
            temperature=0,
            streaming=True,
            api_key = api_key,
        )
    else:
        raise ValueError(f"Không hỗ trợ model_choice này !")
    tools = [tool]
    
    # Prompt template for agent
    system = """
    You are ChatchatAI - chuyên gia trả lời về sổ tay sinh viên Đại học Thủy Lợi.
    Luôn sử dụng công cụ `find_documents` khi cần tra cứu thông tin.
    Không được tự bịa đặt nếu không có dữ liệu.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create and return agent
    agent = create_openai_functions_agent(llm=llm, tools=tools,prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Initialize retriever and agent
retriever = get_retriever()
agent_executor = get_llm_and_agent(retriever)