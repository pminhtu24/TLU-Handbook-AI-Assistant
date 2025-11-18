from langchain_classic.tools.retriever import create_retriever_tool
from langchain_ollama import ChatOllama
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from seed_data import connect_to_milvus
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

def get_retriever(collection_name: str = "student_handbook"):
    """
    Args:
        collection_name (str): Tên collection trong Milvus để truy vấn
    Returns:
        EnsembleRetriever: Retriever kết hợp với weights:
            - 70% Milvus vector search 
            - 30% BM25 text search 

    """
    try:
        # Milvus retriever
        vectorstore = connect_to_milvus(
            'http://localhost:19530',
            collection_name
        )

        milvus_retriever = vectorstore.as_retriever(
            search_type = "similarity",
            search_kwargs= {"k": 4},
        )

        #Create BM25 retriever from whole documents
        documents = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in vectorstore.similarity_search("",k=100)
        ]

        if not documents:
            raise ValueError(f"Không tìm thấy documents trong collection '{collection_name}'")

        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 4
        
        # Ensemble retriever with weights
        ensemble_retriever = EnsembleRetriever(
            retrievers = [milvus_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )

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
    
    
def get_llm_and_agent(retriever):
    tool = create_retriever_tool(
        retriever,
        "find",
        "Search for information of Student Handbook"
    )

    llm = ChatOllama(
        model="qwen3:4b-instruct",
        temperature=0.1,
    )
    
    tools =[tool]

    # System prompt template
    prompt = PromptTemplate.from_template("""You are ChatchatAI - chuyên gia trả lời về sổ tay sinh viên Đại học Thủy Lợi.

        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format instruction:
        ```
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        Final Answer: the final answer to the original input question
        ```
                                          
        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
        ```
        Thought: Do I need to use a tool? No
        Final Answer: [your response here]
        
        ```
        IMPORTANT RULES:
        - ALWAYS use the find_documents tool when answering questions about student handbook
        - DO NOT make up information if you don't have data
        - Answer in Vietnamese
        - Be concise and accurate
                                                            
        Begin!
                                          
        Previous conversation history: {chat_history}
        New input: {input}
        {agent_scratchpad}
        """)
        

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=2,
        )
    
retriever = get_retriever()
agent_executor = get_llm_and_agent(retriever)