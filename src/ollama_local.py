from langchain_classic.tools.retriever import create_retriever_tool
from langchain_ollama import ChatOllama
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from seed_data import connect_to_milvus
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from reranker import ViRanker, CustomRetrieverWithReranker

def get_retriever(
    collection_name: str = "student_handbook",
    use_rerank: bool = True,
    top_n: int = 3):
    """
    Args:
        collection_name (str)
    Returns:
        EnsembleRetriever: Retriever combine with weights:
            - 70% Milvus vector search 
            - 30% BM25 text search 
            - (Optional) ViRanker reranking
    """
    try:
        # Milvus retriever
        vectorstore = connect_to_milvus(
            'http://localhost:19530',
            collection_name
        )
        k_retriever = 8 if use_rerank else 4
        milvus_retriever = vectorstore.as_retriever(
            search_type = "similarity",
            search_kwargs= {"k": k_retriever},
        )

        #Create BM25 retriever from whole documents
        documents = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in vectorstore.similarity_search("",k=100)
        ]

        if not documents:
            raise ValueError(f"No documents found in collection '{collection_name}'")

        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = k_retriever
        
        # Ensemble retriever with weights
        ensemble_retriever = EnsembleRetriever(
            retrievers = [milvus_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        if use_rerank:
            reranker = ViRanker(
                model_name="namdp-ptit/ViRanker",
                top_n=top_n,
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
        print(f"Error initializing retriever: {str(e)}")
        # Return back to default retriever with document if it errors
        default_doc = [
            Document(
                page_content="An error occurred while connecting to the database. Please try again later.",
                metadata = {"source": "error"}
            )
        ]
        return BM25Retriever.from_documents(default_doc)
    

def get_llm_and_agent(retriever):
    tool = create_retriever_tool(
        retriever,
        "find_documents",
        "Search for information of Student Handbook"
    )

    llm = ChatOllama(
        model="qwen3:4b-instruct",
        temperature=0.1,
    )
    
    tools =[tool]

    # System prompt template
    prompt = PromptTemplate.from_template("""You are ChatChitAI - chuyên gia trả lời về sổ tay sinh viên Đại học Thủy Lợi.

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
        
        Use the format instruction above !
                           
        Previous conversation history: {chat_history}
        Question: {input}
        {agent_scratchpad}
        """)
        

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=3,
        )
    
retriever = get_retriever(use_rerank=True, top_n=3)
agent_executor = get_llm_and_agent(retriever)