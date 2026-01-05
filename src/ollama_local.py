from langchain_classic.tools.retriever import create_retriever_tool
from langchain_ollama import ChatOllama
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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
    

def get_qa_chain(
        collection_name: str = "student_handbook",
        use_rerank: bool = True,
        top_n: int = 3 ):
    retriever = get_retriever(
        collection_name=collection_name,
        use_rerank=use_rerank,
        top_n=top_n
    )

    llm = ChatOllama(
        model="qwen3:4b-instruct",
        temperature=0.1
    )

    template = """Bạn là ChatChitAI - trợ lý chuyên trả lời về Sổ tay sinh viên Đại học Thủy Lợi.

    Chỉ dựa vào thông tin sau để trả lời, KHÔNG bịa đặt thêm:

    Context:
    {context}

    Câu hỏi: {question}

    Trả lời ngắn gọn, chính xác bằng tiếng Việt. 
    Nếu không có thông tin trong context thì trả lời: "Tôi không tìm thấy thông tin về vấn đề này trong sổ tay sinh viên."""" """

    prompt = PromptTemplate.from_template(template)
    # Format context đẹp, có source tài liệu
    def format_docs(docs):
        return "\n\n".join(
            f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}"
            for doc in docs
        )

    # Chain đơn giản: retriever → format → prompt → LLM → string
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


retriever = get_retriever(use_rerank=True, top_n=3)
agent_executor = get_qa_chain(use_rerank=True, top_n=3)