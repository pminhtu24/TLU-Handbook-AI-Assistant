from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from FlagEmbedding import FlagReranker
from typing import List
import numpy as np

class ViRanker:
    def __init__(
            self, 
            model_name: str = "namdp-ptit/ViRanker",
            top_n: int = 3 ,
            use_fp16: bool = False,
            normalize: bool = True
    ):
        """
        Args:
            model_name: model rerank on HuggingFace
            top_n: nums of documents after rerank
            use_fp16: speed up compute
            normalize: sigmoid
        """
        self.model_name = model_name
        self.top_n = top_n
        self.normalize = normalize
        self.reranker = FlagReranker(
            model_name,
            use_fp16=use_fp16,
            device="cpu"
        )

    def rerank_documents(
            self,
            query: str,
            documents: List[Document],
    )-> List[Document]:
        """
        Rerank documents base on query úing FlagReranker
        
        Args:
            query: query from user
            documents: list of Document from retriever
            
        Returns:
            List of documents have been reranked and sorted according scores
        """
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]

        scores = self.reranker.compute_score(
            pairs,
            normalize=self.normalize
        )
        # convert to list[float] để type checker im mồm
        if scores is None:
            raise ValueError("Scores returned None")
        scores = np.atleast_1d(scores).tolist() 
        
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_n documents
        return [pair[0] for pair in doc_score_pairs[:self.top_n]]
    

class CustomRetrieverWithReranker(BaseRetriever):
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    base_retriever: BaseRetriever
    reranker: ViRanker

    def _get_relevant_documents(
            self, 
            query: str, 
            *,  
            run_manager: CallbackManagerForRetrieverRun | None = None) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        reranked_docs = self.reranker.rerank_documents(query, docs)
        return reranked_docs
    
