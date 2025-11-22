from langchain_core.documents import Document
from FlagEmbedding import FlagReranker
from typing import List
import numpy as np

class ViRanker:
    def __init__(
            self, 
            model_name: str = "namdp-ptit/ViRanker",
            top_n: int = 3 ,
            use_fp16: bool = True,
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
            use_fp16=use_fp16
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
            raise ValueError("Scores returned None, wtf?")
        scores = np.atleast_1d(scores).tolist() 
        
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_n documents
        return [pair[0] for pair in doc_score_pairs[:self.top_n]]
    

class CustomRetrieverWithReranker:
    """
    
    Custom Retriever combining base retriever and re-ranker
    To integrate with LangChain Agent
    
    """

    def __init__(self, base_retriever, reranker: ViRanker):
        self.base_retriever = base_retriever
        self.reranker = reranker

    def get_relevant_documents(self, query: str):
        docs = self.base_retriever.get_relevant_documents(query)
        reranked_docs = self.reranker.rerank_documents(query, docs)
        return reranked_docs