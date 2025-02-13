"""Utilities for retrieving relevant document context for chat."""
import logging
from typing import List, Optional, Dict, Any

import pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.schema import Document

from utils.pinecone_utils import init_pinecone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalHandler:
    """Handles document retrieval for chat context using a standard Pinecone vector store."""

    def __init__(
        self,
        index_name: str = "chat-docs-index",
        agent_name: str = None,
        top_k: int = 5
    ):
        """
        Args:
            index_name: Name of the Pinecone index
            agent_name: Namespace for the agent
            top_k: Number of results to retrieve
        """
        self.index_name = index_name
        self.namespace = agent_name
        self.top_k = top_k

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings()

        # Ensure Pinecone is up
        if not init_pinecone():
            raise RuntimeError("Failed to initialize Pinecone")

        # Classic usage: pinecone.Index(...)
        self.index = pinecone.Index(index_name)

        # Create vector store with the old langchain.vectorstores.Pinecone
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding_function=self.embeddings,
            text_key="content",
            namespace=self.namespace
        )

        # We'll use a standard "similarity" search retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )

    def get_relevant_context(
        self,
        query: str,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks for `query`. Optionally accept metadata filter.

        Returns:
            A list of dicts with 'content', 'metadata', 'score'
        """
        try:
            # We won't do metadata-based filtering automatically here,
            # since the built-in as_retriever doesn't handle a "filter" param by default.
            docs: List[Document] = self.retriever.get_relevant_documents(query)

            results = []
            for doc in docs:
                metadata = dict(doc.metadata)
                content = doc.page_content
                score = metadata.get("score", 0.0)  # fallback
                results.append({
                    "content": content,
                    "metadata": metadata,
                    "score": score
                })

            logger.info(f"Retrieved {len(results)} relevant contexts for query")
            return results

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []

    def get_contextual_summary(self, query: str, max_tokens: int = 1000) -> Optional[str]:
        """
        Generate a summary from the retrieved chunks for `query`.
        (Placeholder for user-specified summarization approach.)
        """
        try:
            contexts = self.get_relevant_context(query)
            if not contexts:
                return None
            combined_text = "\n\n".join(c["content"] for c in contexts)
            truncated = combined_text[: max_tokens * 4]
            return truncated

        except Exception as e:
            logger.error(f"Error generating context summary: {e}")
            return None