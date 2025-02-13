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
        agent_name: str = None,  # We'll use this as the namespace
        top_k: int = 5
    ):
        """Initialize retrieval handler with agent namespace.

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

        # Create vector store instance
        # This wraps a pinecone.Index object but avoids the problematic
        # PineconeHybridSearchRetriever from langchain_community.
        index = pinecone.Index(index_name)
        self.vectorstore = PineconeVectorStore(
            index=index,
            embedding_function=self.embeddings,
            text_key="content",        # Our chunk text is stored under metadata['content']
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
        Retrieve the most relevant chunks for `query`. Optionally accept a metadata filter.

        Returns:
            A list of dicts containing:
              {
                'content': chunk text,
                'metadata': {...},
                'score': similarity score or 0.0
              }
        """
        try:
            # If we want to apply metadata-based filtering, pass it to search
            # but the standard "as_retriever" approach doesn't handle a "filter" param by default.
            # We can implement it if needed; for now, ignoring filter_metadata or do a manual post-filter.

            docs: List[Document] = self.retriever.get_relevant_documents(query)

            results = []
            for doc in docs:
                # doc.page_content is the chunk text
                # doc.metadata is the stored metadata
                # 'score' is not always provided by default; we can store it in metadata if needed
                metadata = dict(doc.metadata)
                content = doc.page_content
                # We won't have a raw "score" from similarity unless we store it in doc.metadata,
                # so let's just default to 0.0
                results.append({
                    "content": content,
                    "metadata": metadata,
                    "score": metadata.get("score", 0.0)  # fallback
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
            # Basic approach: get top docs, combine, then truncate
            contexts = self.get_relevant_context(query)
            if not contexts:
                return None

            combined_text = "\n\n".join(c["content"] for c in contexts)
            # No real summarization logic yet; just truncate
            truncated = combined_text[: max_tokens * 4]  # rough char estimate
            return truncated

        except Exception as e:
            logger.error(f"Error generating context summary: {e}")
            return None