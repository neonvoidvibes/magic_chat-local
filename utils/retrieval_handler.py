"""Utilities for retrieving relevant document context for chat."""
import logging
from typing import List, Optional, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from utils.pinecone_utils import init_pinecone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalHandler:
    """Handles document retrieval for chat context."""

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

        # Use the new Pinecone class-based usage
        pc = init_pinecone()
        if not pc:
            raise RuntimeError("Failed to initialize Pinecone")

        # Create or retrieve the actual index object
        self.index = pc.Index(self.index_name)

        # Create the LangChain vector store
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
        Retrieve the most relevant chunks for `query`.
        Currently ignoring filter_metadata for the default similarity search.

        Returns:
            A list of dicts with keys: 'content', 'metadata', 'score'
        """
        try:
            docs = self.retriever.get_relevant_documents(query)
            results = []
            for doc in docs:
                metadata = dict(doc.metadata)
                content = doc.page_content
                # Score is not guaranteed unless stored in metadata
                score = metadata.get("score", 0.0)
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

    def get_contextual_summary(
        self,
        query: str,
        max_tokens: int = 1000
    ) -> Optional[str]:
        """
        Generate a summary of relevant context for a query.
        (This is a placeholder method without real summarization logic.)
        """
        try:
            contexts = self.get_relevant_context(query)
            if not contexts:
                return None

            combined_text = "\n\n".join(c["content"] for c in contexts)
            # Just truncating for now
            return combined_text[: max_tokens * 4]

        except Exception as e:
            logger.error(f"Error generating context summary: {e}")
            return None