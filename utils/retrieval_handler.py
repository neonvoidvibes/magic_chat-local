"""Utilities for retrieving relevant document context for chat."""
import logging
from typing import List, Optional, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from utils.pinecone_utils import init_pinecone
from utils.document_handler import DocumentHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalHandler:
    """Handles document retrieval for chat context."""

    def __init__(
        self,
        index_name: str = "magicchat",
        agent_name: str = None,  # We'll use this as the namespace
        session_id: str = None,  # Current session ID
        event_id: str = None,   # Current event ID
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
        self.session_id = session_id
        self.event_id = event_id
        self.top_k = top_k

        # Initialize embeddings and document handler
        self.embeddings = OpenAIEmbeddings()
        self.doc_handler = DocumentHandler()

        # Use the new Pinecone class-based usage
        pc = init_pinecone()
        if not pc:
            raise RuntimeError("Failed to initialize Pinecone")

        # Create or retrieve the actual index object
        self.index = pc.Index(self.index_name)

        # Use {agent}-{event} as namespace
        namespace = f"{self.namespace}-{event_id}" if event_id else f"{self.namespace}-0000"
        
        # Create the LangChain vector store
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            text_key="content",
            namespace=namespace
        )

        # We'll use a standard "similarity" search retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )

    def get_relevant_context(
        self,
        query: str,
        filter_metadata: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        is_transcript: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks for `query`.
        Returns clear source attribution with each result.
        
        Args:
            query: Search query
            filter_metadata: Additional metadata filters
            top_k: Number of results to return
            is_transcript: If True, only search in event-specific namespace
                         If False, search in both agent and event namespaces
        """
        try:
            # Build metadata filter
            base_filter = {}
            if self.session_id:
                base_filter['session_id'] = self.session_id
            if self.event_id:
                base_filter['event_id'] = self.event_id
            
            # Combine with any additional filters
            if filter_metadata:
                base_filter.update(filter_metadata)
            
            # For transcripts, only use event namespace
            if is_transcript:
                event_namespace = f"{self.namespace}-{self.event_id or '0000'}"
                docs = self.vectorstore.similarity_search(
                    query,
                    k=top_k or self.top_k,
                    filter=base_filter,
                    namespace=event_namespace
                )
            else:
                # For regular docs, search both namespaces
                agent_docs = self.vectorstore.similarity_search(
                    query,
                    k=top_k or self.top_k,
                    filter=base_filter,
                    namespace=self.namespace  # Agent's base namespace
                )
                
                event_namespace = f"{self.namespace}-{self.event_id or '0000'}"
                event_docs = self.vectorstore.similarity_search(
                    query,
                    k=top_k or self.top_k,
                    filter=base_filter,
                    namespace=event_namespace
                )
                
                # Combine and sort by relevance score
                docs = sorted(
                    agent_docs + event_docs,
                    key=lambda x: x.metadata.get('score', 0),
                    reverse=True
                )[:top_k or self.top_k]

            results = []
            for doc in docs:
                metadata = dict(doc.metadata)
                content = doc.page_content
                filename = metadata.get('file_name', 'unknown')
                source_path = metadata.get('source_path', '')
                
                # Format source information prominently
                source_header = f"[VECTOR DB CONTENT]\nSource: {filename}"
                if source_path:
                    source_header += f"\nPath: {source_path}"
                
                # Add clear separation between source and content
                labeled_content = f"{source_header}\n{'='*50}\n{content}\n{'='*50}"
                
                score = metadata.get("score", 0.0)
                results.append({
                    "content": labeled_content,
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