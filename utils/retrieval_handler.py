"""Utilities for retrieving relevant document context for chat."""
import logging
from typing import List, Optional, Dict, Any
import pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import PineconeHybridSearchRetriever
from utils.pinecone_utils import init_pinecone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalHandler:
    """Handles document retrieval for chat context."""
    
    def __init__(
        self,
        index_name: str = "chat-docs-index",
        namespace: Optional[str] = None,
        top_k: int = 5
    ):
        """Initialize retrieval handler.
        
        Args:
            index_name: Name of Pinecone index to use
            namespace: Optional namespace within index
            top_k: Number of results to retrieve
        """
        self.index_name = index_name
        self.namespace = namespace
        self.top_k = top_k
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize Pinecone
        if not init_pinecone():
            raise RuntimeError("Failed to initialize Pinecone")
            
        # Get index instance
        self.index = pinecone.Index(index_name)
        
        # Initialize hybrid retriever
        self.retriever = PineconeHybridSearchRetriever(
            embeddings=self.embeddings,
            index=self.index,
            namespace=namespace,
            top_k=top_k
        )
        
    def get_relevant_context(
        self,
        query: str,
        agent_name: str,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for a query, filtered by agent.
        
        Args:
            query: Search query text
            agent_name: Name of agent to filter documents
            filter_metadata: Additional optional metadata filters
            
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            # Generate query embedding
            vector = self.embeddings.embed_query(query)
            
            # Combine agent filter with any additional filters
            agent_filter = {
                "agent_name": agent_name
            }
            if filter_metadata:
                agent_filter.update(filter_metadata)
            
            # Query index with agent filter
            results = self.index.query(
                vector=vector,
                filter=agent_filter,
                namespace=self.namespace,
                top_k=self.top_k,
                include_metadata=True
            )
            
            # Format results
            contexts = []
            for match in results.matches:
                contexts.append({
                    'content': match.metadata['content'],
                    'metadata': {
                        k: v for k, v in match.metadata.items()
                        if k != 'content'  # Exclude content from metadata
                    },
                    'score': match.score
                })
                
            logger.info(f"Retrieved {len(contexts)} relevant contexts for query")
            return contexts
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
            
    def get_hybrid_search_results(
        self,
        query: str,
        filter_metadata: Optional[Dict[str, Any]] = None,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword matching.
        
        Args:
            query: Search query text
            filter_metadata: Optional metadata filters
            alpha: Weight between vector and keyword search (0=keywords only, 1=vectors only)
            
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            # Use hybrid retriever
            results = self.retriever.get_relevant_documents(
                query,
                filter=filter_metadata,
                alpha=alpha
            )
            
            # Format results
            contexts = []
            for doc in results:
                contexts.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': doc.metadata.get('score', 0.0)  # Score if available
                })
                
            logger.info(f"Retrieved {len(contexts)} results from hybrid search")
            return contexts
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
            
    def get_contextual_summary(
        self,
        query: str,
        max_tokens: int = 1000
    ) -> Optional[str]:
        """Generate a summary of relevant context for a query.
        
        Args:
            query: Search query text
            max_tokens: Maximum tokens for summary
            
        Returns:
            Summarized context string if successful
        """
        try:
            # Get relevant contexts
            contexts = self.get_relevant_context(query)
            if not contexts:
                return None
                
            # Combine contexts
            combined_text = "\n\n".join(
                c['content'] for c in contexts
            )
            
            # TODO: Add summarization logic here
            # For now, just return truncated text
            return combined_text[:max_tokens * 4]  # Rough char estimate
            
        except Exception as e:
            logger.error(f"Error generating context summary: {e}")
            return None