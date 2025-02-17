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
            logging.info(f"Retrieving context for query: {query}")
            logging.info(f"Is transcript query: {is_transcript}")
            
            # Build metadata filter - only use essential fields
            base_filter = {}
            # First try without filters to see what's in the index
            try:
                stats = self.index.describe_index_stats()
                logging.info(f"Index stats: {stats}")
            except Exception as e:
                logging.error(f"Error getting index stats: {e}")
                
            # Try to match on file name or source if filters are provided
            if filter_metadata:
                if 'file_name' in filter_metadata:
                    base_filter['file_name'] = filter_metadata['file_name']
                if 'source' in filter_metadata:
                    base_filter['source'] = filter_metadata['source']
                    
            logging.info(f"Using metadata filters: {base_filter}")
            
            logging.info(f"Using metadata filter: {base_filter}")
            
            # Get embedding for query
            try:
                query_embedding = self.embeddings.embed_query(query)
                logging.info(f"Generated embedding vector of length: {len(query_embedding)}")
            except Exception as e:
                logging.error(f"Error generating embedding: {e}")
                raise

            # Search in the base namespace
            try:
                # Get raw matches first
                raw_response = self.index.query(
                    vector=query_embedding,
                    top_k=top_k or self.top_k,
                    namespace=self.namespace,
                    include_metadata=True
                )
                logging.info(f"Raw matches: {len(raw_response.matches)}")
                
                # Convert to Documents
                docs = []
                for match in raw_response.matches:
                    content = match.metadata.get('content', '')
                    metadata = {
                        'score': match.score,
                        'file_name': match.metadata.get('file_name', 'unknown')
                    }
                    docs.append(Document(page_content=content, metadata=metadata))
                    
                logging.info(f"Found {len(docs)} documents in namespace {self.namespace}")
                return docs
            except Exception as e:
                logging.error(f"Error querying Pinecone: {e}")
                return []
            else:
                return []

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