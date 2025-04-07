"""Utilities for retrieving relevant document context for chat."""
import logging
from typing import List, Optional, Dict, Any
from langchain_openai import OpenAIEmbeddings
# We use the Pinecone client directly, so Langchain PineconeVectorStore is not strictly needed here
# from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_core.documents import Document # Use Document class from langchain core
from pinecone import Pinecone
from utils.pinecone_utils import init_pinecone
from utils.document_handler import DocumentHandler # DocumentHandler might not be needed here if not re-chunking

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalHandler:
    """Handles document retrieval for chat context using direct Pinecone queries."""

    def __init__(
        self,
        index_name: str = "magicchat",
        agent_name: Optional[str] = None,  # Namespace corresponds to agent name
        session_id: Optional[str] = None,  # Current session ID (optional usage)
        event_id: Optional[str] = None,    # Current event ID (optional usage for namespace/filtering)
        top_k: int = 5
    ):
        """Initialize retrieval handler with agent namespace.

        Args:
            index_name: Name of the Pinecone index
            agent_name: Namespace for the agent (mandatory for retrieval)
            session_id: Optional session ID for potential future filtering
            event_id: Optional event ID for potential future filtering or namespace construction
            top_k: Default number of results to retrieve
        """
        if not agent_name:
            raise ValueError("agent_name is required for RetrievalHandler")

        self.index_name = index_name
        self.namespace = agent_name # Use agent_name directly as the primary namespace
        self.session_id = session_id
        self.event_id = event_id
        self.top_k = top_k

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings()
        # self.doc_handler = DocumentHandler() # Not needed for retrieval only

        # Use the new Pinecone class-based usage
        pc = init_pinecone()
        if not pc:
            raise RuntimeError("Failed to initialize Pinecone")

        # Get the actual index object - assume it exists after create_or_verify elsewhere
        try:
            self.index = pc.Index(self.index_name)
            logger.info(f"Successfully connected to Pinecone index '{self.index_name}'")
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone index '{self.index_name}': {e}")
            raise RuntimeError(f"Failed to connect to Pinecone index '{self.index_name}'") from e

        # We are using direct queries, so Langchain retriever setup is commented out
        # self.vectorstore = PineconeVectorStore(...)
        # self.retriever = self.vectorstore.as_retriever(...)

    def get_relevant_context(
        self,
        query: str,
        filter_metadata: Optional[Dict[str, Any]] = None, # Allow passing extra filters
        top_k: Optional[int] = None,
        is_transcript: bool = False # Flag might influence which namespace/filters are prioritized
    ) -> List[Document]: # Return Langchain Document objects
        """
        Retrieve the most relevant document chunks for `query` using direct Pinecone query
        with appropriate namespace and metadata filters.

        Args:
            query: Search query
            filter_metadata: Optional dictionary of additional metadata key-value pairs to filter on.
            top_k: Number of results to return (overrides default if provided)
            is_transcript: Hint that query relates to transcript (might influence strategy in future)

        Returns:
            List of Langchain Document objects containing relevant context and metadata.
        """
        try:
            k = top_k or self.top_k
            logger.info(f"Retrieving top {k} contexts for query (transcript hint: {is_transcript}): {query[:100]}...")

            # Generate embedding for the user query
            try:
                query_embedding = self.embeddings.embed_query(query)
                logger.debug(f"Generated query embedding vector of length: {len(query_embedding)}")
            except Exception as e:
                logger.error(f"Error generating query embedding: {e}")
                return [] # Cannot proceed without embedding

            # --- Define namespaces to search ---
            namespaces_to_search = [self.namespace] # Always search the base agent namespace
            event_namespace = f"{self.namespace}-{self.event_id}" if self.event_id else None
            if event_namespace and event_namespace not in namespaces_to_search:
                 # If event_id exists, also search the event-specific namespace
                 # Useful if transcripts/event docs go into a separate namespace
                 # Note: cli_embed currently only uploads to the base agent namespace
                 logger.info(f"Also searching event-specific namespace: {event_namespace}")
                 namespaces_to_search.append(event_namespace)

            # --- Perform queries with filters ---
            all_matches = []
            for ns in namespaces_to_search:
                # Build the metadata filter for this namespace query
                # Start with the mandatory agent_name filter
                query_filter = {"agent_name": self.namespace} # Filter based on agent_name stored in metadata

                # Add event_id filter if searching the event namespace *and* event_id is known
                if ns == event_namespace and self.event_id:
                    query_filter["event_id"] = self.event_id
                    logger.debug(f"Adding event_id filter for namespace {ns}")

                # Merge any additional filters passed externally
                if filter_metadata:
                    query_filter.update(filter_metadata)

                logger.info(f"Querying namespace '{ns}' with filter: {query_filter}")

                try:
                    response = self.index.query(
                        vector=query_embedding,
                        top_k=k, # Query for k results from each relevant namespace
                        namespace=ns,
                        filter=query_filter, # Apply the constructed filter
                        include_metadata=True # Essential to get content and other details
                    )
                    logger.info(f"Found {len(response.matches)} raw matches in namespace '{ns}'.")
                    all_matches.extend(response.matches)
                except Exception as query_e:
                    logger.error(f"Error querying Pinecone namespace '{ns}': {query_e}")
                    # Continue to next namespace if one fails

            if not all_matches:
                logger.warning("No relevant context matches found across all searched namespaces.")
                return []

            # --- Process and Rank Results ---
            # Combine results from all namespaces and re-rank by score
            all_matches.sort(key=lambda x: x.score, reverse=True)

            # Limit to the overall top_k results
            top_matches = all_matches[:k]

            # Convert top Pinecone matches to Langchain Documents
            docs = []
            for match in top_matches:
                content = match.metadata.get('content', '') if match.metadata else ''
                if not content:
                    logger.warning(f"Match found (ID: {match.id}, Score: {match.score:.4f}) but 'content' key missing or empty in metadata.")
                    continue

                # Clean up metadata for Langchain Document - remove content to avoid duplication
                doc_metadata = {k: v for k, v in match.metadata.items() if k != 'content'}
                doc_metadata['score'] = match.score # Add score to metadata
                doc_metadata['vector_id'] = match.id # Add vector ID for reference
                doc_metadata['namespace'] = match.namespace if hasattr(match, 'namespace') else 'unknown' # Store namespace if available

                docs.append(Document(page_content=content, metadata=doc_metadata))

            logger.info(f"Retrieved and processed {len(docs)} relevant document contexts.")
            return docs

        except Exception as e:
            logger.error(f"Error during context retrieval: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    # get_contextual_summary method might be removed if not used, or updated to use the retrieved docs.
    # def get_contextual_summary(...) -> Optional[str]: ...