"""Utilities for retrieving relevant document context for chat."""
import logging
from typing import List, Optional, Dict, Any
from langchain_openai import OpenAIEmbeddings
# We use the Pinecone client directly, so Langchain PineconeVectorStore is not strictly needed here
# from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_core.documents import Document # Use Document class from langchain core
from pinecone import Pinecone
from utils.pinecone_utils import init_pinecone
# from utils.document_handler import DocumentHandler # Not needed here

# Configure logging
# Ensure logger level is set appropriately elsewhere (e.g., main script with --debug)
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

        # Use the new Pinecone class-based usage
        pc = init_pinecone()
        if not pc:
            raise RuntimeError("Failed to initialize Pinecone")

        # Get the actual index object
        try:
            self.index = pc.Index(self.index_name)
            logger.info(f"RetrievalHandler: Successfully connected to Pinecone index '{self.index_name}'")
        except Exception as e:
            logger.error(f"RetrievalHandler: Failed to connect to Pinecone index '{self.index_name}': {e}")
            raise RuntimeError(f"Failed to connect to Pinecone index '{self.index_name}'") from e

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
        k = top_k or self.top_k
        logger.debug(f"RetrievalHandler: Attempting to retrieve top {k} contexts.")
        logger.debug(f"RetrievalHandler: Query: '{query[:100]}...' (is_transcript={is_transcript})")
        logger.debug(f"RetrievalHandler: Base namespace: {self.namespace}, Event ID: {self.event_id}")

        try:
            # Generate embedding for the user query
            try:
                query_embedding = self.embeddings.embed_query(query)
                logger.debug(f"RetrievalHandler: Generated query embedding vector (first 5 dims): {query_embedding[:5]}...")
            except Exception as e:
                logger.error(f"RetrievalHandler: Error generating query embedding: {e}")
                return [] # Cannot proceed without embedding

            # --- Define namespaces to search ---
            namespaces_to_search = [self.namespace] # Always search the base agent namespace
            logger.debug(f"RetrievalHandler: Initially searching namespace: {self.namespace}")
            event_namespace = f"{self.namespace}-{self.event_id}" if self.event_id and self.event_id != '0000' else None # Avoid using default '0000' as event ID here
            if event_namespace and event_namespace != self.namespace:
                 # If event_id exists and is specific, also search the event-specific namespace
                 logger.debug(f"RetrievalHandler: Adding event-specific namespace to search: {event_namespace}")
                 namespaces_to_search.append(event_namespace)
            else:
                 logger.debug(f"RetrievalHandler: No specific event namespace to search (event_id: {self.event_id})")


            # --- Perform queries with filters ---
            all_matches = []
            for ns in namespaces_to_search:
                # Build the metadata filter for this namespace query
                query_filter = {"agent_name": self.namespace} # Always filter based on agent_name stored in metadata

                # Add event_id filter *only if* event_id is present and specific
                # And only if we are querying the event-specific namespace? Or always if event_id exists? Let's try always.
                if self.event_id and self.event_id != '0000':
                     query_filter["event_id"] = self.event_id
                     logger.debug(f"RetrievalHandler: Added event_id='{self.event_id}' to filter.")
                else:
                     # Optionally, explicitly filter out things *with* an event_id if we *don't* have one?
                     # query_filter["event_id"] = {"$exists": False} # Or handle based on requirements
                     logger.debug("RetrievalHandler: No specific event_id provided or it's default '0000', not adding event_id to filter.")


                # Merge any additional filters passed externally
                if filter_metadata:
                    logger.debug(f"RetrievalHandler: Merging external filters: {filter_metadata}")
                    query_filter.update(filter_metadata)

                logger.debug(f"RetrievalHandler: Querying namespace='{ns}' with filter={query_filter}, top_k={k}")

                try:
                    response = self.index.query(
                        vector=query_embedding,
                        top_k=k,
                        namespace=ns,
                        filter=query_filter, # Apply the constructed filter
                        include_metadata=True
                    )
                    logger.debug(f"RetrievalHandler: Raw response from Pinecone namespace '{ns}': {response}") # Log the whole response object
                    if response.matches:
                         logger.info(f"RetrievalHandler: Found {len(response.matches)} raw matches in namespace '{ns}'.")
                         # Log details of top N matches for inspection
                         for i, match in enumerate(response.matches[:3]): # Log top 3
                              logger.debug(f"  Match {i+1}: ID={match.id}, Score={match.score:.4f}, Metadata={match.metadata}")
                         all_matches.extend(response.matches)
                    else:
                         logger.info(f"RetrievalHandler: No matches found in namespace '{ns}'.")

                except Exception as query_e:
                    logger.error(f"RetrievalHandler: Error querying Pinecone namespace '{ns}': {query_e}")
                    import traceback
                    logger.error(traceback.format_exc()) # Log full traceback for query errors
                    # Continue to next namespace if one fails

            if not all_matches:
                logger.warning("RetrievalHandler: No relevant context matches found across all searched namespaces.")
                return []

            # --- Process and Rank Results ---
            logger.debug(f"RetrievalHandler: Total raw matches found: {len(all_matches)}")
            all_matches.sort(key=lambda x: x.score, reverse=True)
            top_matches = all_matches[:k]
            logger.debug(f"RetrievalHandler: Top {len(top_matches)} matches after sorting:")
            for i, match in enumerate(top_matches):
                 logger.debug(f"  Rank {i+1}: ID={match.id}, Score={match.score:.4f}")


            # Convert top Pinecone matches to Langchain Documents
            docs = []
            for match in top_matches:
                if not match.metadata:
                     logger.warning(f"RetrievalHandler: Match (ID: {match.id}, Score: {match.score:.4f}) has no metadata. Skipping.")
                     continue

                content = match.metadata.get('content')
                if not content:
                    logger.warning(f"RetrievalHandler: Match (ID: {match.id}, Score: {match.score:.4f}) metadata lacks 'content' key or value is empty. Metadata: {match.metadata}. Skipping.")
                    continue

                # Prepare metadata for Langchain Document
                doc_metadata = {k: v for k, v in match.metadata.items() if k != 'content'} # Exclude content
                doc_metadata['score'] = match.score
                doc_metadata['vector_id'] = match.id
                # Add namespace if available (Pinecone SDK might not always include it in match object directly?)
                # Let's assume it was added during query or is derivable if needed.

                docs.append(Document(page_content=content, metadata=doc_metadata))

            logger.info(f"RetrievalHandler: Returning {len(docs)} processed document contexts.")
            if docs:
                 logger.debug(f"RetrievalHandler: First returned document metadata: {docs[0].metadata}")

            return docs

        except Exception as e:
            logger.error(f"RetrievalHandler: Unhandled error during context retrieval: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    # get_contextual_summary method might be removed if not used, or updated to use the retrieved docs.
    # def get_contextual_summary(...) -> Optional[str]: ...