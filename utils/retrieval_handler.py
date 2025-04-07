"""Utilities for retrieving relevant document context for chat."""
import logging
from typing import List, Optional, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document # Use Document class from langchain core
from pinecone import Pinecone
from utils.pinecone_utils import init_pinecone
import traceback

# Configure logging
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
        self.event_id = event_id if event_id and event_id != '0000' else None
        self.top_k = top_k
        self.embedding_model_name = "text-embedding-ada-002" # Define model name

        # Initialize embeddings, explicitly specifying the model
        try:
            self.embeddings = OpenAIEmbeddings(model=self.embedding_model_name)
            logger.info(f"RetrievalHandler: Initialized OpenAIEmbeddings with model '{self.embedding_model_name}'.")
        except Exception as e:
            logger.error(f"RetrievalHandler: Failed to initialize OpenAIEmbeddings: {e}", exc_info=True)
            raise RuntimeError("Failed to initialize OpenAIEmbeddings") from e

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
        top_k: Optional[int] = None,
        is_transcript: bool = False
    ) -> List[Document]:
        """
        Retrieve the most relevant document chunks for `query` using direct Pinecone query
        with appropriate namespace and metadata filters based on handler's agent_name and event_id.
        """
        k = top_k or self.top_k
        logger.debug(f"RetrievalHandler: Attempting to retrieve top {k} contexts.")
        logger.debug(f"RetrievalHandler: Query: '{query[:100]}...' (is_transcript={is_transcript})")
        logger.debug(f"RetrievalHandler: Base namespace: {self.namespace}, Specific Event ID for filtering: {self.event_id}")

        try:
            # Generate embedding for the user query
            try:
                # Check if embeddings object exists
                if not hasattr(self, 'embeddings') or self.embeddings is None:
                     logger.error("RetrievalHandler: OpenAIEmbeddings object not initialized.")
                     return []
                query_embedding = self.embeddings.embed_query(query)
                logger.debug(f"RetrievalHandler: Generated query embedding vector (first 5 dims): {query_embedding[:5]}...")
            except Exception as e:
                logger.error(f"RetrievalHandler: Error generating query embedding: {e}", exc_info=True)
                return []

            # --- Define namespaces to search ---
            namespaces_to_search = [self.namespace]
            logger.debug(f"RetrievalHandler: Primary namespace to search: {self.namespace}")
            event_namespace = f"{self.namespace}-{self.event_id}" if self.event_id else None
            if event_namespace and event_namespace != self.namespace:
                 logger.debug(f"RetrievalHandler: Adding event-specific namespace to search: {event_namespace}")
                 namespaces_to_search.append(event_namespace)

            # --- Perform queries with filters ---
            all_matches = []
            for ns in namespaces_to_search:
                query_filter = {"agent_name": self.namespace}
                if self.event_id:
                     query_filter["event_id"] = self.event_id
                     logger.debug(f"RetrievalHandler: Added specific event_id='{self.event_id}' to filter.")
                else:
                     logger.debug("RetrievalHandler: No specific event_id set, not adding event_id to filter.")

                logger.debug(f"RetrievalHandler: Querying namespace='{ns}' with filter={query_filter}, top_k={k}")

                try:
                    response = self.index.query(
                        vector=query_embedding,
                        top_k=k,
                        namespace=ns,
                        filter=query_filter,
                        include_metadata=True
                    )
                    logger.debug(f"RetrievalHandler: Raw response from Pinecone namespace '{ns}': {response}")
                    if response.matches:
                         logger.info(f"RetrievalHandler: Found {len(response.matches)} raw matches in namespace '{ns}'.")
                         for i, match in enumerate(response.matches[:3]):
                              logger.debug(f"  Match {i+1}: ID={match.id}, Score={match.score:.4f}, Metadata={match.metadata}")
                         all_matches.extend(response.matches)
                    else:
                         logger.info(f"RetrievalHandler: No matches found in namespace '{ns}'.")
                except Exception as query_e:
                    logger.error(f"RetrievalHandler: Error querying Pinecone namespace '{ns}': {query_e}")
                    logger.error(traceback.format_exc())

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
                doc_metadata = {k: v for k, v in match.metadata.items() if k != 'content'}
                doc_metadata['score'] = match.score
                doc_metadata['vector_id'] = match.id
                docs.append(Document(page_content=content, metadata=doc_metadata))

            logger.info(f"RetrievalHandler: Returning {len(docs)} processed document contexts.")
            if docs:
                 logger.debug(f"RetrievalHandler: First returned document metadata: {docs[0].metadata}")
            return docs

        except Exception as e:
            logger.error(f"RetrievalHandler: Unhandled error during context retrieval: {e}")
            logger.error(traceback.format_exc())
            return []