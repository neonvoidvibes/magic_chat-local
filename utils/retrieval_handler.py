"""Utilities for retrieving relevant document context for chat."""
import logging
from typing import List, Optional, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain.schema import Document
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
        self.namespace = agent_name # Use agent_name directly as the namespace
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

        # Use agent_name directly as the namespace (matches cli_embed.py)
        namespace = self.namespace # Use agent_name passed as self.namespace directly

        # Create the LangChain vector store
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            text_key="content",
            namespace=namespace # Use the simplified namespace
        )

        # We'll use a standard "similarity" search retriever
        # Note: We use the raw index.query below for more control over logging/filtering
        # self.retriever = self.vectorstore.as_retriever(
        #     search_type="similarity",
        #     search_kwargs={"k": self.top_k}
        # )

    def get_relevant_context(
        self,
        query: str,
        filter_metadata: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        is_transcript: bool = False # This argument seems less relevant now we query single namespace
    ) -> List[Document]: # Return Langchain Document objects directly
        """
        Retrieve the most relevant chunks for `query` from the agent's namespace.
        Returns clear source attribution with each result.

        Args:
            query: Search query
            filter_metadata: Additional metadata filters
            top_k: Number of results to return

        Returns:
            List of Langchain Document objects containing relevant context.
        """
        try:
            logging.info(f"Retrieving context for query: {query}")
            # logging.info(f"Is transcript query: {is_transcript}") # Less relevant now

            # Build metadata filter - only use essential fields
            base_filter = {}
            # First try without filters to see what's in the index
            try:
                stats = self.index.describe_index_stats(namespace=self.namespace) # Check specific namespace
                logging.debug(f"Index stats for namespace '{self.namespace}': {stats}")
            except Exception as e:
                logging.warning(f"Could not get index stats for namespace '{self.namespace}': {e}")

            # Try to match on file name or source if filters are provided
            if filter_metadata:
                if 'file_name' in filter_metadata:
                     # Make sure to filter on the *original* filename stored in metadata
                    base_filter['filename'] = filter_metadata['file_name']
                if 'source' in filter_metadata:
                    base_filter['source'] = filter_metadata['source']
                # Add other potential filters from metadata if needed
                if 'agent_name' in filter_metadata:
                     # This might be redundant if using namespace, but potentially useful
                    base_filter['agent_name'] = filter_metadata['agent_name']
                if 'event_id' in filter_metadata:
                    base_filter['event_id'] = filter_metadata['event_id']


            logging.debug(f"Using combined metadata filter: {base_filter}") # Changed from info to debug

            # Get embedding for query
            try:
                logging.debug(f"Generating embedding for query: '{query[:100]}...'")
                query_embedding = self.embeddings.embed_query(query)
                logging.debug(f"Generated query embedding vector, length: {len(query_embedding)}")
            except Exception as e:
                logging.error(f"Error generating query embedding: {e}")
                raise

            # Search in the agent's namespace
            docs = []
            current_namespace = self.namespace # The direct agent name namespace
            logging.debug(f"Querying Pinecone index '{self.index_name}' in namespace '{current_namespace}' with top_k={top_k or self.top_k}")
            logging.debug(f"Applying metadata filter: {base_filter if base_filter else 'None'}")

            try:
                # Search in the agent's namespace
                response = self.index.query(
                    vector=query_embedding,
                    top_k=top_k or self.top_k,
                    namespace=current_namespace, # self.namespace now holds the direct agent name
                    filter=base_filter if base_filter else None, # Apply filters here
                    include_metadata=True
                )
                logging.debug(f"Pinecone query successful. Raw matches found in namespace '{current_namespace}': {len(response.matches)}")

                # Convert matches to Documents
                for match in response.matches:
                    # Ensure metadata is fetched correctly
                    match_metadata = match.metadata if hasattr(match, 'metadata') else {}
                    if match_metadata is None: # Handle case where metadata might explicitly be None
                        match_metadata = {}

                    content = match_metadata.get('content', '') # Get content from metadata as stored by embed_handler
                    if not content and hasattr(match, 'page_content'): # Fallback if content wasn't in metadata
                         content = match.page_content

                    metadata = {
                        'score': match.score,
                        'file_name': match_metadata.get('filename', match_metadata.get('file_name', 'unknown')), # Check both keys
                        'source': match_metadata.get('source', 'unknown'),
                        'chunk_index': match_metadata.get('chunk_index', -1),
                        'event_id': match_metadata.get('event_id', None),
                        # Add any other relevant metadata fields you stored
                        'namespace': current_namespace # Use the queried namespace
                    }
                    # Prepare content snippet outside the f-string to avoid backslash issue
                    content_snippet = content[:60].replace('\n', ' ') + "..."
                    logging.debug(f"  Match: ID='{match.id}', Score={match.score:.3f}, Filename='{metadata.get('file_name')}', Content Snippet='{content_snippet}'")
                    docs.append(Document(page_content=content, metadata=metadata))

                # Sort results by score
                docs.sort(key=lambda x: x.metadata['score'], reverse=True)

                # Limit to top_k results if specified
                if top_k:
                    docs = docs[:top_k]

                logging.info(f"Found {len(docs)} documents in namespace '{current_namespace}'")
                return docs # Return the directly queried docs

            # This block should now be correctly indented relative to the try block above
            except Exception as e:
                logging.error(f"Error querying Pinecone namespace '{current_namespace}': {e}")
                return [] # Return empty list on query error


        # This block should be correctly indented relative to the main try block
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []

    # get_contextual_summary might need adjustments if used, as it expects a different format now
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
            docs = self.get_relevant_context(query) # Now returns Document objects
            if not docs:
                return None

            # Extract page_content from Document objects
            combined_text = "\n\n".join(doc.page_content for doc in docs)
            # Just truncating for now
            return combined_text[: max_tokens * 4]

        except Exception as e:
            logger.error(f"Error generating context summary: {e}")
            return None