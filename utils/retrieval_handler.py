"""Utilities for retrieving relevant document context for chat."""
import logging
import traceback
from typing import List, Optional, Dict, Any

# Attempt early tiktoken import
try: import tiktoken; logging.getLogger(__name__).debug("Imported tiktoken early.")
except Exception as e: logging.getLogger(__name__).warning(f"Early tiktoken import failed: {e}")

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone
from utils.pinecone_utils import init_pinecone
from anthropic import Anthropic # Import Anthropic client

logger = logging.getLogger(__name__)

# Define a default transform prompt
DEFAULT_QUERY_TRANSFORM_PROMPT = """Rewrite the following user query to be more effective for searching a vector database containing document chunks. Focus on extracting key entities (people, projects, organizations), topics, dates, and the core question intent. Output only the rewritten query, no preamble.

User Query: '{user_query}'

Rewritten Query:"""


class RetrievalHandler:
    """Handles document retrieval using direct Pinecone queries."""

    def __init__(
        self,
        index_name: str = "magicchat",
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        event_id: Optional[str] = None,
        top_k: int = 10, # Keep moderate top_k for now
        anthropic_client: Optional[Anthropic] = None # Expect client instance
    ):
        """Initialize retrieval handler."""
        if not agent_name: raise ValueError("agent_name required")
        if not anthropic_client: raise ValueError("anthropic_client required for query transformation")

        self.index_name = index_name
        self.namespace = agent_name
        self.session_id = session_id
        self.event_id = event_id if event_id and event_id != '0000' else None
        self.top_k = top_k
        self.embedding_model_name = "text-embedding-ada-002"
        self.anthropic_client = anthropic_client # Store client instance

        try:
            self.embeddings = OpenAIEmbeddings(model=self.embedding_model_name)
            logger.info(f"Retriever: Initialized Embeddings model '{self.embedding_model_name}'.")
        except Exception as e: raise RuntimeError("Failed Embeddings init") from e

        pc = init_pinecone();
        if not pc: raise RuntimeError("Failed Pinecone init")
        try:
            self.index = pc.Index(self.index_name)
            logger.info(f"Retriever: Connected to index '{self.index_name}'. Default top_k={self.top_k}")
        except Exception as e: raise RuntimeError(f"Failed connection to index '{self.index_name}'") from e

    def _transform_query(self, query: str) -> str:
        """Uses LLM to rewrite the query for better vector search."""
        logger.debug(f"Transforming query: '{query}'")
        try:
            # Use a smaller/faster model for transformation if available and cost-effective
            # model_for_transform = "claude-3-haiku-20240307"
            # For now, use the main client's default or passed model if needed
            # Note: This adds an extra LLM call.

            prompt = DEFAULT_QUERY_TRANSFORM_PROMPT.format(user_query=query)

            # Using non-streaming call for simplicity here
            message = self.anthropic_client.messages.create(
                 # Consider using a faster/cheaper model if possible for this task
                 model="claude-3-haiku-20240307",
                 max_tokens=100, # Should be short
                 messages=[{"role": "user", "content": prompt}]
            )
            transformed_query = message.content[0].text.strip()
            logger.debug(f"Transformed query: '{transformed_query}'")
            # Basic validation: if empty or just punctuation, return original
            if not transformed_query or transformed_query in ['.', '?', '!']:
                 logger.warning("Query transformation resulted in empty/trivial output. Using original.")
                 return query
            return transformed_query
        except Exception as e:
            logger.error(f"Error transforming query: {e}. Using original query.")
            return query # Fallback to original query on error

    def get_relevant_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        is_transcript: bool = False
    ) -> List[Document]:
        """Retrieve relevant document chunks, applying query transformation."""
        k = top_k or self.top_k
        logger.debug(f"Retriever: Original query: '{query[:100]}...' (is_tx={is_transcript})")

        # 1. Transform the query
        transformed_query = self._transform_query(query)
        if transformed_query != query:
            logger.info(f"Retriever: Using transformed query: '{transformed_query[:100]}...'")
        else:
            logger.info("Retriever: Using original query (transformation failed or unchanged).")

        logger.debug(f"Retriever: Attempting retrieve top {k}. Base ns: {self.namespace}, Event ID filter: {self.event_id}")

        try:
            # 2. Generate embedding for the (potentially transformed) query
            try:
                if not hasattr(self, 'embeddings'): raise RuntimeError("Embeddings missing")
                # Embed the transformed query
                query_embedding = self.embeddings.embed_query(transformed_query)
                logger.debug(f"Retriever: Query embedding generated (first 5): {query_embedding[:5]}...")
            except Exception as e: logger.error(f"Retriever: Embedding error: {e}", exc_info=True); return []

            # 3. Define namespaces & filters (no change here)
            namespaces = [self.namespace]
            event_ns = f"{self.namespace}-{self.event_id}" if self.event_id else None
            if event_ns and event_ns != self.namespace: namespaces.append(event_ns); logger.debug(f"Adding event ns: {event_ns}")

            # 4. Perform queries (no change here)
            all_matches = []
            for ns in namespaces:
                query_filter = {"agent_name": self.namespace}
                if self.event_id: query_filter["event_id"] = self.event_id; logger.debug(f"Adding event_id filter: {self.event_id}")
                else: logger.debug("No specific event_id filter applied.")
                logger.debug(f"Querying ns='{ns}', filter={query_filter}, top_k={k}")
                try:
                    response = self.index.query(vector=query_embedding, top_k=k, namespace=ns, filter=query_filter, include_metadata=True)
                    logger.debug(f"Raw response ns '{ns}': {response}")
                    if response.matches: logger.info(f"Found {len(response.matches)} matches in ns '{ns}'."); all_matches.extend(response.matches)
                    else: logger.info(f"No matches in ns '{ns}'.")
                except Exception as query_e: logger.error(f"Pinecone query error ns '{ns}': {query_e}", exc_info=True)

            if not all_matches: logger.warning("No matches found across namespaces."); return []

            # 5. Process & Rank (no change here)
            logger.debug(f"Total raw matches: {len(all_matches)}")
            all_matches.sort(key=lambda x: x.score, reverse=True)
            top_matches = all_matches[:k]
            logger.debug(f"Top {len(top_matches)} matches after sort:")
            for i, match in enumerate(top_matches): logger.debug(f"  Rank {i+1}: ID={match.id}, Score={match.score:.4f}")

            # 6. Convert to Documents (no change here)
            docs = []
            for match in top_matches:
                if not match.metadata: logger.warning(f"Match {match.id} lacks metadata."); continue
                content = match.metadata.get('content')
                if not content: logger.warning(f"Match {match.id} metadata lacks 'content'."); continue
                doc_metadata = {k: v for k, v in match.metadata.items() if k != 'content'}
                doc_metadata['score'] = match.score; doc_metadata['vector_id'] = match.id
                docs.append(Document(page_content=content, metadata=doc_metadata))

            logger.info(f"Retriever: Returning {len(docs)} processed contexts.")
            if docs: logger.debug(f"First doc metadata: {docs[0].metadata}")
            return docs

        except Exception as e: logger.error(f"Context retrieval error: {e}", exc_info=True); return []