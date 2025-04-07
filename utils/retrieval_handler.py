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

logger = logging.getLogger(__name__)

class RetrievalHandler:
    """Handles document retrieval using direct Pinecone queries."""

    def __init__(
        self,
        index_name: str = "magicchat",
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        event_id: Optional[str] = None,
        top_k: int = 10 # Increased default top_k
    ):
        """Initialize retrieval handler."""
        if not agent_name: raise ValueError("agent_name required")

        self.index_name = index_name
        self.namespace = agent_name
        self.session_id = session_id
        self.event_id = event_id if event_id and event_id != '0000' else None
        self.top_k = top_k # Use the provided or default value
        self.embedding_model_name = "text-embedding-ada-002"

        try:
            self.embeddings = OpenAIEmbeddings(model=self.embedding_model_name)
            logger.info(f"Retriever: Initialized Embeddings model '{self.embedding_model_name}'.")
        except Exception as e:
            logger.error(f"Retriever: Failed Embeddings init: {e}", exc_info=True)
            raise RuntimeError("Failed Embeddings init") from e

        pc = init_pinecone()
        if not pc: raise RuntimeError("Failed Pinecone init")

        try:
            self.index = pc.Index(self.index_name)
            logger.info(f"Retriever: Connected to Pinecone index '{self.index_name}'. Default top_k={self.top_k}")
        except Exception as e:
            logger.error(f"Retriever: Failed connection to index '{self.index_name}': {e}")
            raise RuntimeError(f"Failed connection to index '{self.index_name}'") from e

    def get_relevant_context(
        self,
        query: str,
        top_k: Optional[int] = None, # Allow overriding default k per query
        is_transcript: bool = False
    ) -> List[Document]:
        """Retrieve relevant document chunks."""
        k = top_k or self.top_k # Use per-query k or instance default
        logger.debug(f"Retriever: Attempting retrieve top {k}. Query: '{query[:100]}...' (is_tx={is_transcript})")
        logger.debug(f"Retriever: Base ns: {self.namespace}, Event ID for filter: {self.event_id}")

        try:
            # Generate embedding
            try:
                if not hasattr(self, 'embeddings') or self.embeddings is None: raise RuntimeError("Embeddings missing")
                query_embedding = self.embeddings.embed_query(query)
                logger.debug(f"Retriever: Query embedding generated (first 5): {query_embedding[:5]}...")
            except Exception as e: logger.error(f"Retriever: Embedding error: {e}", exc_info=True); return []

            # Define namespaces & filters
            namespaces = [self.namespace]
            event_ns = f"{self.namespace}-{self.event_id}" if self.event_id else None
            if event_ns and event_ns != self.namespace: namespaces.append(event_ns); logger.debug(f"Adding event ns: {event_ns}")

            # Perform queries
            all_matches = []
            for ns in namespaces:
                query_filter = {"agent_name": self.namespace} # Always filter by agent
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

            # Process & Rank
            logger.debug(f"Total raw matches: {len(all_matches)}")
            all_matches.sort(key=lambda x: x.score, reverse=True)
            top_matches = all_matches[:k] # Limit to overall k
            logger.debug(f"Top {len(top_matches)} matches after sort:")
            for i, match in enumerate(top_matches): logger.debug(f"  Rank {i+1}: ID={match.id}, Score={match.score:.4f}")

            # Convert to Documents
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