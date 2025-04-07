"""Utilities for generating and managing document embeddings."""
import os
import sys
try:
    from langchain_core.language_models.chat_models import LangSmithParams
except ImportError:
    # Define a dummy LangSmithParams to bypass the import error.
    class LangSmithParams:
        pass
    # Inject the dummy into the module namespace so that subsequent imports work.
    import langchain_core.language_models.chat_models as chat_models
    chat_models.LangSmithParams = LangSmithParams
import logging
import re
import urllib.parse # Import url encoding module
from typing import List, Optional, Dict, Any
import pinecone
try:
    from langchain_core import utils as lc_utils
    if not hasattr(lc_utils, "from_env"):
         lc_utils.from_env = lambda key, default=None: default
    if not hasattr(lc_utils, "secret_from_env"):
         lc_utils.secret_from_env = lambda key, default=None: default
except Exception as e:
    pass
# Pre-register gpt2 encoding to avoid duplicate registration errors in tiktoken
# import tiktoken # Import removed, as direct usage isn't needed here
# try:
#     _ = tiktoken.encoding_for_model("gpt2") # Pre-registration removed, rely on langchain/openai handling
# except Exception as e:
#     import logging
#     logging.warning(f"Pre-registration of gpt2 encoding failed: {e}")

from langchain_openai import OpenAIEmbeddings
from .document_handler import DocumentHandler
from .pinecone_utils import init_pinecone, create_or_verify_index
import traceback # Import traceback for detailed error logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sanitize_for_pinecone_id(input_string: str) -> str:
    """Sanitizes a string using URL encoding (percent-encoding) for Pinecone ID compliance."""
    sanitized = urllib.parse.quote_plus(input_string.encode('utf-8'))
    max_len = 512
    if len(sanitized) > max_len:
        logger.warning(f"Sanitized ID exceeded max length ({max_len}). Truncating: {input_string}")
        sanitized = sanitized[:max_len]
    if not sanitized:
        logger.error(f"Filename resulted in empty sanitized ID: {input_string}")
        return "sanitized_empty_filename"
    return sanitized


class EmbeddingHandler:
    """Handles document embedding generation and storage."""

    def __init__(
        self,
        index_name: str = "magicchat",
        namespace: Optional[str] = None
    ):
        """Initialize embedding handler with Pinecone configuration.

        Args:
            index_name: Name of Pinecone index to use
            namespace: Optional namespace within index
        """
        self.index_name = index_name
        self.namespace = namespace
        self.embedding_model_name = "text-embedding-ada-002" # Define model name

        # Initialize OpenAI embeddings, explicitly specifying the model
        try:
            self.embeddings = OpenAIEmbeddings(model=self.embedding_model_name)
            logger.info(f"EmbeddingHandler: Initialized OpenAIEmbeddings with model '{self.embedding_model_name}'.")
        except Exception as e:
            logger.error(f"EmbeddingHandler: Failed to initialize OpenAIEmbeddings: {e}", exc_info=True)
            raise RuntimeError("Failed to initialize OpenAIEmbeddings") from e


        # Initialize document handler with default settings
        self.doc_handler = DocumentHandler()

        # Initialize Pinecone
        self.pc = init_pinecone()
        if not self.pc:
            raise RuntimeError("Failed to initialize Pinecone")

        # Create or verify index
        self.index = create_or_verify_index(index_name)
        if not self.index:
            raise RuntimeError(f"Failed to get or create index '{index_name}'")

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for text using OpenAI.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            # Check if embeddings object exists
            if not hasattr(self, 'embeddings') or self.embeddings is None:
                 logger.error("EmbeddingHandler: OpenAIEmbeddings object not initialized.")
                 return None
            vector = self.embeddings.embed_query(text)
            return vector
        except Exception as e:
            # Catch the specific tiktoken error if possible, or general errors
            logger.error(f"EmbeddingHandler: Error generating embedding: {e}", exc_info=True)
            # Log the type of exception
            logger.error(f"Exception type: {type(e)}")
            return None

    def embed_and_upsert(
        self,
        content: str,
        metadata: Dict[str, Any],
        batch_size: int = 100
    ) -> bool:
        """Generate embeddings for document chunks and upsert to Pinecone.

        Args:
            content: Document content to process
            metadata: Metadata to attach to vectors (must include 'file_name')
            batch_size: Number of vectors to upsert in each batch

        Returns:
            bool indicating success
        """
        try:
            original_file_name = metadata.get('file_name')
            if not original_file_name:
                logger.error("Metadata must include 'file_name' for upserting.")
                return False

            chunks = self.doc_handler.process_document(content, metadata)
            if not chunks:
                logger.warning(f"No chunks generated for file: {original_file_name}")
                return False

            vectors_to_upsert = []
            for chunk in chunks:
                vector = self.generate_embedding(chunk['content'])
                if not vector:
                    logger.warning(f"Failed to generate embedding for a chunk in {original_file_name}, skipping chunk.")
                    continue # Skip this chunk if embedding failed

                sanitized_file_name_for_id = sanitize_for_pinecone_id(original_file_name)
                chunk_index = chunk['metadata'].get('chunk_index', 'unknown')
                vector_id = f"{sanitized_file_name_for_id}_{chunk_index}"

                vector_metadata = {
                    **chunk['metadata'],
                    'content': chunk['content'],
                    'file_name': original_file_name,
                    'source': metadata.get('source', 'unknown'),
                    'agent_name': metadata.get('agent_name', 'unknown')
                }
                vectors_to_upsert.append((vector_id, vector, vector_metadata))

            if not vectors_to_upsert:
                 logger.warning(f"No vectors generated or prepared for upsert for file: {original_file_name}")
                 # If chunks existed but all embeddings failed, maybe return False?
                 return False # Treat as failure if no vectors could be prepared

            logger.info(f"Preparing to upsert {len(vectors_to_upsert)} vectors for '{original_file_name}' in namespace '{self.namespace}'...")
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                try:
                    upsert_response = self.index.upsert(vectors=batch, namespace=self.namespace)
                    logger.info(f"Upserted batch {i//batch_size + 1} ({len(batch)} vectors). Response: {upsert_response}")
                except Exception as upsert_e:
                    logger.error(f"Error during Pinecone upsert for batch {i//batch_size + 1} of '{original_file_name}': {upsert_e}")
                    failing_ids = [item[0] for item in batch]
                    logger.error(f"Failing batch IDs (first few): {failing_ids[:5]}")
                    return False

            logger.info(f"Successfully embedded and upserted {len(vectors_to_upsert)} vectors for '{original_file_name}'")
            return True

        except Exception as e:
            logger.error(f"Error in embed_and_upsert for file '{metadata.get('file_name', 'unknown')}': {e}")
            logger.error(traceback.format_exc())
            return False

    def delete_document(self, file_name: str) -> bool:
        """Delete all vectors associated with a specific file_name from the index."""
        if not self.index:
             logger.error("Cannot delete document, Pinecone index not initialized.")
             return False
        try:
            logger.warning(f"Attempting to delete vectors for document with file_name='{file_name}' in namespace '{self.namespace}'...")
            delete_response = self.index.delete(
                filter={"file_name": file_name},
                namespace=self.namespace
            )
            logger.info(f"Delete operation by filter for file_name='{file_name}' completed. Response: {delete_response}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document vectors for file_name='{file_name}': {e}")
            return False