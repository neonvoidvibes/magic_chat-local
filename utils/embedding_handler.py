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
import tiktoken
try:
    _ = tiktoken.encoding_for_model("gpt2")
except Exception as e:
    import logging
    logging.warning(f"Pre-registration of gpt2 encoding failed: {e}")

from langchain_openai import OpenAIEmbeddings
from .document_handler import DocumentHandler
from .pinecone_utils import init_pinecone, create_or_verify_index

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sanitize_for_pinecone_id(input_string: str) -> str:
    """Sanitizes a string using URL encoding (percent-encoding) for Pinecone ID compliance."""
    # URL encode the string using UTF-8. quote_plus replaces spaces with '+'.
    # This converts non-ASCII and reserved characters into %XX format.
    sanitized = urllib.parse.quote_plus(input_string.encode('utf-8'))

    # Pinecone IDs have length limits (e.g., 512 chars)
    max_len = 512
    if len(sanitized) > max_len:
        # Truncate *after* encoding
        logger.warning(f"Sanitized ID exceeded max length ({max_len}). Truncating: {input_string}")
        sanitized = sanitized[:max_len]
        # We could add more logic here to avoid cutting mid-% encoding, but it's rare with reasonable filenames.

    # Ensure it's not empty after sanitization (highly unlikely with quote_plus)
    if not sanitized:
        logger.error(f"Filename resulted in empty sanitized ID: {input_string}")
        return "sanitized_empty_filename" # Fallback ID
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

        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings()

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
            vector = self.embeddings.embed_query(text)
            return vector
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
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

            # Process document into chunks
            # Pass the original metadata, it will be attached to each chunk by process_document
            chunks = self.doc_handler.process_document(content, metadata)
            if not chunks:
                logger.warning(f"No chunks generated for file: {original_file_name}")
                return False

            # Prepare vectors for upserting
            vectors_to_upsert = []
            for chunk in chunks:
                # Generate embedding for chunk content
                vector = self.generate_embedding(chunk['content'])
                if not vector:
                    logger.warning(f"Failed to generate embedding for a chunk in {original_file_name}")
                    continue # Skip this chunk

                # Sanitize the original filename specifically for the ID using URL encoding
                sanitized_file_name_for_id = sanitize_for_pinecone_id(original_file_name)
                chunk_index = chunk['metadata'].get('chunk_index', 'unknown')
                vector_id = f"{sanitized_file_name_for_id}_{chunk_index}"

                # Prepare metadata for the vector in Pinecone
                # This should include the original filename and the chunk content
                vector_metadata = {
                    **chunk['metadata'], # Includes original metadata like file_name, agent_name, etc. plus chunk_index, total_chunks
                    'content': chunk['content'],  # Add chunk content to metadata for retrieval
                    # Ensure essential fields from original metadata are present
                    'file_name': original_file_name, # Keep original name in metadata
                    'source': metadata.get('source', 'unknown'),
                    'agent_name': metadata.get('agent_name', 'unknown')
                }
                vectors_to_upsert.append((vector_id, vector, vector_metadata))

            if not vectors_to_upsert:
                 logger.warning(f"No vectors generated or prepared for upsert for file: {original_file_name}")
                 return False

            # Upsert vectors in batches
            logger.info(f"Preparing to upsert {len(vectors_to_upsert)} vectors for '{original_file_name}' in namespace '{self.namespace}'...")
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                try:
                    upsert_response = self.index.upsert(vectors=batch, namespace=self.namespace)
                    logger.info(f"Upserted batch {i//batch_size + 1} ({len(batch)} vectors). Response: {upsert_response}")
                except Exception as upsert_e:
                    logger.error(f"Error during Pinecone upsert for batch {i//batch_size + 1} of '{original_file_name}': {upsert_e}")
                    # Log the failing IDs for inspection if possible
                    failing_ids = [item[0] for item in batch]
                    logger.error(f"Failing batch IDs (first few): {failing_ids[:5]}")
                    return False # Fail the entire operation if one batch fails

            logger.info(f"Successfully embedded and upserted {len(vectors_to_upsert)} vectors for '{original_file_name}'")
            return True

        except Exception as e:
            logger.error(f"Error in embed_and_upsert for file '{metadata.get('file_name', 'unknown')}': {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def delete_document(self, file_name: str) -> bool:
        """Delete all vectors associated with a specific file_name from the index.

        Args:
            file_name: The exact 'file_name' metadata value to filter and delete by.

        Returns:
            bool indicating success or if no vectors were found.
        """
        if not self.index:
             logger.error("Cannot delete document, Pinecone index not initialized.")
             return False
        try:
            logger.warning(f"Attempting to delete vectors for document with file_name='{file_name}' in namespace '{self.namespace}'...")
            # Use the metadata filter to delete vectors associated with the file_name
            delete_response = self.index.delete(
                filter={"file_name": file_name},
                namespace=self.namespace
            )
            logger.info(f"Delete operation by filter for file_name='{file_name}' completed. Response: {delete_response}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document vectors for file_name='{file_name}': {e}")
            return False