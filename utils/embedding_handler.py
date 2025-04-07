"""Utilities for generating and managing document embeddings."""
import os
import sys
import logging
import re
import urllib.parse
import traceback
from typing import List, Optional, Dict, Any

# Attempt early tiktoken import
try: import tiktoken; logging.getLogger(__name__).debug("Imported tiktoken early.")
except Exception as e: logging.getLogger(__name__).warning(f"Early tiktoken import failed: {e}")

# Langchain imports
from langchain_openai import OpenAIEmbeddings
# Use RecursiveCharacterTextSplitter for more flexible splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Document object for metadata handling with splitter
from langchain_core.documents import Document

# Local imports
# from .document_handler import DocumentHandler # No longer needed, using splitter directly
from .pinecone_utils import init_pinecone, create_or_verify_index
import pinecone # Keep for potential direct use

# Configure logging
logger = logging.getLogger(__name__)

def sanitize_for_pinecone_id(input_string: str) -> str:
    """Sanitizes a string using URL encoding for Pinecone ID compliance."""
    sanitized = urllib.parse.quote_plus(input_string.encode('utf-8'))
    max_len = 512
    if len(sanitized) > max_len:
        logger.warning(f"Sanitized ID >{max_len} chars. Truncating: {input_string}")
        sanitized = sanitized[:max_len]
    if not sanitized:
        logger.error(f"Empty sanitized ID for: {input_string}"); return "sanitized_empty"
    return sanitized

class EmbeddingHandler:
    """Handles document embedding generation and storage."""

    def __init__(
        self,
        index_name: str = "magicchat",
        namespace: Optional[str] = None,
        # Add chunking parameters here if they need to be configurable per instance
        chunk_size: int = 1500, # Keep larger chunk size
        chunk_overlap: int = 150
    ):
        """Initialize embedding handler."""
        self.index_name = index_name
        self.namespace = namespace
        self.embedding_model_name = "text-embedding-ada-002"

        try:
            self.embeddings = OpenAIEmbeddings(model=self.embedding_model_name)
            logger.info(f"EmbeddingHandler: Initialized OpenAIEmbeddings model '{self.embedding_model_name}'.")
        except Exception as e:
            logger.error(f"EmbeddingHandler: Failed OpenAIEmbeddings init: {e}", exc_info=True)
            raise RuntimeError("Failed OpenAIEmbeddings init") from e

        # Initialize RecursiveCharacterTextSplitter directly
        # Prioritize splitting between Q&A sections, then common markdown/text separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False, # Treat separators literally for now
            separators=[
                "\n\n**Hur", # Try to split before each question first
                "\n\n",      # Then paragraphs
                "\n",        # Then lines
                " ",         # Then spaces
                ""           # Finally characters
                ]
        )
        logger.info(f"EmbeddingHandler: Initialized RecursiveCharacterTextSplitter (chunk={chunk_size}, overlap={chunk_overlap}).")

        self.pc = init_pinecone()
        if not self.pc: raise RuntimeError("Failed Pinecone init")
        self.index = create_or_verify_index(index_name)
        if not self.index: raise RuntimeError(f"Failed index access '{index_name}'")

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for text."""
        try:
            if not hasattr(self, 'embeddings') or self.embeddings is None: logger.error("Embeddings obj missing."); return None
            return self.embeddings.embed_query(text)
        except Exception as e: logger.error(f"Embedding generation error: {e}", exc_info=True); return None

    def embed_and_upsert(
        self,
        content: str,
        metadata: Dict[str, Any], # Base metadata passed in
        batch_size: int = 100
    ) -> bool:
        """Split, embed, and upsert document content."""
        try:
            original_file_name = metadata.get('file_name')
            if not original_file_name: logger.error("Missing 'file_name' in metadata."); return False

            # Use the text splitter directly
            # create_documents adds metadata to each doc automatically
            # We pass the base metadata which will be copied to each chunk
            documents = self.text_splitter.create_documents([content], metadatas=[metadata]) # Pass metadata once for the whole doc

            if not documents: logger.warning(f"No documents/chunks created for: {original_file_name}"); return False
            logger.info(f"Split {original_file_name} into {len(documents)} documents/chunks.")

            vectors_to_upsert = []
            embedding_failed_count = 0
            for i, doc in enumerate(documents):
                # Content is in doc.page_content
                # Metadata is in doc.metadata (includes base metadata + potentially splitter additions like start_index)
                chunk_content = doc.page_content
                chunk_metadata = doc.metadata

                vector = self.generate_embedding(chunk_content)
                if not vector:
                    logger.warning(f"Embedding failed for chunk {i} of {original_file_name}. Skipping.")
                    embedding_failed_count += 1
                    continue

                # Ensure file_name from metadata is used for ID generation
                file_name_for_id = chunk_metadata.get('file_name', original_file_name) # Fallback just in case
                sanitized_id_part = sanitize_for_pinecone_id(file_name_for_id)
                # Use simple index 'i' for uniqueness as splitter might not add chunk_index
                vector_id = f"{sanitized_id_part}_{i}"

                # Prepare metadata for Pinecone - Ensure required fields are present
                # The splitter copies the base metadata, we add the content itself
                pinecone_metadata = {
                    **chunk_metadata, # Copy all metadata from the split document
                    'content': chunk_content, # Explicitly add content field
                    'chunk_index': i, # Explicitly add chunk index
                    'total_chunks': len(documents) # Add total chunks for context
                }
                # Ensure agent_name is present if not added by splitter metadata copy
                if 'agent_name' not in pinecone_metadata:
                     pinecone_metadata['agent_name'] = metadata.get('agent_name', 'unknown')
                if 'source' not in pinecone_metadata:
                     pinecone_metadata['source'] = metadata.get('source', 'unknown')


                vectors_to_upsert.append((vector_id, vector, pinecone_metadata))

            if embedding_failed_count > 0: logger.warning(f"{embedding_failed_count} chunks failed embedding for {original_file_name}")
            if not vectors_to_upsert: logger.warning(f"No vectors to upsert for {original_file_name}"); return False

            logger.info(f"Upserting {len(vectors_to_upsert)} vectors for '{original_file_name}' (namespace: '{self.namespace}')...")
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                try:
                    upsert_response = self.index.upsert(vectors=batch, namespace=self.namespace)
                    logger.info(f"Upserted batch {i//batch_size + 1} ({len(batch)} vectors). Resp: {upsert_response}")
                except Exception as upsert_e:
                    logger.error(f"Pinecone upsert error (Batch {i//batch_size + 1}, File: '{original_file_name}'): {upsert_e}")
                    failing_ids = [item[0] for item in batch]; logger.error(f"Failing IDs (sample): {failing_ids[:5]}")
                    return False # Fail operation if batch fails

            logger.info(f"Successfully embedded/upserted {len(vectors_to_upsert)} vectors for '{original_file_name}'.")
            return True

        except Exception as e:
            logger.error(f"Error embedding/upserting file '{metadata.get('file_name', '?')}': {e}", exc_info=True)
            return False

    def delete_document(self, file_name: str) -> bool:
        """Delete all vectors associated with a specific file_name."""
        if not self.index: logger.error("Delete failed: Pinecone index missing."); return False
        try:
            logger.warning(f"Attempting delete vectors for file_name='{file_name}' in namespace '{self.namespace}'...")
            # It's often safer/simpler to delete by ID if possible, but requires fetching first.
            # Deleting by metadata filter is simpler but less direct.
            delete_response = self.index.delete(filter={"file_name": file_name}, namespace=self.namespace)
            logger.info(f"Delete by filter for file_name='{file_name}' completed. Resp: {delete_response}")
            return True
        except Exception as e: logger.error(f"Error deleting doc vectors for '{file_name}': {e}"); return False