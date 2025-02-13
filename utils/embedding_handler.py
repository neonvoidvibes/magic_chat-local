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

from langchain_openai import OpenAIEmbeddings
from .document_handler import DocumentHandler
from .pinecone_utils import init_pinecone, create_or_verify_index

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingHandler:
    """Handles document embedding generation and storage."""
    
    def __init__(
        self,
        index_name: str = "chat-docs-index",
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
        if not init_pinecone():
            raise RuntimeError("Failed to initialize Pinecone")
            
        # Create or verify index
        self.index = create_or_verify_index(index_name)
        if not self.index:
            raise RuntimeError(f"Failed to create/verify index '{index_name}'")
            
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
            metadata: Metadata to attach to vectors
            batch_size: Number of vectors to upsert in each batch
            
        Returns:
            bool indicating success
        """
        try:
            # Process document into chunks
            chunks = self.doc_handler.process_document(content, metadata)
            if not chunks:
                return False
                
            # Prepare vectors for upserting
            vectors = []
            for chunk in chunks:
                # Generate embedding for chunk
                vector = self.generate_embedding(chunk['content'])
                if not vector:
                    continue
                    
                # Create vector record with standardized metadata field
                vector_id = f"{metadata.get('file_name', 'doc')}_{chunk['metadata']['chunk_index']}"
                vector_metadata = {
                    **chunk['metadata'],
                    'content': chunk['content'],  # Consistent content field for retrieval
                    'file_name': metadata.get('file_name', 'unknown'),  # Ensure file_name is present
                    'source': metadata.get('source', 'manual_upload')
                }
                vectors.append((vector_id, vector, vector_metadata))
                
            # Upsert vectors in batches
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=self.namespace)
                logger.info(f"Upserted batch of {len(batch)} vectors")
                
            logger.info(f"Successfully embedded and upserted {len(vectors)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error in embed_and_upsert: {e}")
            return False
            
    def delete_document(self, file_name: str) -> bool:
        """Delete all vectors for a document from the index.
        
        Args:
            file_name: Name of file to delete vectors for
            
        Returns:
            bool indicating success
        """
        try:
            # Delete vectors both by file_name filter and by vector IDs
            # First get all vectors matching the file name
            vector_ids = []
            filter_query = {"file_name": file_name}
            fetch_response = self.index.query(
                vector=[0] * 1536,  # Dummy vector for metadata-only query
                filter=filter_query,
                namespace=self.namespace,
                top_k=10000,  # Large enough to get all matches
                include_metadata=True
            )
            
            if fetch_response.matches:
                # Collect vector IDs
                vector_ids = [match.id for match in fetch_response.matches]
                
                # Delete by IDs for precise removal
                self.index.delete(
                    ids=vector_ids,
                    namespace=self.namespace
                )
                
                # Also delete by filter as backup
                self.index.delete(
                    filter=filter_query,
                    namespace=self.namespace
                )
                
            logger.info(f"Deleted {len(vector_ids)} vectors for document: {file_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document vectors: {e}")
            return False