"""Utilities for generating and managing document embeddings."""
import os
import logging
from typing import List, Optional, Dict, Any
import pinecone
from langchain_openai import OpenAIEmbeddings
from utils.document_handler import DocumentHandler
from utils.pinecone_utils import init_pinecone, create_or_verify_index

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
                    
                # Create vector record
                vector_id = f"{metadata.get('file_name', 'doc')}_{chunk['metadata']['chunk_index']}"
                vector_metadata = {
                    **chunk['metadata'],
                    'content': chunk['content']  # Store text for retrieval
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
            # Delete vectors by matching metadata
            self.index.delete(
                filter={"file_name": file_name},
                namespace=self.namespace
            )
            logger.info(f"Deleted vectors for document: {file_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document vectors: {e}")
            return False