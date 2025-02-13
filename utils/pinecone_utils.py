"""Utilities for managing Pinecone vector store operations."""
import os
import time
import logging
from typing import Optional
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_pinecone() -> Optional[Pinecone]:
    """Initialize Pinecone with environment variables and return a Pinecone instance.
    
    Returns:
        Pinecone instance if initialization is successful, None otherwise.
    """
    try:
        load_dotenv()  # Ensure environment variables are loaded
        api_key = os.getenv('PINECONE_API_KEY')
        # Use PINECONE_ENVIRONMENT for compatibility, but for region use a separate variable if needed.
        environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east1-gcp')
        
        if not api_key:
            logger.error("PINECONE_API_KEY not found in environment variables")
            return None
        
        # Create a Pinecone instance using the new API.
        pc = Pinecone(api_key=api_key, environment=environment)
        logger.info(f"Pinecone initialized with environment: {environment}")
        return pc
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        return None

def create_or_verify_index(
    index_name: str = "chat-docs-index",
    dimension: int = 1536,  # OpenAI ada-002 dimension
    metric: str = "cosine"
) -> Optional[object]:
    """Create a new Pinecone index if it doesn't exist, or verify and return the existing index.
    
    Returns:
        The index object if successful, None otherwise.
    """
    try:
        pc = init_pinecone()
        if not pc:
            return None
        indexes = pc.list_indexes().names()
        if index_name in indexes:
            logger.info(f"Index '{index_name}' already exists")
            return pc.Index(index_name)
        logger.info(f"Creating new index '{index_name}'...")
        # Use a region value; if not set, default to environment value or 'us-east1-gcp'
        region = os.getenv('PINECONE_REGION', 'us-east1-gcp')
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud='aws', region=region)
        )
        # Wait for the index to be ready
        while not pc.describe_index(index_name).status.get("ready", False):
            logger.info("Waiting for index to be ready...")
            time.sleep(1)
        logger.info(f"Index '{index_name}' created successfully")
        return pc.Index(index_name)
    except Exception as e:
        logger.error(f"Error creating/verifying index: {e}")
        return None

def delete_index(index_name: str) -> bool:
    """Delete a Pinecone index.
    
    Returns:
        True if deletion is successful, False otherwise.
    """
    try:
        pc = init_pinecone()
        if not pc:
            return False
        if index_name in pc.list_indexes().names():
            pc.delete_index(index_name)
            logger.info(f"Index '{index_name}' deleted successfully")
            return True
        else:
            logger.warning(f"Index '{index_name}' does not exist")
            return False
    except Exception as e:
        logger.error(f"Error deleting index: {e}")
        return False

def get_index_stats(index_name: str) -> Optional[dict]:
    """Get statistics for a Pinecone index.
    
    Returns:
        A dictionary of index statistics if successful, None otherwise.
    """
    try:
        pc = init_pinecone()
        if not pc:
            return None
        if index_name not in pc.list_indexes().names():
            logger.warning(f"Index '{index_name}' does not exist")
            return None
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        logger.info(f"Retrieved stats for index '{index_name}'")
        return stats
    except Exception as e:
        logger.error(f"Error getting index stats: {e}")
        return None