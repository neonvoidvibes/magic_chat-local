"""Utilities for managing Pinecone vector store operations."""
import os
import time
import logging
from typing import Optional
import pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_pinecone() -> bool:
    """Initialize Pinecone with environment variables.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        api_key = os.getenv('PINECONE_API_KEY')
        environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east1-gcp')
        
        if not api_key:
            logger.error("PINECONE_API_KEY not found in environment variables")
            return False
            
        pinecone.init(api_key=api_key, environment=environment)
        logger.info(f"Pinecone initialized with environment: {environment}")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        return False

def create_or_verify_index(
    index_name: str = "chat-docs-index",
    dimension: int = 1536,  # OpenAI ada-002 dimension
    metric: str = "cosine",
) -> Optional[pinecone.Index]:
    """Create a new Pinecone index if it doesn't exist, or verify and return existing one.
    
    Args:
        index_name: Name for the index
        dimension: Vector dimension (1536 for OpenAI ada-002)
        metric: Distance metric for similarity search
        
    Returns:
        pinecone.Index if successful, None if failed
    """
    try:
        # Initialize Pinecone if not already done
        if not init_pinecone():
            return None
            
        # Check if index already exists
        if index_name in pinecone.list_indexes():
            logger.info(f"Index '{index_name}' already exists")
            return pinecone.Index(index_name)
            
        # Create new index (serverless mode)
        logger.info(f"Creating new index '{index_name}'...")
        pinecone.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric
        )
        
        # Wait for index to be ready
        while not pinecone.describe_index(index_name).status.get("ready", False):
            logger.info("Waiting for index to be ready...")
            time.sleep(1)
            
        logger.info(f"Index '{index_name}' created successfully")
        return pinecone.Index(index_name)
        
    except Exception as e:
        logger.error(f"Error creating/verifying index: {e}")
        return None

def delete_index(index_name: str) -> bool:
    """Delete a Pinecone index.
    
    Args:
        index_name: Name of the index to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not init_pinecone():
            return False
            
        if index_name in pinecone.list_indexes():
            pinecone.delete_index(index_name)
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
    
    Args:
        index_name: Name of the index
        
    Returns:
        dict: Index statistics if successful, None if failed
    """
    try:
        if not init_pinecone():
            return None
            
        if index_name not in pinecone.list_indexes():
            logger.warning(f"Index '{index_name}' does not exist")
            return None
            
        index = pinecone.Index(index_name)
        stats = index.describe_index_stats()
        logger.info(f"Retrieved stats for index '{index_name}'")
        return stats
        
    except Exception as e:
        logger.error(f"Error getting index stats: {e}")
        return None