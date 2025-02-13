"""Utilities for managing Pinecone vector store operations."""
import os
import time
import logging
from typing import Optional
from dotenv import load_dotenv
import pinecone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_pinecone() -> bool:
    """
    Initialize Pinecone using classic approach (pinecone.init).
    Returns True if successful, otherwise False.
    """
    try:
        load_dotenv()  # Ensure environment variables are loaded
        api_key = os.getenv('PINECONE_API_KEY')
        environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east1-gcp')
        
        if not api_key:
            logger.error("PINECONE_API_KEY not found in environment variables")
            return False
        
        # Classic style init
        pinecone.init(api_key=api_key, environment=environment)
        logger.info(f"Pinecone initialized with environment: {environment}")
        return True

    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        return False

def create_or_verify_index(
    index_name: str = "chat-docs-index",
    dimension: int = 1536,  # OpenAI ada-002 dimension
    metric: str = "cosine"
) -> Optional[pinecone.Index]:
    """
    Create a new Pinecone index if it doesn't exist, or return the existing index.
    """
    try:
        success = init_pinecone()
        if not success:
            return None
        
        # Check if index exists
        if index_name in pinecone.list_indexes():
            logger.info(f"Index '{index_name}' already exists")
            return pinecone.Index(index_name)
        
        logger.info(f"Creating new index '{index_name}'...")
        pinecone.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric
        )
        # Wait for readiness
        while True:
            desc = pinecone.describe_index(index_name)
            if desc.status.get("ready", False):
                break
            logger.info("Waiting for index to be ready...")
            time.sleep(1)
        logger.info(f"Index '{index_name}' created successfully")

        return pinecone.Index(index_name)

    except Exception as e:
        logger.error(f"Error creating/verifying index: {e}")
        return None

def delete_index(index_name: str) -> bool:
    """Delete a Pinecone index if it exists."""
    try:
        success = init_pinecone()
        if not success:
            return False
        
        existing = pinecone.list_indexes()
        if index_name in existing:
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
    """
    Get statistics for a Pinecone index.
    """
    try:
        success = init_pinecone()
        if not success:
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