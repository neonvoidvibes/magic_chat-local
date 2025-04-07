"""Utilities for managing Pinecone vector store operations."""
import os
import time
import logging
from typing import Optional
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
# Import specific exception for finer control
from pinecone.exceptions import NotFoundException

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
        environment = os.getenv('PINECONE_ENVIRONMENT') # Let Pinecone client handle default if None

        if not api_key:
            logger.error("PINECONE_API_KEY not found in environment variables")
            return None

        pc = Pinecone(api_key=api_key, environment=environment)
        active_environment = environment or "default (client library)" # For logging
        logger.info(f"Pinecone initialized with environment: {active_environment}")
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
    pc = init_pinecone()
    if not pc:
        logger.error("Pinecone client failed to initialize.")
        return None

    try:
        # Try to describe the index first to see if it exists and check dimension
        logger.info(f"Checking for existing index '{index_name}'...")
        index_description = pc.describe_index(index_name)
        logger.info(f"Index '{index_name}' found.")

        # Check dimension
        if index_description.dimension != dimension:
            logger.error(f"Existing index '{index_name}' has dimension {index_description.dimension}, but expected {dimension}. Please delete the index manually via the Pinecone console and retry.")
            return None
        else:
            logger.info(f"Existing index '{index_name}' has the correct dimension ({dimension}).")
            return pc.Index(index_name)

    except NotFoundException:
        # Index does not exist, proceed to create it
        logger.info(f"Index '{index_name}' not found. Proceeding with creation...")
        try:
            region = os.getenv('PINECONE_REGION', 'us-east-1') # Default AWS region
            cloud_provider = 'aws'
            logger.info(f"Creating new index '{index_name}' on cloud: {cloud_provider}, region: {region} with dimension {dimension}")

            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud_provider, region=region)
            )

            # Wait for the index to be ready
            logger.info(f"Waiting for index '{index_name}' to be ready...")
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(5)
            logger.info(f"Index '{index_name}' created successfully and is ready.")
            return pc.Index(index_name)

        except Exception as create_e:
            # Catch errors during creation specifically
            logger.error(f"Error creating index '{index_name}': {create_e}")
            if hasattr(create_e, 'body'):
                 logger.error(f"Creation error response body: {create_e.body}")
            return None

    except Exception as e:
        # Catch any other errors during the describe_index call
        logger.error(f"Error checking or creating index '{index_name}': {e}")
        if hasattr(e, 'body'):
             logger.error(f"Describe index error response body: {e.body}")
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
        logger.info(f"Attempting to delete index '{index_name}'...")
        pc.delete_index(index_name)
        logger.info(f"Index '{index_name}' deleted successfully")
        return True
    except NotFoundException:
        logger.warning(f"Index '{index_name}' does not exist, cannot delete.")
        return False # Or True, depending on desired idempotency behaviour. False seems safer.
    except Exception as e:
        logger.error(f"Error deleting index '{index_name}': {e}")
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
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        stats_dict = stats.to_dict() if hasattr(stats, 'to_dict') else stats
        logger.info(f"Retrieved stats for index '{index_name}': {stats_dict}")
        return stats_dict
    except NotFoundException:
         logger.warning(f"Index '{index_name}' does not exist, cannot get stats.")
         return None
    except Exception as e:
        logger.error(f"Error getting index stats for '{index_name}': {e}")
        return None