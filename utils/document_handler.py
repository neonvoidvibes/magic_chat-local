"""Document handling utilities for splitting and processing text documents."""
import logging
from typing import List, Optional
from langchain.text_splitter import CharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentHandler:
    """Handles document processing and text splitting operations."""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separator: str = "\n\n"
    ):
        """Initialize document handler with configurable chunking parameters.
        
        Args:
            chunk_size: Maximum size of text chunks
            chunk_overlap: Number of characters to overlap between chunks
            separator: String separator to use for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks using the configured splitter.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        try:
            if not text:
                logger.warning("Empty text provided for splitting")
                return []
                
            chunks = self.splitter.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting text: {e}")
            return []
            
    def process_document(
        self,
        content: str,
        metadata: Optional[dict] = None
    ) -> List[dict]:
        """Process a document and prepare chunks with metadata.
        
        Args:
            content: Document content to process
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of dicts containing chunks and metadata
        """
        try:
            chunks = self.split_text(content)
            if not chunks:
                return []
                
            # Initialize metadata dict if none provided
            doc_metadata = metadata or {}
            
            # Create list of chunk objects with metadata
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_obj = {
                    'content': chunk,
                    'metadata': {
                        **doc_metadata,
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }
                }
                processed_chunks.append(chunk_obj)
                
            logger.info(f"Processed document into {len(processed_chunks)} chunks with metadata")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return []

    def validate_chunk_size(self, model_max_tokens: int) -> bool:
        """Validate that chunk size is appropriate for model context window.
        
        Args:
            model_max_tokens: Maximum tokens supported by target model
            
        Returns:
            bool indicating if chunk size is valid
        """
        # Estimate tokens (rough approximation of 4 chars per token)
        estimated_tokens = self.chunk_size / 4
        
        if estimated_tokens > model_max_tokens:
            logger.warning(
                f"Chunk size may be too large for model context window. "
                f"Estimated tokens per chunk: {estimated_tokens}, "
                f"Model max tokens: {model_max_tokens}"
            )
            return False
        return True