#!/usr/bin/env python
"""CLI tool for manually embedding text into Pinecone."""
import argparse
import logging
from pathlib import Path
from document_handler import DocumentHandler
from embedding_handler import EmbeddingHandler

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def embed_file(file_path: str, index_name: str, namespace: str = None):
    """Embed contents of a file into Pinecone."""
    try:
        # Initialize handlers
        doc_handler = DocumentHandler()
        embed_handler = EmbeddingHandler(index_name=index_name, namespace=namespace)
        
        # Read file
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Create metadata
        metadata = {
            'file_name': Path(file_path).name,
            'source': 'manual_upload'
        }
        
        # Embed and upsert
        success = embed_handler.embed_and_upsert(content, metadata)
        if success:
            logging.info(f"Successfully embedded {file_path}")
        else:
            logging.error(f"Failed to embed {file_path}")
            
    except Exception as e:
        logging.error(f"Error embedding file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Embed text files into Pinecone')
    parser.add_argument('file', help='Path to text file to embed')
    parser.add_argument('--index', default='chat-docs-index', help='Pinecone index name')
    parser.add_argument('--namespace', help='Optional Pinecone namespace')
    
    args = parser.parse_args()
    setup_logging()
    
    embed_file(args.file, args.index, args.namespace)

if __name__ == '__main__':
    main()