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

def embed_file(file_path: str, index_name: str, namespace: str = None, metadata: dict = None):
    """Embed contents of a file into Pinecone with agent-specific metadata."""
    try:
        # Initialize handlers
        doc_handler = DocumentHandler()
        embed_handler = EmbeddingHandler(index_name=index_name, namespace=namespace)
        
        # Read file
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Embed and upsert with metadata
        success = embed_handler.embed_and_upsert(content, metadata or {})
        if success:
            logging.info(f"Successfully embedded {file_path} for agent {metadata.get('agent_name')}")
        else:
            logging.error(f"Failed to embed {file_path}")
            
    except Exception as e:
        logging.error(f"Error embedding file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Embed text files into Pinecone')
    parser.add_argument('file', help='Path to text file to embed')
    parser.add_argument('--agent', required=True, help='Agent name (e.g., "river")')
    parser.add_argument('--index', default='chat-docs-index', help='Pinecone index name')
    parser.add_argument('--namespace', help='Optional Pinecone namespace')

    args = parser.parse_args()
    setup_logging()

    # Create S3-like path structure in metadata
    metadata = {
        'agent_path': f'organizations/river/agents/{args.agent}/docs/',
        'agent_name': args.agent,
        'file_name': Path(args.file).name,
        'source': 'manual_upload'
    }

    embed_file(args.file, args.index, args.namespace, metadata)

if __name__ == '__main__':
    main()