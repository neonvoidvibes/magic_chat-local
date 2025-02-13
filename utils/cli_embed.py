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

def embed_file(file_path: str, agent_name: str, index_name: str, metadata: dict = None):
    """Embed contents of a file into Pinecone using agent namespace."""
    try:
        # Initialize handlers
        doc_handler = DocumentHandler()
        embed_handler = EmbeddingHandler(
            index_name=index_name,
            namespace=agent_name  # Use agent name as namespace
        )
        
        # Read file
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Prepare metadata
        file_metadata = {
            'agent_path': f'organizations/river/agents/{agent_name}/docs/',
            'file_name': Path(file_path).name,
            'source': 'manual_upload',
            **(metadata or {})
        }
            
        # Embed and upsert with metadata
        success = embed_handler.embed_and_upsert(content, file_metadata)
        if success:
            logging.info(f"Successfully embedded {file_path} in namespace {agent_name}")
        else:
            logging.error(f"Failed to embed {file_path}")
            
    except Exception as e:
        logging.error(f"Error embedding file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Embed text files into Pinecone')
    parser.add_argument('file', help='Path to text file to embed')
    parser.add_argument('--agent', required=True, help='Agent name (e.g., "river")')
    parser.add_argument('--index', default='chat-docs-index', help='Pinecone index name')
    parser.add_argument('--event', help='Optional event ID for metadata filtering')

    args = parser.parse_args()
    setup_logging()

    # Prepare metadata
    metadata = {}
    if args.event:
        metadata['event_id'] = args.event

    embed_file(args.file, args.agent, args.index, metadata)

if __name__ == '__main__':
    main()