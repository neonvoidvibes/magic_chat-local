#!/usr/bin/env python
"""CLI tool for manually embedding text into Pinecone."""
from dotenv import load_dotenv
load_dotenv()
import argparse
import logging
from pathlib import Path
from .document_handler import DocumentHandler
from .embedding_handler import EmbeddingHandler

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
            namespace=agent_name  # Use agent name as namespace for upload target
        )

        # Read file
        with open(file_path, 'r') as f:
            content = f.read()

        # Prepare metadata - ensure agent_name is included here
        base_metadata = {
            'agent_path': f'organizations/river/agents/{agent_name}/docs/',
            'file_name': Path(file_path).name,
            'source': 'manual_upload',
            'agent_name': agent_name, # Explicitly add agent_name for filtering
            **(metadata or {}) # Include other metadata like event_id if provided
        }
        logging.info(f"Embedding with metadata: {base_metadata}")

        # Embed and upsert with metadata
        # EmbeddingHandler's embed_and_upsert will handle chunking and adding 'content' to metadata
        success = embed_handler.embed_and_upsert(content, base_metadata)
        if success:
            logging.info(f"Successfully embedded chunks from {file_path} in namespace {agent_name}")
        else:
            logging.error(f"Failed to embed {file_path}")

    except Exception as e:
        logging.error(f"Error embedding file {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Embed text files into Pinecone')
    parser.add_argument('file', help='Path to text file or directory of files to embed')
    parser.add_argument('--agent', required=True, help='Agent name (e.g., "yggdrasil") used for namespace and metadata')
    parser.add_argument('--index', default='magicchat', help='Pinecone index name')
    parser.add_argument('--event', help='Optional event ID for metadata filtering')

    args = parser.parse_args()
    setup_logging()

    # Prepare base metadata from args
    additional_metadata = {}
    if args.event:
        additional_metadata['event_id'] = args.event

    target_path = Path(args.file)

    if target_path.is_file():
        embed_file(str(target_path), args.agent, args.index, additional_metadata)
    elif target_path.is_dir():
        logging.info(f"Processing all files in directory: {target_path}")
        for item in target_path.iterdir():
            if item.is_file():
                logging.info(f"Processing file: {item.name}")
                embed_file(str(item), args.agent, args.index, additional_metadata)
            else:
                logging.warning(f"Skipping non-file item: {item.name}")
    else:
        logging.error(f"Path not found or is not a file/directory: {args.file}")


if __name__ == '__main__':
    main()