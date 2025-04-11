#!/usr/bin/env python
"""CLI tool for manually embedding text into Pinecone."""
from dotenv import load_dotenv
load_dotenv()
import argparse
import logging
from pathlib import Path
import os # Import os for path operations

# Import Langchain PDF Loader
from langchain_community.document_loaders import PyPDFLoader

from .document_handler import DocumentHandler # Still needed for non-PDFs? Embedding handler uses its own splitter now.
from .embedding_handler import EmbeddingHandler

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def embed_file(file_path: str, agent_name: str, index_name: str, metadata: dict = None):
    """Embed contents of a file into Pinecone using agent namespace."""
    try:
        # Initialize embedding handler (handles Pinecone connection and embedding logic)
        embed_handler = EmbeddingHandler(
            index_name=index_name,
            namespace=agent_name  # Use agent name as namespace for upload target
        )

        file_extension = os.path.splitext(file_path)[1].lower()
        base_file_name = Path(file_path).name

        # --- Base metadata common to all chunks/pages ---
        base_metadata = {
            'agent_path': f'organizations/river/agents/{agent_name}/docs/', # Example path structure
            'file_name': base_file_name,
            'source': 'manual_upload',
            'agent_name': agent_name, # Explicitly add agent_name for filtering
            **(metadata or {}) # Include other CLI metadata like event_id if provided
        }
        logging.info(f"Base metadata for file '{base_file_name}': {base_metadata}")


        if file_extension == ".pdf":
            logging.info(f"Detected PDF file: {file_path}. Using PyPDFLoader.")
            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load() # Returns list of Document objects (one per page)
                logging.info(f"Loaded {len(pages)} pages from PDF.")

                total_success = True
                for i, page in enumerate(pages):
                    page_content = page.page_content
                    # Get metadata from the loader (like page number) and merge with base
                    page_specific_metadata = page.metadata # Contains {'source': file_path, 'page': page_num}
                    combined_metadata = {
                        **base_metadata,
                        'page_number': page_specific_metadata.get('page', i) + 1 # Use loader's page or index+1
                    }
                    logging.info(f"Processing Page {combined_metadata['page_number']}...")

                    if not page_content.strip():
                        logging.warning(f"Page {combined_metadata['page_number']} has no text content. Skipping.")
                        continue

                    # Call embed_and_upsert for this page's content
                    success = embed_handler.embed_and_upsert(page_content, combined_metadata)
                    if success:
                        logging.info(f"Successfully embedded chunks from Page {combined_metadata['page_number']} of {base_file_name}")
                    else:
                        logging.error(f"Failed to embed Page {combined_metadata['page_number']} from {base_file_name}")
                        total_success = False # Mark overall failure if any page fails

                if not total_success:
                     logging.error(f"Embedding failed for one or more pages in PDF: {base_file_name}")

            except Exception as pdf_e:
                logging.error(f"Error processing PDF file {file_path}: {pdf_e}", exc_info=True)

        else: # Handle as plain text
            logging.info(f"Detected non-PDF file: {file_path}. Reading as text.")
            try:
                # Read file as text (assuming UTF-8)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Embed and upsert the entire text content
                success = embed_handler.embed_and_upsert(content, base_metadata)
                if success:
                    logging.info(f"Successfully embedded chunks from text file {base_file_name} in namespace {agent_name}")
                else:
                    logging.error(f"Failed to embed text file {base_file_name}")

            except UnicodeDecodeError as utf_e:
                 logging.error(f"UTF-8 decoding error for non-PDF file {file_path}: {utf_e}. Try ensuring the file is UTF-8 encoded or handle other encodings.", exc_info=True)
            except Exception as text_e:
                 logging.error(f"Error reading/embedding text file {file_path}: {text_e}", exc_info=True)

    except Exception as e:
        logging.error(f"General error embedding file {file_path}: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description='Embed text or PDF files into Pinecone')
    parser.add_argument('file', help='Path to text/PDF file or directory of files to embed')
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