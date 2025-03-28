#!/usr/bin/env python3
"""
A CLI script to upload a file or all files in a folder to Pinecone using the EmbeddingHandler from utils.embedding_handler.
Usage:
    To upload a single file:
      python pinecone_upload_cli.py [--index INDEX_NAME] /path/to/your/file.txt

    To batch upload all files in a folder:
      python pinecone_upload_cli.py [--index INDEX_NAME] /path/to/your/folder

This script reads the file(s), computes their real embeddings using the EmbeddingHandler,
adjusts vector dimensions to 1024, and upserts them to the specified Pinecone index.
Make sure your environment variables (OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, etc.) are set in a .env file.
"""

import sys
import os
import argparse
import pinecone
from dotenv import load_dotenv
from utils.embedding_handler import EmbeddingHandler

VECTOR_DIMENSION = 1024

def adjust_vector(vector):
    """Adjust the vector to the required dimension (1024).
       If the vector is longer than 1024, truncate it.
       If it is shorter, pad with zeros.
    """
    current_dim = len(vector)
    if current_dim > VECTOR_DIMENSION:
        return vector[:VECTOR_DIMENSION]
    elif current_dim < VECTOR_DIMENSION:
        return vector + [0.0] * (VECTOR_DIMENSION - current_dim)
    return vector

def process_file(file_path: str, embedding_handler, results: dict):
    """Read a file, compute its embedding, adjust dimensions, and upsert to Pinecone using EmbeddingHandler."""
    try:
        with open(file_path, "r") as file:
            content = file.read().strip()
        # Compute the real embedding from the file content using EmbeddingHandler
        vector = embedding_handler.generate_embedding(content)
        vector = adjust_vector(vector)
    except Exception as e:
        results[file_path] = f"Error processing file: {e}"
        return
    
    # Use the base file name as the document ID
    document_id = os.path.basename(file_path)
    metadata = {"filename": document_id}
    
    try:
        response = embedding_handler.index.upsert(vectors=[{"id": document_id, "values": vector, "metadata": metadata}])
        results[file_path] = f"Success: {response}"
    except Exception as e:
        results[file_path] = f"Error uploading: {e}"

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Upload a file or all files in a folder to Pinecone with real embeddings.")
    parser.add_argument("path", help="Path to the file or folder to upload")
    parser.add_argument("--index", default="chat-docs-index", help="Pinecone index name (default: chat-docs-index)")
    args = parser.parse_args()

    target_path = args.path
    index_name = args.index
    
    if not os.path.exists(target_path):
        print(f"Path not found: {target_path}")
        sys.exit(1)
    
    # Initialize EmbeddingHandler with the specified Pinecone index
    try:
        embedding_handler = EmbeddingHandler(index_name=index_name)
    except Exception as e:
        print(f"Failed to initialize EmbeddingHandler for index '{index_name}': {e}")
        sys.exit(1)
    
    # Perform a dimension check using a Pinecone client instance from the pinecone package.
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        environment_var = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
        # Create a Pinecone client instance.
        from pinecone import Pinecone
        pc = Pinecone(api_key=api_key, environment=environment_var)
        index_desc = pc.describe_index(index_name)
        index_dim = index_desc.get("dimension")
        if index_dim is not None and index_dim != VECTOR_DIMENSION:
            print(f"Error: Required vector dimension is {VECTOR_DIMENSION} but the index dimension is {index_dim}.")
            print("Please delete or recreate the index with the correct dimension.")
            sys.exit(1)
    except Exception as e:
        print(f"Failed to check index dimension: {e}")
        sys.exit(1)
    
    results = {}
    # Check if target_path is a directory or a single file
    if os.path.isdir(target_path):
        # Process each file in the directory (non-recursive)
        files = [os.path.join(target_path, f) for f in os.listdir(target_path) if os.path.isfile(os.path.join(target_path, f))]
        if not files:
            print("No files found in the specified folder.")
            sys.exit(0)
        for file_path in files:
            print(f"Processing {file_path}...")
            process_file(file_path, embedding_handler, results)
    else:
        # Single file upload
        process_file(target_path, embedding_handler, results)
        
    # Output results
    print("\nBatch upload summary:")
    for path, status in results.items():
        print(f"{path}: {status}")

if __name__ == "__main__":
    main()
