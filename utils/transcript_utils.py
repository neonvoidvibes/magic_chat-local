import logging
import boto3
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from .retrieval_handler import RetrievalHandler

# Toggle between 'rolling' and 'regular' transcript modes
# TRANSCRIPT_MODE = 'rolling'  # Uses rolling-transcript_ files
TRANSCRIPT_MODE = 'regular'  # Uses transcript_ files (non-rolling)

def get_transcript_mode() -> str:
    """Get the current transcript mode.
    Returns 'rolling' or 'regular' based on the TRANSCRIPT_MODE setting.
    """
    return TRANSCRIPT_MODE

class TranscriptState:
    def __init__(self):
        self.current_key = None
        self.last_position = 0
        self.last_modified = None

def get_latest_transcript_file(agent_name=None, event_id=None, s3_client=None, bucket_name=None):
    """Get the latest transcript file following priority order:
    1. rolling-transcript_ (agent event folder)
    2. transcript_ (agent event folder)
    3. transcript_ (general folder)
    """
    if s3_client is None:
        s3_client = boto3.client(
            's3',
            region_name=os.getenv('AWS_REGION'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
    
    if bucket_name is None:
        bucket_name = os.getenv('AWS_S3_BUCKET')
    
    try:
        # First try agent's event folder
        if agent_name and event_id:
            prefix = f'organizations/river/agents/{agent_name}/events/{event_id}/transcripts/'
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
            
            if 'Contents' in response:
                # Get all transcript files
                all_files = [
                    obj['Key'] for obj in response['Contents']
                    if obj['Key'].startswith(prefix) and obj['Key'] != prefix
                    and not obj['Key'].replace(prefix, '').strip('/').count('/')
                    and obj['Key'].endswith('.txt')
                ]
                
                # Filter based on TRANSCRIPT_MODE
                if TRANSCRIPT_MODE == 'rolling':
                    # Only use rolling transcripts
                    filtered_files = [
                        f for f in all_files
                        if f.replace(prefix, '').startswith('rolling-')
                    ]
                else:  # regular mode
                    # Only use regular transcripts
                    filtered_files = [
                        f for f in all_files
                        if not f.replace(prefix, '').startswith('rolling-')
                    ]
                
                if filtered_files:
                    latest_file = max(filtered_files, key=lambda x: s3_client.head_object(Bucket=bucket_name, Key=x)['LastModified'])
                    logging.debug(f"Selected latest {'rolling' if TRANSCRIPT_MODE == 'rolling' else 'regular'} transcript in agent folder: {latest_file}")
                    return latest_file
        
        # Third priority: transcripts in general folder
        prefix = '_files/transcripts/'
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
        
        if 'Contents' in response:
            transcript_files = [
                obj['Key'] for obj in response['Contents']
                if obj['Key'].startswith(prefix) and obj['Key'] != prefix
                and not obj['Key'].replace(prefix, '').strip('/').count('/')
                and obj['Key'].endswith('.txt')
            ]
            if transcript_files:
                latest_file = max(transcript_files, key=lambda x: s3_client.head_object(Bucket=bucket_name, Key=x)['LastModified'])
                logging.debug(f"Selected latest transcript in general folder: {latest_file}")
                return latest_file
                
        return None
        
    except Exception as e:
        logging.error(f"Error finding transcript files in S3: {e}")
        return None

def read_new_transcript_content(state, agent_name, event_id, s3_client=None, bucket_name=None, read_all=False):
    """
    Read new content from both file-based transcripts and Pinecone vector DB in parallel.
    When read_all=True, combines partial updates from all transcripts.
    Otherwise, uses single 'latest' transcript approach.
    
    File-based transcripts are read in this order:
    1. rolling-transcript_ (agent event folder)
    2. transcript_ (agent event folder)
    3. transcript_ (general folder)
    
    Pinecone vector DB is always queried in parallel for redundancy.
    
    Args:
        state: TranscriptState object tracking file positions
        agent_name: Name of the agent
        event_id: Current event ID
        s3_client: Optional pre-configured S3 client
        bucket_name: Optional S3 bucket name
        read_all: Whether to read from all available transcripts
    """
    try:
        if s3_client is None:
            s3_client = boto3.client(
                's3',
                region_name=os.getenv('AWS_REGION'),
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
        if bucket_name is None:
            bucket_name = os.getenv('AWS_S3_BUCKET')

        # Initialize multi-file tracking if not present
        if not hasattr(state, 'file_positions'):
            state.file_positions = {}  # dict: { key: last_position_int }
        
        if not read_all:
            # Single-latest-file approach (original logic)
            latest_key = get_latest_transcript_file(agent_name, event_id, s3_client, bucket_name)
            if not latest_key:
                logging.debug("No transcript file found")
                return None
            
            response = s3_client.get_object(Bucket=bucket_name, Key=latest_key)
            content = response['Body'].read().decode('utf-8')
            
            if latest_key != state.current_key:
                # New file
                new_content = content
                state.last_position = len(content)
                logging.debug(f"New transcript file detected, read {len(new_content)} bytes")
            else:
                # Existing file updated - read only appended
                new_content = content[state.last_position:]
                state.last_position = len(content)
                logging.debug(f"Existing file updated, read {len(new_content)} new bytes")

            state.current_key = latest_key
            # We won't store multiple positions for single-file mode
            state.file_positions[latest_key] = state.last_position
            
            if new_content:
                # Add timestamp and source labeling with filename
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                file_name = os.path.basename(latest_key)
                labeled_content = f"[LIVE TRANSCRIPT {timestamp}] Source file: {file_name}\n{new_content}"
                return labeled_content
            return None
        
        else:
            # read_all == True => partial updates from all transcripts following priority order
            combined_new = []
            
            if agent_name and event_id:
                prefix = f'organizations/river/agents/{agent_name}/events/{event_id}/transcripts/'
                response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
                
                if 'Contents' in response:
                    # First priority: rolling transcripts
                    rolling_transcripts = [
                        {
                            'Key': obj['Key'],
                            'LastModified': obj['LastModified']
                        }
                        for obj in response['Contents']
                        if obj['Key'].endswith('.txt') and 'rolling-' in obj['Key']
                    ]
                    
                    # Second priority: regular transcripts in agent folder
                    regular_transcripts = [
                        {
                            'Key': obj['Key'],
                            'LastModified': obj['LastModified']
                        }
                        for obj in response['Contents']
                        if obj['Key'].endswith('.txt') and 'rolling-' not in obj['Key']
                    ]
                    
                    # Process rolling transcripts first
                    for t in sorted(rolling_transcripts, key=lambda x: x['LastModified']):
                        k = t['Key']
                        if k not in state.file_positions:
                            state.file_positions[k] = 0
                        
                        obj = s3_client.get_object(Bucket=bucket_name, Key=k)
                        text = obj['Body'].read().decode('utf-8')
                        prev_pos = state.file_positions[k]
                        if prev_pos > len(text):
                            prev_pos = 0
                        
                        new_text = text[prev_pos:]
                        if new_text:
                            combined_new.append(f"[Rolling File: {os.path.basename(k)}]\n{new_text}")
                            logging.debug(f"Read {len(new_text)} new bytes from rolling transcript {k}")
                        
                        state.file_positions[k] = len(text)
                    
                    # Then process regular transcripts
                    for t in sorted(regular_transcripts, key=lambda x: x['LastModified']):
                        k = t['Key']
                        if k not in state.file_positions:
                            state.file_positions[k] = 0
                        
                        obj = s3_client.get_object(Bucket=bucket_name, Key=k)
                        text = obj['Body'].read().decode('utf-8')
                        prev_pos = state.file_positions[k]
                        if prev_pos > len(text):
                            prev_pos = 0
                        
                        new_text = text[prev_pos:]
                        if new_text:
                            combined_new.append(f"[File: {os.path.basename(k)}]\n{new_text}")
                            logging.debug(f"Read {len(new_text)} new bytes from transcript {k}")
                        
                        state.file_positions[k] = len(text)
            
            # Third priority: general folder transcripts
            prefix = '_files/transcripts/'
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            
            if 'Contents' in response:
                general_transcripts = [
                    {
                        'Key': obj['Key'],
                        'LastModified': obj['LastModified']
                    }
                    for obj in response['Contents']
                    if obj['Key'].endswith('.txt')
                ]
                
                for t in sorted(general_transcripts, key=lambda x: x['LastModified']):
                    k = t['Key']
                    if k not in state.file_positions:
                        state.file_positions[k] = 0
                    
                    obj = s3_client.get_object(Bucket=bucket_name, Key=k)
                    text = obj['Body'].read().decode('utf-8')
                    prev_pos = state.file_positions[k]
                    if prev_pos > len(text):
                        prev_pos = 0
                    
                    new_text = text[prev_pos:]
                    if new_text:
                        combined_new.append(f"[General File: {os.path.basename(k)}]\n{new_text}")
                        logging.debug(f"Read {len(new_text)} new bytes from general transcript {k}")
                    
                    state.file_positions[k] = len(text)
            
            # Always try to get content from Pinecone DB in parallel
            try:
                # Initialize retrieval handler
                retriever = RetrievalHandler(
                    agent_name=agent_name,
                    event_id=event_id
                )
                
                # Get recent content from vector DB
                results = retriever.get_relevant_context(
                    query="",  # Empty query to get recent content
                    is_transcript=True,
                    top_k=5  # Adjust this number as needed
                )
                
                if results:
                    # Format vector DB results
                    for result in results:
                        content = result.get('content', '')
                        source = result.get('source', 'Unknown Source')
                        if content:
                            combined_new.append(f"[Vector DB: {source}]\n{content}")
                            logging.debug(f"Retrieved content from vector DB source: {source}")
                            
            except Exception as e:
                logging.error(f"Error retrieving from vector DB: {e}")
            
            if combined_new:
                # combine with extra spacing
                combined_content = "\n\n".join(combined_new)
                # Add timestamp and source labeling
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                labeled_content = f"[LIVE TRANSCRIPT {timestamp}] Multiple source files\n{combined_content}"
                return labeled_content
            else:
                return None

    except Exception as e:
        logging.error(f"Error reading transcripts: {e}")
        return None

def read_all_transcripts_in_folder(agent_name, event_id, s3_client=None, bucket_name=None):
    """
    Read the entire content of all .txt transcripts in 
    organizations/river/agents/{agent_name}/events/{event_id}/transcripts/
    Combine them in chronological order, return as a single string.
    """
    if s3_client is None:
        s3_client = boto3.client(
            's3',
            region_name=os.getenv('AWS_REGION'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
    if bucket_name is None:
        bucket_name = os.getenv('AWS_S3_BUCKET')

    prefix = f'organizations/river/agents/{agent_name}/events/{event_id}/transcripts/'
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' not in response:
            logging.debug(f"No transcripts found in {prefix}")
            return None

        # Collect all .txt files
        transcripts = []
        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('.txt'):
                transcripts.append({
                    'Key': key,
                    'LastModified': obj['LastModified']
                })
        if not transcripts:
            logging.debug(f"No .txt transcripts found in {prefix}")
            return None

        # Sort by LastModified ascending
        transcripts.sort(key=lambda x: x['LastModified'])
        combined_content = []
        for t in transcripts:
            obj = s3_client.get_object(Bucket=bucket_name, Key=t['Key'])
            text = obj['Body'].read().decode('utf-8')
            combined_content.append(text)

        if combined_content:
            return "\n\n".join(combined_content)
        else:
            return None
    except Exception as e:
        logging.error(f"Error reading all transcripts from S3: {e}")
        return None