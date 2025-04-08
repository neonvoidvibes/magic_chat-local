import logging
import boto3
import os
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from .s3_utils import get_s3_client

logger = logging.getLogger(__name__)

TRANSCRIPT_MODE = 'regular' # Set default mode

def get_transcript_mode() -> str:
    """Get the current transcript mode."""
    return TRANSCRIPT_MODE

class TranscriptState:
    """Tracks position across multiple transcript files."""
    def __init__(self):
        self.file_positions: Dict[str, int] = {}
        self.last_modified: Dict[str, datetime] = {}
        self.current_latest_key: Optional[str] = None

def get_latest_transcript_file(agent_name: Optional[str] = None, event_id: Optional[str] = None, s3_client: Optional[boto3.client] = None) -> Optional[str]:
    """Get the latest transcript file key based on TRANSCRIPT_MODE setting and priority."""
    if s3_client is None: s3_client = get_s3_client()
    if not s3_client: logger.error("get_latest_transcript_file: S3 client unavailable."); return None
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not aws_s3_bucket: logger.error("get_latest_transcript_file: AWS_S3_BUCKET not set."); return None

    candidate_files = []
    prefixes_to_check = []
    if agent_name and event_id: prefixes_to_check.append(f'organizations/river/agents/{agent_name}/events/{event_id}/transcripts/')
    # prefixes_to_check.append('_files/transcripts/') # Optional fallback

    logger.debug(f"get_latest_transcript_file: Checking prefixes: {prefixes_to_check}")
    for prefix in prefixes_to_check:
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            logger.debug(f"Listing objects in s3://{aws_s3_bucket}/{prefix}")
            for page in paginator.paginate(Bucket=aws_s3_bucket, Prefix=prefix):
                 if 'Contents' in page:
                     for obj in page['Contents']:
                         key = obj['Key']
                         if key.startswith(prefix) and key != prefix and key.endswith('.txt'):
                              relative_path = key[len(prefix):]
                              if '/' not in relative_path:
                                   filename = os.path.basename(key); is_rolling = filename.startswith('rolling-')
                                   if (TRANSCRIPT_MODE == 'rolling' and is_rolling) or (TRANSCRIPT_MODE == 'regular' and not is_rolling):
                                        candidate_files.append(obj); logger.debug(f"Candidate file: {key}")
        except Exception as e: logger.error(f"Error listing S3 {prefix}: {e}", exc_info=True)

    if not candidate_files: logger.warning(f"No transcript files found (Mode: '{TRANSCRIPT_MODE}')"); return None
    candidate_files.sort(key=lambda x: x['LastModified'], reverse=True)
    latest_file = candidate_files[0]
    logger.info(f"Latest transcript file ({TRANSCRIPT_MODE}): {latest_file['Key']} (Mod: {latest_file['LastModified']})")
    return latest_file['Key']

def read_new_transcript_content(state: TranscriptState, agent_name: str, event_id: str, read_all: bool = False) -> Optional[str]:
    """Read new content from the latest transcript file."""
    s3_client = get_s3_client()
    if not s3_client: logger.error("read_new_transcript_content: S3 client unavailable."); return None
    # Correct variable name here:
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not aws_s3_bucket: logger.error("read_new_transcript_content: AWS_S3_BUCKET not set."); return None

    if read_all:
        logger.warning("read_new_transcript_content: read_all=True mode not fully implemented.")
        pass # Fall through

    try:
        latest_key = get_latest_transcript_file(agent_name, event_id, s3_client)
        if not latest_key: logger.debug("No latest transcript file found."); return None

        try:
            metadata = s3_client.head_object(Bucket=aws_s3_bucket, Key=latest_key) # Use correct var name
            current_size = metadata['ContentLength']; current_modified = metadata['LastModified']
        except Exception as head_e: logger.error(f"Error getting metadata for {latest_key}: {head_e}"); return None

        last_pos = state.file_positions.get(latest_key, 0)
        last_mod = state.last_modified.get(latest_key)
        is_new = (latest_key != state.current_latest_key)
        is_mod = (last_mod is None or current_modified > last_mod)
        has_new = (current_size > last_pos) # Use this variable

        if not is_new and not is_mod: logger.debug(f"Transcript {latest_key} unchanged."); return None

        start_read_pos = 0 # Default for new file or reset
        if not is_new:
            # Replace 'has_new_bytes' with 'has_new' here
            if not has_new and is_mod: logger.warning(f"Tx {latest_key} modified but no size increase. Reading from start."); start_read_pos = 0
            elif has_new: logger.debug(f"Tx {latest_key} updated. Reading from pos {last_pos}."); start_read_pos = last_pos
            else: logger.debug(f"No new bytes detected for {latest_key}."); return None
        else: logger.info(f"New transcript file: {latest_key}. Reading full content.")

        read_range = f"bytes={start_read_pos}-"
        new_content = ""; new_content_bytes = b""
        try:
            response = s3_client.get_object(Bucket=aws_s3_bucket, Key=latest_key, Range=read_range) # Use correct var name
            new_content_bytes = response['Body'].read(); new_content = new_content_bytes.decode('utf-8')
        except s3_client.exceptions.InvalidRange: logger.debug(f"InvalidRange (likely no new content) for {latest_key}.")
        except Exception as get_e: logger.error(f"Error reading {latest_key} (range {read_range}): {get_e}"); return None

        state.current_latest_key = latest_key
        state.last_modified[latest_key] = current_modified
        state.file_positions[latest_key] = start_read_pos + len(new_content_bytes) # Update based on bytes read

        if new_content:
            logger.debug(f"Read {len(new_content)} new chars from {latest_key}.")
            file_name = os.path.basename(latest_key); labeled_content = f"(Source File: {file_name})\n{new_content}"
            return labeled_content
        else: logger.debug(f"No new content read from {latest_key}."); return None

    except Exception as e: logger.error(f"Error reading transcript content: {e}", exc_info=True); return None

def read_all_transcripts_in_folder(agent_name: str, event_id: str) -> Optional[str]:
    """Read and combine content of all relevant transcripts in the folder."""
    s3_client = get_s3_client()
    if not s3_client: logger.error("read_all_transcripts: S3 unavailable."); return None
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not aws_s3_bucket: logger.error("read_all_transcripts: Bucket missing."); return None

    prefix = f'organizations/river/agents/{agent_name}/events/{event_id}/transcripts/'; logger.info(f"Reading all transcripts from: {prefix}")
    all_content = []
    try:
        paginator = s3_client.get_paginator('list_objects_v2'); transcript_files = []
        for page in paginator.paginate(Bucket=aws_s3_bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.startswith(prefix) and key != prefix and key.endswith('.txt'):
                         relative = key[len(prefix):]
                         if '/' not in relative: # Files directly in folder
                            fname = os.path.basename(key); is_rolling = fname.startswith('rolling-')
                            if (TRANSCRIPT_MODE=='rolling' and is_rolling) or (TRANSCRIPT_MODE=='regular' and not is_rolling):
                                 transcript_files.append(obj)
        if not transcript_files: logger.warning(f"No transcripts in {prefix}"); return None
        transcript_files.sort(key=lambda x: x['LastModified']); logger.info(f"Found {len(transcript_files)} tx files.")
        for t_obj in transcript_files:
            key = t_obj['Key']; fname = os.path.basename(key)
            try:
                response = s3_client.get_object(Bucket=aws_s3_bucket, Key=key)
                text = response['Body'].read().decode('utf-8'); all_content.append(f"--- Tx: {fname} ---\n{text}")
            except Exception as read_e: logger.error(f"Error reading {key}: {read_e}")
        if all_content: logger.info("Combined transcript content."); return "\n\n".join(all_content)
        else: logger.warning("No content read from tx files."); return None
    except Exception as e: logger.error(f"Error reading all tx: {e}", exc_info=True); return None