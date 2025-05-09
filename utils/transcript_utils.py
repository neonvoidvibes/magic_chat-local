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

    for prefix in prefixes_to_check:
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=aws_s3_bucket, Prefix=prefix):
                 if 'Contents' in page:
                     for obj in page['Contents']:
                         key = obj['Key']
                         if key.startswith(prefix) and key != prefix and key.endswith('.txt'):
                              relative_path = key[len(prefix):]
                              if '/' not in relative_path:
                                   filename = os.path.basename(key); is_rolling = filename.startswith('rolling-')
                                   if (TRANSCRIPT_MODE == 'rolling' and is_rolling) or (TRANSCRIPT_MODE == 'regular' and not is_rolling):
                                        candidate_files.append(obj)
        except Exception as e: logger.error(f"Error listing S3 {prefix}: {e}", exc_info=True)

    if not candidate_files: logger.warning(f"No transcript files found (Mode: '{TRANSCRIPT_MODE}') in {prefixes_to_check}"); return None
    candidate_files.sort(key=lambda x: x['LastModified'], reverse=True)
    latest_file = candidate_files[0]
    # Reduce logging verbosity
    # logger.info(f"Latest transcript file ({TRANSCRIPT_MODE}): {latest_file['Key']} (Mod: {latest_file['LastModified']})")
    return latest_file['Key']

def read_new_transcript_content(state: TranscriptState, agent_name: str, event_id: str, read_all: bool = False) -> Optional[str]:
    """Read new content from the latest transcript file, robustly updating position."""
    s3_client = get_s3_client()
    if not s3_client: logger.error("read_new_transcript_content: S3 client unavailable."); return None
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not aws_s3_bucket: logger.error("read_new_transcript_content: AWS_S3_BUCKET not set."); return None

    if read_all:
        logger.warning("read_new_transcript_content: read_all=True mode not implemented.")
        pass

    latest_key = get_latest_transcript_file(agent_name, event_id, s3_client)
    if not latest_key:
        return None

    try:
        # --- Get current S3 state ---
        metadata = s3_client.head_object(Bucket=aws_s3_bucket, Key=latest_key)
        current_size = metadata['ContentLength']
        current_modified = metadata['LastModified']

        # --- Get last known state from our state tracker ---
        last_pos = state.file_positions.get(latest_key, 0)
        last_mod = state.last_modified.get(latest_key)

        # --- Determine if changes occurred ---
        is_new_file = (latest_key != state.current_latest_key)
        is_modified = (last_mod is None or current_modified > last_mod) or (current_size > last_pos)
        has_new_bytes_on_s3 = (current_size > last_pos)

        logger.debug(f"Checking Tx: Key='{os.path.basename(latest_key)}', "
                     f"S3_Size={current_size}, S3_Mod={current_modified}, "
                     f"State_Pos={last_pos}, State_Mod={last_mod}, "
                     f"IsNewFile={is_new_file}, IsMod={is_modified}, HasNewBytes={has_new_bytes_on_s3}")

        # --- Decide whether to read ---
        if not is_new_file and not is_modified:
            return None # No change detected

        start_read_pos = 0
        if is_new_file:
            logger.info(f"New transcript file detected: {latest_key}. Resetting read position to 0.")
            start_read_pos = 0
        elif not has_new_bytes_on_s3 and is_modified:
            logger.warning(f"Tx {latest_key} modified but no size increase detected (State_Pos={last_pos}, S3_Size={current_size}). Assuming potential replacement, reading from start.")
            start_read_pos = 0
        elif has_new_bytes_on_s3:
            logger.info(f"Tx {latest_key} updated. Planning to read from position {last_pos} (S3 size: {current_size}).")
            start_read_pos = last_pos
        else:
            logger.debug(f"Tx {latest_key} modified flag was set, but no action condition met. No read needed.")
            if is_modified: # Update mod time if only mod time changed
                 state.current_latest_key = latest_key
                 state.last_modified[latest_key] = current_modified
                 logger.debug(f"State updated for {latest_key}: Only Mod Time to {current_modified}")
            return None

        # --- Add Pre-Read Check ---
        if start_read_pos >= current_size:
             logger.warning(f"Calculated start read position ({start_read_pos}) is >= current S3 size ({current_size}) for {latest_key}. Skipping read. State position might be ahead.")
             # Update mod time anyway, but don't update position based on this check
             state.current_latest_key = latest_key
             state.last_modified[latest_key] = current_modified
             return None

        # --- Attempt to read new content ---
        read_range = f"bytes={start_read_pos}-"
        new_content = ""; new_content_bytes = b""
        bytes_read = 0
        try:
            logger.debug(f"Attempting S3 get_object: Key={latest_key}, Range={read_range}")
            response = s3_client.get_object(Bucket=aws_s3_bucket, Key=latest_key, Range=read_range)
            new_content_bytes = response['Body'].read()
            bytes_read = len(new_content_bytes)
            logger.info(f"S3 get_object successful. Read {bytes_read} bytes.")
            if bytes_read > 0:
                new_content = new_content_bytes.decode('utf-8', errors='ignore')
                logger.info(f"Decoded {len(new_content)} characters.")
            else:
                # This case might happen if start_read_pos == current_size exactly
                logger.info("Read 0 bytes.")

        except s3_client.exceptions.InvalidRange:
            # This should be less likely with the pre-read check, but handle anyway
            logger.warning(f"S3 InvalidRange error for {latest_key} at position {start_read_pos}. File size {current_size}. Resetting position for next check.")
            state.file_positions[latest_key] = 0 # Reset position for this file
            state.last_modified[latest_key] = current_modified
            state.current_latest_key = latest_key
            return None
        except Exception as get_e:
            logger.error(f"Error reading {latest_key} (range {read_range}): {get_e}", exc_info=True)
            return None

        # --- Update state ---
        state.current_latest_key = latest_key
        state.last_modified[latest_key] = current_modified # Always update mod time if read was attempted

        # **MODIFIED (BACK):** Update position to the current S3 size *after* a successful read attempt.
        # This assumes the read captured everything up to that point.
        new_pos = current_size
        state.file_positions[latest_key] = new_pos
        logger.info(f"State updated for {latest_key}: Position set to {new_pos} (current S3 size).")


        if new_content:
            file_name = os.path.basename(latest_key)
            labeled_content = f"(Source File: {file_name})\n{new_content}"
            return labeled_content
        else:
            # Return None if no actual characters were decoded or read
            return None

    except Exception as e:
        logger.error(f"Unhandled error in read_new_transcript_content for {latest_key}: {e}", exc_info=True)
        return None


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