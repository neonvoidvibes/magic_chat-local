import logging
import boto3
import os
from datetime import datetime

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
                # First priority: rolling transcripts in agent folder
                rolling_files = [
                    obj['Key'] for obj in response['Contents']
                    if obj['Key'].startswith(prefix) and obj['Key'] != prefix
                    and not obj['Key'].replace(prefix, '').strip('/').count('/')
                    and obj['Key'].endswith('.txt')
                    and obj['Key'].replace(prefix, '').startswith('rolling-')
                ]
                if rolling_files:
                    latest_file = max(rolling_files, key=lambda x: s3_client.head_object(Bucket=bucket_name, Key=x)['LastModified'])
                    logging.debug(f"Selected latest rolling transcript in agent folder: {latest_file}")
                    return latest_file
                
                # Second priority: regular transcripts in agent folder
                transcript_files = [
                    obj['Key'] for obj in response['Contents']
                    if obj['Key'].startswith(prefix) and obj['Key'] != prefix
                    and not obj['Key'].replace(prefix, '').strip('/').count('/')
                    and obj['Key'].endswith('.txt')
                    and not obj['Key'].replace(prefix, '').startswith('rolling-')
                ]
                if transcript_files:
                    latest_file = max(transcript_files, key=lambda x: s3_client.head_object(Bucket=bucket_name, Key=x)['LastModified'])
                    logging.debug(f"Selected latest transcript in agent folder: {latest_file}")
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
    Read only new content from one or multiple transcripts, depending on read_all.
    When read_all=True, we combine partial updates from all transcripts following priority order:
    1. rolling-transcript_ (agent event folder)
    2. transcript_ (agent event folder)
    3. transcript_ (general folder)
    Otherwise, we stick to single 'latest' transcript approach using same priority.
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