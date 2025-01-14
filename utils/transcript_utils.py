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
    """Get the latest transcript file, first from agent's event folder"""
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
                transcript_files = [
                    obj['Key'] for obj in response['Contents']
                    if obj['Key'].startswith(prefix) and obj['Key'] != prefix
                    and not obj['Key'].replace(prefix, '').strip('/').count('/')
                    and obj['Key'].endswith('.txt')
                ]
                if transcript_files:
                    logging.debug(f"Found {len(transcript_files)} transcript files in agent folder:")
                    for tf in transcript_files:
                        obj = s3_client.head_object(Bucket=bucket_name, Key=tf)
                        logging.debug(f"  - {tf} (Size: {obj['ContentLength']} bytes, Modified: {obj['LastModified']})")
                    
                    latest_file = max(transcript_files, key=lambda x: s3_client.head_object(Bucket=bucket_name, Key=x)['LastModified'])
                    obj = s3_client.head_object(Bucket=bucket_name, Key=latest_file)
                    logging.debug(f"Selected latest transcript in agent folder: {latest_file}")
                    return latest_file
                    
        # Fallback to default transcripts folder
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
                return latest_file
                
        return None
        
    except Exception as e:
        logging.error(f"Error finding transcript files in S3: {e}")
        return None

def read_new_transcript_content(state, agent_name, event_id, s3_client=None, bucket_name=None):
    """Read only new content from transcript file"""
    try:
        latest_key = get_latest_transcript_file(agent_name, event_id, s3_client, bucket_name)
        if not latest_key:
            logging.debug("No transcript file found")
            return None
            
        if s3_client is None:
            s3_client = boto3.client('s3')
        if bucket_name is None:
            bucket_name = os.getenv('AWS_S3_BUCKET')
            
        response = s3_client.get_object(Bucket=bucket_name, Key=latest_key)
        content = response['Body'].read().decode('utf-8')
        
        if latest_key != state.current_key:
            # New file - read from start
            new_content = content
            state.last_position = len(content)
            logging.debug(f"New transcript file detected, read {len(new_content)} bytes")
        else:
            # Existing file updated - get only new content
            new_content = content[state.last_position:]
            state.last_position = len(content)
            logging.debug(f"Existing file updated, read {len(new_content)} new bytes")
            
        state.current_key = latest_key
        state.last_modified = response['LastModified']
        return new_content
            
    except Exception as e:
        logging.error(f"Error reading transcript: {e}")
        return None