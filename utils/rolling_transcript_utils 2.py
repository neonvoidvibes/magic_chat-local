"""Utilities for managing rolling transcripts."""
import logging
import boto3
import os
from datetime import datetime

def get_latest_rolling_transcript(agent_name=None, event_id=None, s3_client=None, bucket_name=None):
    """Get the latest rolling transcript file from the agent's event folder."""
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
                # Filter for rolling transcript files
                rolling_files = [
                    obj['Key'] for obj in response['Contents']
                    if obj['Key'].startswith(prefix) and obj['Key'] != prefix
                    and not obj['Key'].replace(prefix, '').strip('/').count('/')
                    and obj['Key'].endswith('.txt')
                    and obj['Key'].replace(prefix, '').startswith('rolling-')
                ]
                if rolling_files:
                    logging.debug(f"Found {len(rolling_files)} rolling transcript files in agent folder:")
                    for rf in rolling_files:
                        obj = s3_client.head_object(Bucket=bucket_name, Key=rf)
                        logging.debug(f"  - {rf} (Size: {obj['ContentLength']} bytes, Modified: {obj['LastModified']})")
                    
                    # Get the latest by LastModified timestamp
                    latest_file = max(rolling_files, key=lambda x: s3_client.head_object(Bucket=bucket_name, Key=x)['LastModified'])
                    obj = s3_client.head_object(Bucket=bucket_name, Key=latest_file)
                    logging.debug(f"Selected latest rolling transcript in agent folder: {latest_file}")
                    return latest_file
                    
        # Fallback to default transcripts folder
        prefix = '_files/transcripts/'
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
        
        if 'Contents' in response:
            rolling_files = [
                obj['Key'] for obj in response['Contents']
                if obj['Key'].startswith(prefix) and obj['Key'] != prefix
                and not obj['Key'].replace(prefix, '').strip('/').count('/')
                and obj['Key'].endswith('.txt')
                and obj['Key'].replace(prefix, '').startswith('rolling-')
            ]
            if rolling_files:
                latest_file = max(rolling_files, key=lambda x: s3_client.head_object(Bucket=bucket_name, Key=x)['LastModified'])
                return latest_file
                
        return None
        
    except Exception as e:
        logging.error(f"Error finding rolling transcript files in S3: {e}")
        return None
