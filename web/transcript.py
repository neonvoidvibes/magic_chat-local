"""Transcript handling utilities for the web chat application."""
import os
import boto3
from datetime import datetime

# S3 configuration
AWS_S3_BUCKET = os.getenv('AWS_S3_BUCKET', 'aiademomagicaudio')

def get_latest_transcript_file(agent_name):
    """Get the latest transcript file for an agent."""
    try:
        s3 = boto3.client('s3')
        prefix = f'organizations/river/agents/{agent_name}/transcripts/'
        
        # List objects in the transcript directory
        response = s3.list_objects_v2(
            Bucket=AWS_S3_BUCKET,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            return None
            
        # Find the most recent file
        latest = None
        latest_time = None
        
        for obj in response['Contents']:
            if obj['Key'].endswith('.txt'):
                if latest is None or obj['LastModified'] > latest_time:
                    latest = obj['Key']
                    latest_time = obj['LastModified']
        
        return latest
    except Exception as e:
        print(f"Error getting latest transcript: {e}")
        return None

def read_new_transcript_content(transcript_state, agent_name):
    """Read new content from the transcript file."""
    if not transcript_state:
        return None
        
    try:
        s3 = boto3.client('s3')
        current_file = get_latest_transcript_file(agent_name)
        
        if not current_file:
            return None
            
        # Check if this is a new file
        if current_file != transcript_state.last_file:
            transcript_state.last_file = current_file
            transcript_state.last_position = 0
            
        # Get the file content from the last position
        response = s3.get_object(
            Bucket=AWS_S3_BUCKET,
            Key=current_file,
            Range=f'bytes={transcript_state.last_position}-'
        )
        
        new_content = response['Body'].read().decode('utf-8')
        if new_content:
            transcript_state.last_position += len(new_content.encode('utf-8'))
            return new_content
            
        return None
    except Exception as e:
        print(f"Error reading transcript: {e}")
        return None
