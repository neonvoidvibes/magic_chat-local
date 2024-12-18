"""S3 utilities for the web chat application."""
import os
import boto3
from datetime import datetime
import json
import logging

# S3 configuration
AWS_S3_BUCKET = os.getenv('AWS_S3_BUCKET', 'aiademomagicaudio')

def read_file_content(s3_key, file_name):
    """Read content from an S3 file."""
    try:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=AWS_S3_BUCKET, Key=s3_key)
        return obj['Body'].read().decode('utf-8')
    except Exception as e:
        print(f"Error reading {file_name} from S3: {e}")
        return None

def save_chat_to_s3(chat_history, filename, agent_name):
    """Save chat history to S3."""
    try:
        s3 = boto3.client('s3')
        chat_key = f'organizations/river/agents/{agent_name}/chats/{filename}'
        
        # Convert chat history to JSON string
        chat_json = json.dumps({
            'messages': chat_history,
            'timestamp': datetime.now().isoformat()
        })
        
        # Upload to S3
        s3.put_object(
            Bucket=AWS_S3_BUCKET,
            Key=chat_key,
            Body=chat_json.encode('utf-8')
        )
        return True
    except Exception as e:
        print(f"Error saving chat to S3: {e}")
        return False

def load_existing_chats_from_s3(agent_name, max_chats=5):
    """Load existing chat histories from S3."""
    try:
        s3 = boto3.client('s3')
        prefix = f'organizations/river/agents/{agent_name}/chats/'
        
        # List chat files
        response = s3.list_objects_v2(
            Bucket=AWS_S3_BUCKET,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            return []
            
        # Sort by last modified time
        chat_files = sorted(
            response['Contents'],
            key=lambda x: x['LastModified'],
            reverse=True
        )[:max_chats]
        
        # Load chat contents
        chats = []
        for file in chat_files:
            content = read_file_content(file['Key'], 'chat history')
            if content:
                try:
                    chat_data = json.loads(content)
                    chats.append(chat_data)
                except json.JSONDecodeError:
                    print(f"Error decoding chat file {file['Key']}")
                    
        return chats
    except Exception as e:
        print(f"Error loading chats from S3: {e}")
        return []

def summarize_text(text, max_length=100):
    """Create a brief summary of text."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_length:
        return text
    return " ".join(words[:max_length]) + "..."

def find_file_by_base(base_path, description):
    """Find a file by its base path, ignoring extension."""
    try:
        # List objects with the prefix
        prefix = os.path.dirname(base_path)
        base_name = os.path.basename(base_path).split('.')[0]
        
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=AWS_S3_BUCKET, Prefix=prefix)
        
        if 'Contents' not in response:
            return None
            
        # Find matching file
        for obj in response['Contents']:
            obj_base = os.path.basename(obj['Key']).split('.')[0]
            if obj_base == base_name:
                return obj['Key']
                
        return None
    except Exception as e:
        logging.error(f"Error finding {description}: {e}")
        return None

def get_latest_system_prompt(agent_name):
    """Get the latest system prompt for an agent."""
    try:
        # Read base system prompt
        base_key = find_file_by_base('_config/systemprompt_base', 'base system prompt')
        base_content = read_file_content(base_key, 'base system prompt')
        
        # Try to read agent-specific system prompt
        agent_key = find_file_by_base(
            f'organizations/river/agents/{agent_name}/_config/systemprompt_aID-{agent_name}',
            'agent system prompt'
        )
        agent_content = read_file_content(agent_key, 'agent system prompt')
        
        if base_content:
            return agent_content + "\n\n" + base_content if agent_content else base_content
        return None
    except Exception as e:
        print(f"Error reading system prompt: {e}")
        return None

def read_frameworks(agent_name):
    """Read frameworks for an agent."""
    try:
        # Read base frameworks
        base_key = find_file_by_base('_config/frameworks_base', 'base frameworks')
        base_content = read_file_content(base_key, 'base frameworks')
        
        # Try to read agent-specific frameworks
        agent_key = find_file_by_base(
            f'organizations/river/agents/{agent_name}/_config/frameworks_aID-{agent_name}',
            'agent frameworks'
        )
        agent_content = read_file_content(agent_key, 'agent frameworks')
        
        if base_content:
            return agent_content + "\n\n" + base_content if agent_content else base_content
        return None
    except Exception as e:
        print(f"Error reading frameworks: {e}")
        return None

def read_organization_context(agent_name):
    """Read organization context."""
    try:
        context_key = find_file_by_base(f'organizations/river/_config/context_oID-{agent_name}', 'organization context')
        return read_file_content(context_key, 'organization context')
    except Exception as e:
        print(f"Error reading organization context: {e}")
        return None

def read_agent_docs(agent_name):
    """Read agent documentation."""
    try:
        s3 = boto3.client('s3')
        prefix = f'organizations/river/agents/{agent_name}/docs/'
        
        response = s3.list_objects_v2(
            Bucket=AWS_S3_BUCKET,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            return None
            
        docs = []
        for obj in response['Contents']:
            content = read_file_content(obj['Key'], 'agent documentation')
            if content:
                docs.append(content)
                    
        return "\n\n".join(docs) if docs else None
    except Exception as e:
        print(f"Error reading agent docs: {e}")
        return None

def list_context_files(prefix):
    """List context files in a prefix."""
    try:
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=AWS_S3_BUCKET, Prefix=prefix)
        if 'Contents' not in response:
            return []
            
        context_files = []
        for obj in response['Contents']:
            context_files.append(obj['Key'])
        return context_files
    except Exception as e:
        logging.error(f"Error listing context files: {e}")
        return []
