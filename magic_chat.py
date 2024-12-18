import os
import sys
import logging
import time
import argparse
import select
import threading
from anthropic import Anthropic, AnthropicError
from datetime import datetime
import boto3
import json
from models import InsightsOutput
from dotenv import load_dotenv
from config import AppConfig
from web.web_chat import WebChat
import xml.etree.ElementTree as ET
from io import StringIO

SESSION_START_TAG = '<session>'
SESSION_END_TAG = '</session>'
SESSION_END_MARKER = '\n### Chat Session End ###'

abort_requested = False

TOKEN_LIMIT = 4096
AVERAGE_TOKENS_PER_MESSAGE = 50
MAX_MESSAGES = TOKEN_LIMIT // AVERAGE_TOKENS_PER_MESSAGE

# Global transcript position tracker
LAST_TRANSCRIPT_POS = 0

class TranscriptState:
    def __init__(self):
        self.current_key = None
        self.last_position = 0
        self.last_modified = None

def read_new_transcript(transcript_key, agent_name):
    """Read new content from transcript file in S3 starting from last read position"""
    global LAST_TRANSCRIPT_POS
    new_content = ""
    try:
        logging.debug(f"Reading transcript from S3: {transcript_key}")
        logging.debug(f"Current transcript position: {LAST_TRANSCRIPT_POS}")
        
        response = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=transcript_key)
        transcript_stream = response['Body']
        total_size = response.get('ContentLength', 0)
        logging.debug(f"Total transcript size: {total_size} bytes")
        
        transcript_stream.seek(LAST_TRANSCRIPT_POS)
        new_content = transcript_stream.read().decode('utf-8')
        
        if new_content:
            old_pos = LAST_TRANSCRIPT_POS
            LAST_TRANSCRIPT_POS = transcript_stream.tell()
            logging.debug(f"Read {len(new_content)} bytes from position {old_pos} to {LAST_TRANSCRIPT_POS}")
            logging.debug(f"New content preview: {new_content[:100]}...")
        else:
            logging.debug("No new content found in transcript")
            
    except Exception as e:
        logging.error(f"Error reading transcript from S3: {e}")
    return new_content

def read_new_transcript_content(state, agent_name):
    """Read only new content from transcript file"""
    try:
        latest_key = get_latest_transcript_file(agent_name)
        if not latest_key:
            logging.debug("No transcript file found")
            return None
            
        logging.debug(f"Found transcript file: {latest_key}")
        response = s3_client.head_object(Bucket=AWS_S3_BUCKET, Key=latest_key)
        current_modified = response['LastModified']
        current_size = response['ContentLength']
        
        logging.debug(f"Current file: size={current_size}, modified={current_modified}")
        logging.debug(f"Previous state: key={state.current_key}, position={state.last_position}, modified={state.last_modified}")
        
        # If file changed or new file
        if (latest_key != state.current_key or 
            current_modified != state.last_modified):
            
            logging.debug("Transcript file has changed, reading new content")
            response = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=latest_key)
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
            state.last_modified = current_modified
            return new_content
            
        logging.debug("No changes detected in transcript file")
        return None
        
    except Exception as e:
        logging.error(f"Error reading transcript: {e}")
        return None

def check_transcript_updates(transcript_state, conversation_history, agent_name):
    logging.debug("Checking for transcript updates...")
    new_content = read_new_transcript_content(transcript_state, agent_name)
    if new_content:
        logging.debug(f"Adding new transcript content: {new_content[:100]}...")
        conversation_history.append({
            "role": "transcript",
            "content": new_content
        })
        return True
    return False

# Load environment variables from .env file
load_dotenv()

# Retrieve AWS configurations from environment variables
AWS_REGION = os.getenv('AWS_REGION')
AWS_S3_BUCKET = os.getenv('AWS_S3_BUCKET')

# Retrieve API keys from environment variables
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Validate required environment variables
missing_vars = []
if not AWS_REGION:
    missing_vars.append('AWS_REGION')
if not AWS_S3_BUCKET:
    missing_vars.append('AWS_S3_BUCKET')
if not ANTHROPIC_API_KEY:
    missing_vars.append('ANTHROPIC_API_KEY')
if not OPENAI_API_KEY:
    missing_vars.append('OPENAI_API_KEY')

if missing_vars:
    logging.error(f"Missing environment variables in .env file: {', '.join(missing_vars)}")
    sys.exit(1)

# Initialize AWS S3 client
s3_client = boto3.client(
    's3',
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a Claude agent instance.")
    parser.add_argument('--agent', required=True, help='Unique name for the agent.')
    parser.add_argument('--memory', nargs='*', help='Names of agents to load chat history from.', default=None)
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('--listen', action='store_true', help='Enable summary listening at startup.')
    parser.add_argument('--listen-transcript', action='store_true', help='Enable transcript listening at startup.')
    parser.add_argument('--listen-insights', action='store_true', help='Enable insights listening at startup.')
    parser.add_argument('--listen-deep', action='store_true', help='Enable summary and insights listening at startup.')
    parser.add_argument('--listen-all', action='store_true', help='Enable all listening at startup.')
    parser.add_argument('--interface-mode', choices=['cli', 'web', 'web_only'], default='cli', help='Interface mode.')
    return parser.parse_args()

def setup_logging(debug):
    log_filename = 'claude_chat.log'
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.ERROR)

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG if debug else logging.ERROR)
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if debug:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Disable debug logging for external libraries
    logging.getLogger('anthropic').setLevel(logging.WARNING)
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('s3transfer').setLevel(logging.WARNING)

def get_latest_system_prompt(agent_name=None):
    """Get and combine system prompts from S3"""
    try:
        s3_client = boto3.client('s3')
        
        # Get base system prompt
        base_prompt = read_file_content('_config/systemprompt_base.md', "base system prompt")
        
        # Get agent-specific system prompt if agent name is provided
        agent_prompt = ""
        if agent_name:
            agent_prompt_key = f'organizations/river/agents/{agent_name}/_config/systemprompt_aID-{agent_name}.md'
            agent_prompt = read_file_content(agent_prompt_key, "agent system prompt")
        
        # Combine prompts
        system_prompt = base_prompt
        if agent_prompt:
            system_prompt += "\n\n" + agent_prompt
            
        return system_prompt
    except Exception as e:
        logging.error(f"Error getting system prompts: {e}")
        return None

def get_latest_frameworks(agent_name=None):
    """Get and combine frameworks from S3"""
    try:
        s3_client = boto3.client('s3')
        
        # Get base frameworks
        base_frameworks = read_file_content('_config/frameworks_base.md', "base frameworks")
        
        # Get agent-specific frameworks if agent name is provided
        agent_frameworks = ""
        if agent_name:
            agent_frameworks_key = f'organizations/river/agents/{agent_name}/_config/frameworks_aID-{agent_name}.md'
            agent_frameworks = read_file_content(agent_frameworks_key, "agent frameworks")
        
        # Combine frameworks
        frameworks = base_frameworks
        if agent_frameworks:
            frameworks += "\n\n" + agent_frameworks
            
        return frameworks
    except Exception as e:
        logging.error(f"Error getting frameworks: {e}")
        return None

def find_file_any_extension(base_pattern, description):
    """Find a file matching base pattern with any extension in S3.
    Args:
        base_pattern: Base filename pattern without extension (e.g. 'path/to/file')
        description: Description for logging
    Returns:
        Tuple of (file_key, content) or (None, None) if not found
    """
    try:
        # List objects with the base pattern
        prefix = base_pattern.rsplit('/', 1)[0] + '/'
        logging.debug(f"Searching for {description} with prefix '{prefix}'")
        response = s3_client.list_objects_v2(Bucket=AWS_S3_BUCKET, Prefix=prefix)
        
        if 'Contents' in response:
            base_name = base_pattern.rsplit('/', 1)[1]
            logging.debug(f"Found {len(response['Contents'])} objects in prefix '{prefix}'")
            # Find files matching base pattern regardless of extension
            matching_files = [
                obj['Key'] for obj in response['Contents']
                if obj['Key'].rsplit('.', 1)[0] == base_pattern
            ]
            
            if matching_files:
                logging.debug(f"Found {len(matching_files)} matching files for {description}: {matching_files}")
                # Sort by last modified time to get the most recent
                matching_files.sort(
                    key=lambda k: s3_client.head_object(Bucket=AWS_S3_BUCKET, Key=k)['LastModified'],
                    reverse=True
                )
                content = read_file_content(matching_files[0], description)
                if content:
                    logging.debug(f"Successfully loaded content from {matching_files[0]}, length: {len(content)}")
                return matching_files[0], content
        else:
            logging.debug(f"No objects found in prefix '{prefix}'")
        return None, None
        
    except Exception as e:
        logging.error(f"Error finding {description} file for pattern '{base_pattern}': {e}")
        return None, None

def get_latest_context(agent_name, event_id=None):
    """Get and combine contexts from S3"""
    try:
        logging.debug(f"Getting context for agent '{agent_name}' event '{event_id}'")
        # Get organization-specific context
        org_context_base = f'organizations/river/_config/context_oID-{agent_name}'
        logging.debug(f"Looking for organization context at base path: {org_context_base}")
        org_key, org_context = find_file_any_extension(org_context_base, "organization context")
        
        if org_context:
            logging.debug(f"Found organization context at {org_key}")
        
        # Get event-specific context if event ID is provided
        event_context = ""
        if event_id:
            event_context_base = f'organizations/river/agents/{agent_name}/events/{event_id}/_config/context_aID-{agent_name}_eID-{event_id}'
            logging.debug(f"Looking for event context at base path: {event_context_base}")
            event_key, event_context = find_file_any_extension(event_context_base, "event context")
            if event_context:
                logging.debug(f"Found event context at {event_key}")
        
        # Combine contexts
        context = org_context if org_context else ""
        if event_context:
            context += "\n\n" + event_context
            
        if context:
            logging.debug(f"Combined context length: {len(context)}")
        
        return context if context else None
        
    except Exception as e:
        logging.error(f"Error getting contexts: {e}")
        return None

def get_agent_docs(agent_name):
    """Get documentation files for the specified agent."""
    docs_path = f"organizations/river/agents/{agent_name}/docs/"
    docs_content = []
    
    try:
        # List objects in the docs directory
        response = s3_client.list_objects_v2(Bucket=AWS_S3_BUCKET, Prefix=docs_path)
        
        if 'Contents' in response:
            # Filter out directory entry
            doc_files = [obj['Key'] for obj in response['Contents'] if not obj['Key'].endswith('/')]
            logging.debug(f"Found {len(doc_files)} documentation files: {doc_files}")
            
            for doc_file in doc_files:
                content = read_file_content(doc_file, f"documentation file {doc_file}")
                if content:
                    logging.debug(f"\n=== Doc file: {doc_file} ===")
                    logging.debug(f"Content type: {type(content)}")
                    logging.debug(f"First 100 chars: {content[:100]}")
                    logging.debug(f"Last 100 chars: {content[-100:]}")
                    docs_content.append(content)
            
            if docs_content:
                combined_docs = "\n\n".join(docs_content)
                logging.debug(f"\n=== Combined docs ===")
                logging.debug(f"Number of files combined: {len(docs_content)}")
                logging.debug(f"First 100 chars of combined: {combined_docs[:100]}")
                logging.debug(f"Last 100 chars of combined: {combined_docs[-100:]}")
                return combined_docs
                
    except Exception as e:
        logging.error(f"Error loading agent docs: {str(e)}")
    
    logging.debug("No documentation files found for agent")
    return None

def load_existing_chats_from_s3(agent_name, memory_agents=None):
    """Load chat history from S3 for the specified agent(s)"""
    try:
        chat_histories = []
        agents_to_load = [agent_name] if memory_agents is None else memory_agents

        for agent in agents_to_load:
            # Use default event '0000' since events are not yet implemented
            # Only load from saved directory when memory is enabled
            prefix = f'organizations/river/agents/{agent}/events/0000/chats/saved/'
            
            try:
                response = s3_client.list_objects_v2(Bucket=AWS_S3_BUCKET, Prefix=prefix)
                if 'Contents' in response:
                    # Find all chat files regardless of extension
                    chat_files = [
                        obj for obj in response['Contents'] 
                        if obj['Key'].startswith(prefix + 'chat_')
                    ]
                    
                    for chat_file in chat_files:
                        try:
                            chat_content = read_file_content(chat_file['Key'], f"chat file {chat_file['Key']}")
                            if not chat_content:
                                continue
                                
                            # Parse chat content into messages
                            messages = []
                            current_role = None
                            current_content = []
                            
                            for line in chat_content.split('\n'):
                                if line.startswith('**User:**'):
                                    if current_role and current_content:
                                        messages.append({
                                            'role': current_role,
                                            'content': '\n'.join(current_content).strip()
                                        })
                                    current_role = 'user'
                                    current_content = []
                                elif line.startswith('**Agent:**'):
                                    if current_role and current_content:
                                        messages.append({
                                            'role': current_role,
                                            'content': '\n'.join(current_content).strip()
                                        })
                                    current_role = 'assistant'
                                    current_content = []
                                elif line.strip():
                                    current_content.append(line.strip())
                            
                            # Add the last message if exists
                            if current_role and current_content:
                                messages.append({
                                    'role': current_role,
                                    'content': '\n'.join(current_content).strip()
                                })
                            
                            if messages:  # Only add if there are valid messages
                                chat_histories.append({
                                    'agent': agent,
                                    'file': chat_file['Key'],
                                    'messages': messages
                                })
                            
                        except Exception as e:
                            logging.error(f"Error reading chat file {chat_file['Key']}: {e}")
                            continue
                    
            except Exception as e:
                logging.error(f"Error listing chat files for agent {agent}: {e}")
                continue
                
        return chat_histories
        
    except Exception as e:
        logging.error(f"Error loading chat histories from S3: {e}")
        return []

def parse_xml_content(xml_string):
    """Parse XML content and return a formatted string"""
    try:
        # Parse XML
        root = ET.fromstring(xml_string)
        logging.debug(f"XML root element: <{root.tag}>")
        
        # Extract text content recursively
        def extract_text(element, depth=0):
            result = []
            # Add element name as section header if it's not a technical element
            if not element.tag.startswith('{'):
                header = '#' * (depth + 1) + ' ' + element.tag.capitalize()
                result.append(header)
                if depth == 0:  # Log top-level sections
                    logging.debug(f"Processing XML section: {header}")
            
            # Add element text if it exists and is not just whitespace
            if element.text and element.text.strip():
                result.append(element.text.strip())
            
            # Process child elements
            for child in element:
                result.extend(extract_text(child, depth + 1))
                # Add tail text if it exists and is not just whitespace
                if child.tail and child.tail.strip():
                    result.append(child.tail.strip())
            
            return result
        
        # Convert to formatted string
        content = '\n\n'.join(extract_text(root))
        preview = content[:100].replace('\n', '\\n')
        logging.debug(f"XML parsed successfully. Preview of formatted content: {preview}...")
        return content
    except ET.ParseError as e:
        logging.warning(f"Failed to parse XML content, returning raw text: {e}")
        return xml_string

def read_file_content(file_key, description):
    """Read content from S3 file"""
    try:
        # Verify S3 key exists before reading
        try:
            s3_client.head_object(Bucket=AWS_S3_BUCKET, Key=file_key)
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                logging.warning(f"S3 key not found: {file_key}")
                return None
            else:
                raise

        logging.debug(f"Reading {description} from S3: {file_key}")
        response = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=file_key)
        content = response['Body'].read().decode('utf-8')
        
        if content:
            logging.debug(f"Successfully read {description} ({len(content)} chars)")
            return content
        else:
            logging.warning(f"Empty content for {description}")
            return None
            
    except Exception as e:
        logging.error(f"Error reading {description} from S3: {e}")
        return None

def summarize_text(text, max_length=None):
    if max_length is None or len(text) <= max_length:
        return text
    else:
        return text[:max_length] + "..."

def analyze_with_claude(client, messages, system_prompt):
    """Process messages with Claude API, handling transcript updates appropriately"""
    logging.debug(f"\n=== Claude API Request ===")
    logging.debug(f"System prompt length: {len(system_prompt)} chars")
    
    # Format messages for Claude API - handle transcript updates
    formatted_messages = []
    for msg in messages:
        if msg["role"] == "transcript":
            # Add transcript updates as assistant messages
            formatted_messages.append({
                "role": "assistant",
                "content": f"## Transcript Update:\n{msg['content']}"
            })
        else:
            # Include all messages, including system messages
            formatted_messages.append(msg)

    logging.debug(f"Number of messages: {len(formatted_messages)}")
    logging.debug("Message sizes:")
    for i, msg in enumerate(formatted_messages):
        logging.debug(f"  Message {i}: {len(msg['content'])} chars ({msg['role']})")
    
    try:
        max_retries = 5
        initial_backoff = 1
        backoff_factor = 2
        full_response = ""
        request_start_time = time.time()
        info_displayed = False
        response_received = threading.Event()

        response = client.messages.create(
            model="claude-3-opus-20240229",
            system=system_prompt,
            messages=formatted_messages,
            max_tokens=4096
        )
        logging.debug("\n=== Claude API Response ===")
        logging.debug(f"Response length: {len(response.content[0].text)} chars")
        return response.content[0].text
    except Exception as e:
        logging.error(f"Error calling Claude API: {str(e)}")
        return f"Error: {str(e)}"

def save_chat_to_s3(agent_name, chat_content, is_saved=False, filename=None):
    """Save chat content to S3 bucket or copy from archive to saved.
    
    Args:
        agent_name: Name of the agent
        chat_content: Content to append to chat file
        is_saved: Whether this is a manual save (True) or auto-archive (False)
        filename: Optional filename to use, if None one will be generated
        
    Returns:
        Tuple of (success boolean, filename used)
    """
    try:
        if not filename:
            # Generate filename if not provided
            timestamp = datetime.now().strftime('%Y%m%d-T%H%M%S')
            filename = f"chat_D{timestamp}_aID-{agent_name}_eID-{config.event_id}.txt"
            logging.debug(f"Generated new filename: {filename}")
        
        # Base path for both archive and saved folders
        base_path = f"organizations/river/agents/{agent_name}/events/0000/chats"
        archive_key = f"{base_path}/archive/{filename}"
        saved_key = f"{base_path}/saved/{filename}"
        
        if is_saved:
            try:
                # Copy from archive to saved
                copy_source = {
                    'Bucket': AWS_S3_BUCKET,
                    'Key': archive_key
                }
                s3_client.copy_object(
                    CopySource=copy_source,
                    Bucket=AWS_S3_BUCKET,
                    Key=saved_key
                )
                logging.debug(f"Successfully copied from {archive_key} to {saved_key}")
                return True, filename
            except Exception as e:
                logging.error(f"Error copying chat file from archive to saved: {e}")
                return False, None
        else:
            # Regular save to archive
            try:
                # Try to get existing content
                existing_obj = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=archive_key)
                existing_content = existing_obj['Body'].read().decode('utf-8')
                # Append new content
                full_content = existing_content + '\n' + chat_content
            except s3_client.exceptions.NoSuchKey:
                # File doesn't exist yet, use just the new content
                logging.debug(f"File {archive_key} does not exist. Creating new file.")
                full_content = chat_content
            
            # Save the combined content
            s3_client.put_object(
                Bucket=AWS_S3_BUCKET,
                Key=archive_key,
                Body=full_content.encode('utf-8')
            )
            logging.debug(f"Successfully saved to {archive_key}")
            return True, filename
            
    except Exception as e:
        logging.error(f"Error saving chat file {filename}: {e}")
        return False, None

def reload_memory(agent_name, memory_agents, initial_system_prompt):
    """Reload memory from chat history files"""
    previous_chats = load_existing_chats_from_s3(agent_name, memory_agents)
    logging.debug(f"Loaded {len(previous_chats)} chat files for memory")
    
    # Combine all chat content
    all_content = []
    for chat in previous_chats:
        chat_content = []
        for msg in chat['messages']:
            chat_content.append(msg['content'])
        if chat_content:
            all_content.append("\n\n".join(chat_content))
            logging.debug(f"Added chat content from {chat['file']}, length: {len(chat_content[-1])}")
    
    combined_content = "\n\n---\n\n".join(all_content)
    logging.debug(f"Combined content length: {len(combined_content)}")
    
    # Add the content to the system prompt with clear context
    if combined_content:
        new_system_prompt = (
            initial_system_prompt + 
            "\n\n## Previous Chat History\nThe following is a summary of previous chat interactions:\n\n" + 
            combined_content
        )
        logging.debug(f"Final system prompt length: {len(new_system_prompt)}")
    else:
        new_system_prompt = initial_system_prompt
        logging.debug("No chat history to add to system prompt")
    
    return new_system_prompt

def display_help():
    print("\nAvailable commands:")
    print("!help          - Display this help message")
    print("!exit          - Exit the chat")
    print("!clear         - Clear the chat history")
    print("!save          - Save current chat history to S3 saved folder")
    print("!memory        - Toggle memory mode (load chat history)")
    print("!listen        - Enable summary listening")
    print("!listen-all    - Enable all listening modes")
    print("!listen-deep   - Enable summary and insights listening")
    print("!listen-insights - Enable insights listening")
    print("!listen-transcript - Enable transcript listening")
    print("\nStartup flags:")
    print("--memory       - Start with memory mode enabled")
    print("--listen       - Start with summary listening enabled")
    print("--listen-all   - Start with all listening modes enabled")
    print("--listen-deep  - Start with summary and insights listening enabled")
    print("--listen-insights - Start with insights listening enabled")
    print("--listen-transcript - Start with transcript listening enabled")

def format_chat_history(messages):
    chat_content = ""
    for msg in messages:
        if msg["role"] == "user":
            chat_content += f"**User:**\n{msg['content']}\n\n"
        else:
            chat_content += f"**Agent:**\n{msg['content']}\n\n"
    return chat_content

def get_latest_transcript_file(agent_name=None):
    """Get the latest transcript file, first from agent's event folder, then from default folder"""
    try:
        # First try agent's default event folder
        if agent_name:
            prefix = f'organizations/river/agents/{agent_name}/events/0000/transcripts/'
            response = s3_client.list_objects_v2(Bucket=AWS_S3_BUCKET, Prefix=prefix, Delimiter='/')
            
            if 'Contents' in response:
                # Only consider files directly in this folder
                transcript_files = [
                    obj['Key'] for obj in response['Contents'] 
                    if obj['Key'].startswith(prefix) and obj['Key'] != prefix
                    and not obj['Key'].replace(prefix, '').strip('/').count('/')  # No additional folders
                    and obj['Key'].endswith('.txt')
                ]
                if transcript_files:
                    latest_file = max(transcript_files, key=lambda x: s3_client.head_object(Bucket=AWS_S3_BUCKET, Key=x)['LastModified'])
                    logging.debug(f"Found latest transcript in agent folder: {latest_file}")
                    return latest_file
                else:
                    logging.debug(f"No transcript files found in agent folder: {prefix}")
        
        # Fallback to default transcripts folder
        prefix = '_files/transcripts/'
        response = s3_client.list_objects_v2(Bucket=AWS_S3_BUCKET, Prefix=prefix, Delimiter='/')
        
        if 'Contents' in response:
            # Only consider files directly in this folder
            transcript_files = [
                obj['Key'] for obj in response['Contents'] 
                if obj['Key'].startswith(prefix) and obj['Key'] != prefix
                and not obj['Key'].replace(prefix, '').strip('/').count('/')  # No additional folders
                and obj['Key'].endswith('.txt')
            ]
            if transcript_files:
                latest_file = max(transcript_files, key=lambda x: s3_client.head_object(Bucket=AWS_S3_BUCKET, Key=x)['LastModified'])
                logging.debug(f"Found latest transcript in default folder: {latest_file}")
                return latest_file
            else:
                logging.debug(f"No transcript files found in default folder: {prefix}")
                
        logging.debug("No transcript files found.")
        return None
        
    except Exception as e:
        logging.error(f"Error finding transcript files in S3: {e}")
        return None

def main():
    global abort_requested
    try:
        # Load configuration
        config = AppConfig.from_env_and_args()
        
        # Setup logging
        setup_logging(config.debug)
        
        # Initialize chat filename with timestamp at session start
        timestamp = datetime.now().strftime('%Y%m%d-T%H%M%S')
        event_id = "0000"  # Default event ID if not provided
        current_chat_file = f"chat_D{timestamp}_aID-{config.agent_name}_eID-{event_id}.txt"
        logging.debug(f"Initialized chat filename: {current_chat_file}")
        
        # Initialize last saved message index
        last_saved_index = 0
        
        # Start web interface if requested
        if config.interface_mode in ['web', 'web_only']:
            web_interface = WebChat(config)
            web_thread = web_interface.run(port=config.web_port, debug=config.debug)
            print(f"\nWeb interface available at http://127.0.0.1:{config.web_port}")
            
            if config.interface_mode == 'web_only':
                print("\nRunning in web-only mode. Press Ctrl+C to exit.")
                # In web-only mode, just keep the main thread alive
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nShutting down...")
                    return
        
        # Continue with CLI if not web-only
        if config.interface_mode != 'web_only':
            if config.interface_mode == 'web':
                print("CLI interface also available. Type '!help' for commands.\n")
            print(f"Chat agent '{config.agent_name}' is running. Enter your message or type '!help' for commands.\n")
        
            client = Anthropic(api_key=ANTHROPIC_API_KEY)

            # Initialize system prompt with frameworks and context
            system_prompt = get_latest_system_prompt(config.agent_name)
            if not system_prompt:
                logging.error("Failed to load system prompt")
                sys.exit(1)

            # Add frameworks
            frameworks = get_latest_frameworks(config.agent_name)
            if frameworks:
                logging.info("Adding frameworks to system prompt")
                system_prompt += "\n\n=== Frameworks ===\n[Source: _config/frameworks_base.md]\n" + frameworks
            else:
                logging.warning("No frameworks found for agent")
                
            # Add context
            context = get_latest_context(config.agent_name)
            if context:
                logging.info("Adding context to system prompt")
                context_file = f'organizations/river/_config/context_oID-{config.agent_name}.xml'
                system_prompt += f"\n\n=== Context ===\n[Source: {context_file}]\n" + context
            else:
                logging.warning("No context found for agent")
                
            # Add docs if available
            docs = get_agent_docs(config.agent_name)
            if docs:
                logging.info("Adding documentation to system prompt")
                docs_path = f'organizations/river/agents/{config.agent_name}/docs/'
                system_prompt += f"\n\n=== Documentation ===\n[Source: {docs_path}]\n" + docs

            # Load memory after adding all content
            if config.memory is not None:
                if len(config.memory) == 0:
                    config.memory = [config.agent_name]
                system_prompt = reload_memory(config.agent_name, config.memory, system_prompt)

            # Log final system prompt for verification
            logging.debug("\n=== Final System Prompt ===")
            logging.debug(f"Total length: {len(system_prompt)} chars")
            sections = [s for s in system_prompt.split("\n\n=== ") if s.strip()]
            for section in sections:
                lines = section.split("\n")
                if lines[0].endswith(" ==="):
                    section_name = lines[0].replace("===", "").strip()
                    source_line = next((line for line in lines[1:] if line.strip().startswith("[Source:")), "No source file")
                    content = "\n".join(lines[2:])
                    logging.debug(f"\n=== {section_name} ===")
                    logging.debug(f"{source_line}")
                    logging.debug(f"Content length: {len(content)} chars")
                    logging.debug(f"First 100 chars: {content[:100]}")
                    logging.debug(f"Last 100 chars: {content[-100:]}")

            # Initialize transcript handling
            transcript_state = TranscriptState()
            last_transcript_check = time.time()
            TRANSCRIPT_CHECK_INTERVAL = 5  # seconds
            config.listen_transcript_enabled = False  # Start with transcript listening disabled

            # Only load initial content if --listen-transcript flag was used
            if config.listen_transcript:
                config.listen_transcript_enabled = True
                if check_transcript_updates(transcript_state, conversation_history, config.agent_name):
                    print("Initial transcript loaded.")
                last_transcript_check = time.time()

            print("\nUser: ", end='', flush=True)  # Initial prompt
            
            # Main chat loop
            conversation_history = []
            while True:
                try:
                    # Check for transcript updates periodically if enabled
                    current_time = time.time()
                    if config.listen_transcript_enabled and current_time - last_transcript_check > TRANSCRIPT_CHECK_INTERVAL:
                        if check_transcript_updates(transcript_state, conversation_history, config.agent_name):
                            logging.debug("New transcript content added")
                        last_transcript_check = current_time

                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        user_input = sys.stdin.readline().strip()
                        
                        if not user_input:
                            print("\nUser: ", end='', flush=True)
                            continue
                        
                        if user_input.startswith('!'):
                            command = user_input[1:].lower()
                            if command == 'exit':
                                break
                            elif command == 'help':
                                display_help()
                                print("\nUser: ", end='', flush=True)
                                continue
                            elif command == 'clear':
                                conversation_history = []
                                print("\nUser: ", end='', flush=True)
                                continue
                            elif command == 'save':
                                # Save chat history to saved folder
                                new_messages = conversation_history[last_saved_index:]
                                if not new_messages:
                                    print("No new messages to save.")
                                    print("\nUser: ", end='', flush=True)
                                    continue
                                
                                chat_content = format_chat_history(new_messages)
                                logging.debug(f"Saving chat to {current_chat_file}")
                                success, _ = save_chat_to_s3(config.agent_name, chat_content, is_saved=True, filename=current_chat_file)
                                
                                if success:
                                    print(f"Chat history saved to {current_chat_file}")
                                    last_saved_index = len(conversation_history)
                                else:
                                    print("Failed to save chat history")
                                print("\nUser: ", end='', flush=True)
                                continue
                            elif command == 'memory':
                                if config.memory is None:
                                    config.memory = [config.agent_name]
                                    system_prompt = reload_memory(config.agent_name, config.memory, system_prompt)
                                    print("Memory mode activated.")
                                else:
                                    config.memory = None
                                    system_prompt = get_latest_system_prompt(config.agent_name)
                                    print("Memory mode deactivated.")
                                print("\nUser: ", end='', flush=True)
                                continue
                            elif command in ['listen', 'listen-all', 'listen-deep', 'listen-insights', 'listen-transcript']:
                                # Enable transcript loading only for relevant commands
                                if command in ['listen', 'listen-all', 'listen-transcript']:
                                    config.listen_transcript_enabled = True
                                    if check_transcript_updates(transcript_state, conversation_history, config.agent_name):
                                        print("Transcript loaded and automatic listening mode activated.")
                                    else:
                                        print("No new transcript content found.")
                                    last_transcript_check = time.time()  # Reset check timer
                                elif command == 'silent':
                                    config.listen_transcript_enabled = False
                                    print("Transcript listening mode deactivated.")

                                # Handle other listen modes
                                if command in ['listen', 'listen-all', 'listen-deep', 'listen-insights']:
                                    # Existing insights/summary loading code...
                                    pass
                                
                                print("\nUser: ", end='', flush=True)
                                continue
                        
                        # Process user message
                        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        user_content = f"On {current_timestamp}, user said: {user_input}"
                        conversation_history.append({"role": "user", "content": user_content})
                        
                        try:
                            response = analyze_with_claude(client, conversation_history, system_prompt)
                            if response is None:
                                print("\nUser: ", end='', flush=True)
                                continue
                            conversation_history.append({"role": "assistant", "content": response})
                            
                            # Format and save only the latest message round
                            latest_messages = conversation_history[-2:]  # Get user message and assistant response
                            chat_content = format_chat_history(latest_messages)
                            logging.debug(f"Saving latest message round to {current_chat_file}")
                            success, _ = save_chat_to_s3(config.agent_name, chat_content, is_saved=False, filename=current_chat_file)
                            
                            if not success:
                                print("Failed to save chat history")
                            
                            print("\nUser: ", end='', flush=True)  # Add prompt for next input
                            
                        except Exception as e:
                            logging.error(f"Error processing message: {e}")
                            print(f"\nError: {e}")
                            print("\nUser: ", end='', flush=True)  # Add prompt even after error
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting the chat.")
                    break

    except Exception as e:
        logging.error(f"Error in main loop: {e}")

if __name__ == '__main__':
    main()
