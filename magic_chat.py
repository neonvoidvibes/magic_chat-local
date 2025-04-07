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
from typing import Optional, List, Dict, Any # Added Any
from utils.retrieval_handler import RetrievalHandler
from config import AppConfig
from web.web_chat import WebChat
import xml.etree.ElementTree as ET
from io import StringIO
from utils.transcript_utils import TranscriptState, get_latest_transcript_file, read_new_transcript_content, read_all_transcripts_in_folder

SESSION_START_TAG = '<session>'
SESSION_END_TAG = '</session>'
SESSION_END_MARKER = '\n### Chat Session End ###'

abort_requested = False

TOKEN_LIMIT = 4096
AVERAGE_TOKENS_PER_MESSAGE = 50
MAX_MESSAGES = TOKEN_LIMIT // AVERAGE_TOKENS_PER_MESSAGE

# Global transcript position tracker (potentially problematic with multiple instances/threads)
# Consider moving state management if scaling
LAST_TRANSCRIPT_POS = 0

def check_transcript_updates(transcript_state: TranscriptState, conversation_history: List[Dict[str, Any]], agent_name: str, event_id: str, read_all: bool = False) -> bool:
    """Checks for transcript updates and appends new content to history."""
    logging.debug("Checking for transcript updates...")
    new_content = read_new_transcript_content(transcript_state, agent_name, event_id, read_all=read_all)
    if new_content:
        logging.debug(f"Adding new transcript content: {new_content[:100]}...")
        # Add as a user message for the LLM to see in context
        conversation_history.append({
            "role": "user",
            "content": f"[LIVE TRANSCRIPT UPDATE]\n{new_content}" # Label it clearly
        })
        return True
    return False

# Load environment variables from .env file - load_dotenv() is called in AppConfig now
# load_dotenv()

# AWS and API Key retrieval is handled by AppConfig now
# Initialize S3 client (consider lazy initialization or passing from main)
s3_client = None
def get_s3_client():
    """Initializes and returns an S3 client."""
    global s3_client
    if s3_client is None:
        try:
            s3_client = boto3.client(
                's3',
                region_name=os.getenv('AWS_REGION'),
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
            logging.info("S3 client initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize S3 client: {e}")
            # Depending on usage, might want to raise an error here
    return s3_client

# Argument parsing is handled by AppConfig now

def setup_logging(debug: bool):
    """Sets up logging configuration."""
    log_filename = 'claude_chat.log'
    # Get the root logger
    logger = logging.getLogger()

    # Set level based on debug flag
    log_level = logging.DEBUG if debug else logging.INFO # Changed non-debug to INFO
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates if called multiple times
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler
    try:
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(log_level) # Match root logger level
        file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s') # Added logger name
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logger: {e}", file=sys.stderr)


    # Console handler (always add INFO and above, add DEBUG if debug=True)
    console_handler = logging.StreamHandler(sys.stdout)
    console_log_level = logging.DEBUG if debug else logging.INFO
    console_handler.setLevel(console_log_level)
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s') # Simpler format for console
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)


    # Reduce noise from libraries
    logging.getLogger('anthropic').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING) # httpx can be noisy
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('s3transfer').setLevel(logging.WARNING)
    logging.getLogger('pinecone').setLevel(logging.INFO) # Allow INFO for Pinecone

    logging.info(f"Logging setup complete. Level: {logging.getLevelName(log_level)}")


def find_file_any_extension(base_pattern: str, description: str) -> Optional[tuple[str, str]]:
    """Find a file matching base pattern with any extension in S3.
    Args:
        base_pattern: Base filename pattern without extension (e.g. 'path/to/file')
        description: Description for logging
    Returns:
        Tuple of (file_key, content) or None if not found/error
    """
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        logging.error(f"S3 client or bucket not available for finding {description}.")
        return None

    try:
        # List objects with the base pattern's directory prefix
        prefix = ""
        base_name = base_pattern
        if '/' in base_pattern:
             prefix = base_pattern.rsplit('/', 1)[0] + '/'
             base_name = base_pattern.rsplit('/', 1)[1]

        logging.debug(f"Searching for {description} with prefix '{prefix}' and base name '{base_name}'")
        paginator = s3.get_paginator('list_objects_v2')
        matching_files = []

        for page in paginator.paginate(Bucket=aws_s3_bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Check if the filename part (without extension) matches the base_name
                    if '/' in key:
                         filename_part = key.rsplit('/', 1)[1]
                    else:
                         filename_part = key # Should not happen if prefix is used correctly, but safe check

                    # Split filename from its extension
                    name_only, ext = os.path.splitext(filename_part)

                    # Ensure we match the base name exactly (case-sensitive)
                    if name_only == base_name:
                        matching_files.append(obj) # Store the whole object for LastModified

        if not matching_files:
             logging.warning(f"No files found matching base pattern '{base_pattern}' for {description}.")
             return None

        logging.debug(f"Found {len(matching_files)} potential files for {description}.")
        # Sort by last modified time to get the most recent
        matching_files.sort(key=lambda obj: obj['LastModified'], reverse=True)
        latest_file_key = matching_files[0]['Key']

        logging.debug(f"Latest file found for {description}: {latest_file_key}")
        content = read_file_content(latest_file_key, description)
        if content is not None:
            logging.debug(f"Successfully loaded content from {latest_file_key}, length: {len(content)}")
            return latest_file_key, content
        else:
            logging.error(f"Failed to read content from {latest_file_key} for {description}.")
            return None

    except Exception as e:
        logging.error(f"Error finding {description} file for pattern '{base_pattern}': {e}", exc_info=True)
        return None

def get_latest_system_prompt(agent_name: Optional[str] = None) -> Optional[str]:
    """Get and combine system prompts from S3"""
    try:
        # Get base system prompt
        base_result = find_file_any_extension('_config/systemprompt_base', "base system prompt")
        base_prompt = base_result[1] if base_result else None

        if not base_prompt:
             logging.error("Base system prompt not found or failed to load.")
             # Return None or a default minimal prompt
             return "You are a helpful assistant." # Fallback

        # Get agent-specific system prompt if agent name is provided
        agent_prompt = ""
        if agent_name:
            agent_result = find_file_any_extension(
                f'organizations/river/agents/{agent_name}/_config/systemprompt_aID-{agent_name}',
                "agent system prompt"
            )
            if agent_result:
                agent_prompt = agent_result[1]
                logging.info(f"Loaded agent-specific system prompt for '{agent_name}'.")
            else:
                 logging.warning(f"No agent-specific system prompt found for '{agent_name}'.")

        # Combine prompts
        system_prompt = base_prompt
        if agent_prompt:
            system_prompt += "\n\n" + agent_prompt

        return system_prompt
    except Exception as e:
        logging.error(f"Error getting system prompts: {e}", exc_info=True)
        return None # Or fallback prompt

def get_latest_frameworks(agent_name: Optional[str] = None) -> Optional[str]:
    """Get and combine frameworks from S3"""
    try:
        base_result = find_file_any_extension('_config/frameworks_base', "base frameworks")
        base_frameworks = base_result[1] if base_result else "" # Default to empty string

        agent_frameworks = ""
        if agent_name:
            agent_result = find_file_any_extension(
                f'organizations/river/agents/{agent_name}/_config/frameworks_aID-{agent_name}',
                "agent frameworks"
            )
            if agent_result:
                agent_frameworks = agent_result[1]
                logging.info(f"Loaded agent-specific frameworks for '{agent_name}'.")
            else:
                 logging.warning(f"No agent-specific frameworks found for '{agent_name}'.")

        # Combine frameworks
        frameworks = base_frameworks
        if agent_frameworks:
            frameworks += ("\n\n" + agent_frameworks) if frameworks else agent_frameworks

        return frameworks if frameworks else None # Return None if totally empty
    except Exception as e:
        logging.error(f"Error getting frameworks: {e}", exc_info=True)
        return None

def get_latest_context(agent_name: str, event_id: Optional[str] = None) -> Optional[str]:
    """Get and combine contexts from S3, with optional event_id"""
    # Note: Org context seems misnamed in original find call (using agent_name for org id)
    # Let's assume it means agent's org, maybe derive org from agent? For now, use as is.
    try:
        org_context = ""
        org_result = find_file_any_extension(
            f'organizations/river/_config/context_oID-{agent_name}', # Check this path logic
            "organization context"
        )
        if org_result:
             org_context = org_result[1]
             logging.info("Loaded organization context.")
        else:
             logging.warning(f"No organization context found using pattern 'organizations/river/_config/context_oID-{agent_name}'.")

        event_context = ""
        # Only look for event context if event_id is specific (not None or '0000')
        if event_id and event_id != '0000':
            event_result = find_file_any_extension(
                f'organizations/river/agents/{agent_name}/events/{event_id}/_config/context_aID-{agent_name}_eID-{event_id}',
                "event context"
            )
            if event_result:
                event_context = event_result[1]
                logging.info(f"Loaded event-specific context for event '{event_id}'.")
            else:
                 logging.warning(f"No event-specific context found for event '{event_id}'.")

        # Combine contexts
        context = org_context
        if event_context:
            context += ("\n\n" + event_context) if context else event_context

        return context if context else None
    except Exception as e:
        logging.error(f"Error getting contexts: {e}", exc_info=True)
        return None

def get_agent_docs(agent_name: str) -> Optional[str]:
    """Get documentation files for the specified agent."""
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        logging.error("S3 client or bucket not available for getting agent docs.")
        return None

    try:
        prefix = f'organizations/river/agents/{agent_name}/docs/'
        logging.debug(f"Searching for agent documentation in S3 prefix '{prefix}'")

        paginator = s3.get_paginator('list_objects_v2')
        docs = []
        for page in paginator.paginate(Bucket=aws_s3_bucket, Prefix=prefix):
             if 'Contents' in page:
                  for obj in page['Contents']:
                      key = obj['Key']
                      # Skip if it's the directory placeholder itself
                      if key == prefix or key.endswith('/'):
                          continue
                      # Assume all files in docs are relevant text/markdown
                      content = read_file_content(key, 'agent documentation')
                      if content:
                          # Add source filename for clarity in the combined doc string
                          filename = os.path.basename(key)
                          docs.append(f"--- START {filename} ---\n{content}\n--- END {filename} ---")

        if not docs:
            logging.warning(f"No documentation files found in '{prefix}'")
            return None

        logging.info(f"Found and loaded {len(docs)} documentation files for agent '{agent_name}'.")
        return "\n\n".join(docs)
    except Exception as e:
        logging.error(f"Error getting agent documentation: {e}", exc_info=True)
        return None

def load_existing_chats_from_s3(agent_name: str, memory_agents: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Load chat history from S3 for the specified agent(s).
       Focuses on the 'saved' directory for explicit memory.
    """
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        logging.error("S3 client or bucket not available for loading chats.")
        return []

    chat_histories = []
    agents_to_load = memory_agents if memory_agents else [agent_name] # Default to self if None/empty

    logging.info(f"Loading saved chat history for agents: {agents_to_load}")

    for agent in agents_to_load:
        # Hardcoding event '0000' for now, assuming memory is not event-specific yet
        # ONLY loading from saved directory
        prefix = f'organizations/river/agents/{agent}/events/0000/chats/saved/'
        logging.debug(f"Checking for saved chats in prefix: {prefix}")

        try:
            paginator = s3.get_paginator('list_objects_v2')
            chat_files_to_process = []
            for page in paginator.paginate(Bucket=aws_s3_bucket, Prefix=prefix):
                if 'Contents' in page:
                    # Filter for actual chat files (e.g., .txt or .json)
                    chat_files_to_process.extend(
                        obj for obj in page['Contents']
                        if not obj['Key'].endswith('/') and (obj['Key'].endswith('.txt') or obj['Key'].endswith('.json'))
                    )

            if not chat_files_to_process:
                 logging.debug(f"No saved chat files found for agent {agent} in {prefix}")
                 continue

            # Sort by last modified (most recent first) and limit? Or load all? Load all for now.
            chat_files_to_process.sort(key=lambda obj: obj['LastModified'], reverse=True)
            logging.info(f"Found {len(chat_files_to_process)} saved chat files for agent {agent}.")

            for chat_obj in chat_files_to_process:
                file_key = chat_obj['Key']
                logging.debug(f"Reading saved chat file: {file_key}")
                try:
                    chat_content_str = read_file_content(file_key, f"saved chat file {file_key}")
                    if not chat_content_str:
                        logging.warning(f"Empty content for chat file {file_key}")
                        continue

                    # Attempt to parse based on extension or content
                    messages = []
                    if file_key.endswith('.json'):
                        try:
                            chat_data = json.loads(chat_content_str)
                            # Assuming structure like {'messages': [{'role': 'user', 'content': '...'}, ...]}
                            messages = chat_data.get('messages', [])
                            if not isinstance(messages, list): messages = [] # Ensure it's a list
                        except json.JSONDecodeError as json_err:
                            logging.warning(f"Failed to parse JSON chat file {file_key}: {json_err}. Trying text parse.")
                            # Fallback to text parsing
                            messages = parse_text_chat(chat_content_str)
                    else: # Assume .txt or other is text format
                        messages = parse_text_chat(chat_content_str)

                    if messages:
                        chat_histories.append({
                            'agent': agent,
                            'file': file_key,
                            'messages': messages # List of {'role': ..., 'content': ...}
                        })
                    else:
                         logging.warning(f"No valid messages extracted from chat file {file_key}")

                except Exception as read_err:
                    logging.error(f"Error reading or parsing chat file {file_key}: {read_err}", exc_info=True)
                    continue # Skip this file

        except Exception as list_err:
            logging.error(f"Error listing chat files for agent {agent}: {list_err}", exc_info=True)
            continue # Skip this agent

    logging.info(f"Loaded {len(chat_histories)} chat history files overall.")
    return chat_histories

def parse_text_chat(chat_content_str: str) -> List[Dict[str, str]]:
    """Parses chat content from a text format (e.g., **User:** ... **Agent:** ...)"""
    messages = []
    current_role = None
    current_content = []
    lines = chat_content_str.splitlines()

    for line in lines:
        line_strip = line.strip()
        is_new_speaker = False
        if line_strip.startswith('**User:**'):
            if current_role and current_content:
                 messages.append({'role': current_role, 'content': '\n'.join(current_content).strip()})
            current_role = 'user'
            current_content = [line_strip[len('**User:**'):].strip()] # Start content after marker
            is_new_speaker = True
        elif line_strip.startswith('**Agent:**'):
            if current_role and current_content:
                 messages.append({'role': current_role, 'content': '\n'.join(current_content).strip()})
            current_role = 'assistant'
            current_content = [line_strip[len('**Agent:**'):].strip()] # Start content after marker
            is_new_speaker = True

        if not is_new_speaker and current_role:
             # Only append if it's part of the current speaker's message and not just whitespace
             if line_strip or line: # Keep empty lines within a message block if desired
                  current_content.append(line) # Append original line to preserve formatting

    # Add the last message
    if current_role and current_content:
        messages.append({'role': current_role, 'content': '\n'.join(current_content).strip()})

    # Basic validation: Ensure role and content exist
    validated_messages = [msg for msg in messages if 'role' in msg and 'content' in msg and msg['content']]
    return validated_messages


def read_file_content(file_key: str, description: str) -> Optional[str]:
    """Read content from S3 file, handling potential errors."""
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        logging.error(f"S3 client or bucket not available for reading {description}.")
        return None
    try:
        logging.debug(f"Reading {description} from S3: s3://{aws_s3_bucket}/{file_key}")
        response = s3.get_object(Bucket=aws_s3_bucket, Key=file_key)
        content = response['Body'].read().decode('utf-8')

        if content:
            logging.debug(f"Successfully read {description} ({len(content)} chars)")
            return content
        else:
            logging.warning(f"Empty content for {description} file: {file_key}")
            return "" # Return empty string instead of None for empty files

    except s3.exceptions.NoSuchKey:
        logging.warning(f"{description} file not found at S3 key: {file_key}")
        return None
    except Exception as e:
        logging.error(f"Error reading {description} from S3 key {file_key}: {e}", exc_info=True)
        return None

# analyze_with_claude needs the model_name parameter
def analyze_with_claude(client: Anthropic, messages: List[Dict[str, Any]], system_prompt: str, model_name: str) -> Optional[str]:
    """Process messages with Claude API, handling transcript updates appropriately"""
    logging.debug(f"\n=== Claude API Request ===")
    logging.debug(f"Using model: {model_name}")
    logging.debug(f"System prompt length: {len(system_prompt)} chars")

    # Ensure messages have valid roles ('user' or 'assistant') for the API
    formatted_messages = []
    for msg in messages:
         role = msg.get("role")
         content = msg.get("content", "")
         # Skip empty messages or system messages within the history
         if not content or role == "system":
              continue
         # Map 'transcript' or other custom roles to 'user' for the API call
         api_role = "assistant" if role == "assistant" else "user"
         formatted_messages.append({"role": api_role, "content": content})


    # Add instruction to not summarize transcript updates if present
    transcript_instruction = "\nIMPORTANT: When you receive transcript updates (marked with [LIVE TRANSCRIPT UPDATE]), do not summarize them. Simply acknowledge that you've received the update and continue the conversation based on the new information."
    final_system_prompt = system_prompt + transcript_instruction

    logging.debug(f"Number of messages for API: {len(formatted_messages)}")
    if logging.getLogger().isEnabledFor(logging.DEBUG):
         for i, msg in enumerate(formatted_messages[-5:]): # Log last 5 messages
              logging.debug(f"  Message {len(formatted_messages)-5+i}: Role={msg['role']}, Length={len(msg['content'])}, Content='{msg['content'][:100]}...'")

    try:
        # Use the passed model_name
        response = client.messages.create(
            model=model_name,
            system=final_system_prompt,
            messages=formatted_messages,
            max_tokens=4096 # Consider making this configurable
        )
        response_text = response.content[0].text
        logging.debug("\n=== Claude API Response ===")
        logging.debug(f"Response length: {len(response_text)} chars")
        logging.debug(f"Response text (first 100): {response_text[:100]}...")
        return response_text
    except AnthropicError as e:
         logging.error(f"Anthropic API Error calling Claude model {model_name}: {e}")
         return f"Error communicating with AI: {e}"
    except Exception as e:
        logging.error(f"Unexpected error calling Claude model {model_name}: {e}", exc_info=True)
        return f"Unexpected error: {e}"

def save_chat_to_s3(agent_name: str, chat_content: str, event_id: Optional[str], is_saved: bool = False, filename: Optional[str] = None) -> tuple[bool, Optional[str]]:
    """Save chat content to S3 bucket (archive or saved folder).

    Args:
        agent_name: Name of the agent.
        chat_content: Formatted string content to save.
        event_id: Event ID for folder path (defaults to '0000').
        is_saved: If True, copy from archive to saved. If False, save/append to archive.
        filename: Optional filename to use; if None, one will be generated.

    Returns:
        Tuple of (success boolean, filename used or None on failure).
    """
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        logging.error("S3 client or bucket not available for saving chat.")
        return False, None

    event_id = event_id or '0000' # Ensure default if None

    try:
        if not filename:
            # Generate filename if not provided
            timestamp = datetime.now().strftime('%Y%m%d-T%H%M%S')
            filename = f"chat_D{timestamp}_aID-{agent_name}_eID-{event_id}.txt" # Assume .txt for now
            logging.debug(f"Generated new chat filename: {filename}")

        # Define paths
        base_path = f"organizations/river/agents/{agent_name}/events/{event_id}/chats"
        archive_key = f"{base_path}/archive/{filename}"
        saved_key = f"{base_path}/saved/{filename}"

        if is_saved:
            # --- Copy from archive to saved ---
            logging.info(f"Attempting to save chat by copying from {archive_key} to {saved_key}")
            try:
                 # Check if source (archive) exists first
                 s3.head_object(Bucket=aws_s3_bucket, Key=archive_key)

                 copy_source = {'Bucket': aws_s3_bucket, 'Key': archive_key}
                 s3.copy_object(
                    CopySource=copy_source,
                    Bucket=aws_s3_bucket,
                    Key=saved_key
                 )
                 logging.info(f"Successfully copied chat to saved folder: {saved_key}")
                 return True, filename
            except s3.exceptions.ClientError as e:
                 if e.response['Error']['Code'] == '404':
                      logging.error(f"Cannot save: Archive file {archive_key} does not exist to copy.")
                      return False, None
                 else:
                      logging.error(f"Error checking/copying chat file {archive_key} to {saved_key}: {e}", exc_info=True)
                      return False, None
            except Exception as e:
                 logging.error(f"Unexpected error copying chat file {archive_key} to {saved_key}: {e}", exc_info=True)
                 return False, None
        else:
            # --- Append to or create archive file ---
            logging.info(f"Attempting to save/append chat content to archive: {archive_key}")
            try:
                # Try to get existing content
                existing_obj = s3.get_object(Bucket=aws_s3_bucket, Key=archive_key)
                existing_content = existing_obj['Body'].read().decode('utf-8')
                # Append new content with a separator
                full_content = existing_content.strip() + '\n\n' + chat_content.strip()
                logging.debug(f"Appending {len(chat_content)} chars to existing archive file.")
            except s3.exceptions.NoSuchKey:
                # File doesn't exist yet, use just the new content
                logging.debug(f"Archive file {archive_key} does not exist. Creating new file.")
                full_content = chat_content.strip()
            except Exception as get_e:
                 logging.error(f"Error reading existing archive file {archive_key}: {get_e}", exc_info=True)
                 return False, None # Don't proceed if reading failed

            # Save the combined content
            try:
                s3.put_object(
                    Bucket=aws_s3_bucket,
                    Key=archive_key,
                    Body=full_content.encode('utf-8'),
                    ContentType='text/plain; charset=utf-8' # Specify content type
                )
                logging.info(f"Successfully saved chat content to archive: {archive_key}")
                return True, filename
            except Exception as put_e:
                 logging.error(f"Error writing chat content to archive file {archive_key}: {put_e}", exc_info=True)
                 return False, None

    except Exception as e:
        logging.error(f"General error in save_chat_to_s3 for filename {filename}: {e}", exc_info=True)
        return False, None

def reload_memory(agent_name: str, memory_agents: List[str], initial_system_prompt: str) -> str:
    """Reload memory from saved chat history files and append to system prompt."""
    try:
        logging.debug("Reloading memory...")
        previous_chats = load_existing_chats_from_s3(agent_name, memory_agents)

        if not previous_chats:
            logging.debug("No saved chat history found to load into memory.")
            return initial_system_prompt

        # Combine content from all loaded chats
        all_content_items = []
        for chat in previous_chats:
            file_info = f"(From file: {os.path.basename(chat.get('file', 'unknown'))})"
            logging.debug(f"Processing memory from {file_info}")
            for msg in chat.get('messages', []):
                role = msg.get('role', 'unknown').capitalize()
                content = msg.get('content', '')
                if content:
                    all_content_items.append(f"{role} {file_info}: {content}") # Add file context

        combined_content = "\n\n---\n\n".join(all_content_items)
        # Apply truncation or summarization if needed (simple truncation here)
        max_mem_len = 10000 # Example limit
        summarized_content = combined_content[:max_mem_len] + ("..." if len(combined_content) > max_mem_len else "")

        if summarized_content:
            memory_section = "\n\n## Previous Chat History (Memory)\n" + summarized_content
            logging.debug(f"Appending memory summary ({len(summarized_content)} chars) to system prompt.")
            # Avoid appending duplicates if called multiple times
            if "## Previous Chat History (Memory)" not in initial_system_prompt:
                return initial_system_prompt + memory_section
            else:
                logging.warning("Memory section already present in system prompt, not appending again.")
                return initial_system_prompt
        else:
            logging.debug("No content extracted from previous chats for memory.")
            return initial_system_prompt

    except Exception as e:
        logging.error(f"Error reloading memory: {e}", exc_info=True)
        return initial_system_prompt # Return original on error

def display_help():
    """Prints available CLI commands and startup flags."""
    print("\nAvailable commands:")
    print("!help          - Display this help message")
    print("!exit          - Exit the chat")
    print("!clear         - Clear the current chat session history (in memory only)")
    print("!save          - Save current chat session to 'saved' folder in S3")
    print("!memory        - Toggle memory mode (loads chat history from 'saved' folder)")
    # print("!listen        - Enable summary listening") # Add back if feature exists
    print("!listen-transcript - Toggle automatic transcript listening")
    # print("!listen-insights - Enable insights listening")
    # print("!listen-all    - Enable all listening modes")
    # print("!listen-deep   - Enable summary and insights listening")
    print("\nStartup flags:")
    print("--agent NAME   - (Required) Unique name for the agent.")
    print("--index NAME   - Pinecone index name (default: magicchat).")
    print("--event ID     - Event ID for context/saving (default: 0000).")
    print("--memory [AGENT...] - Start with memory mode enabled, optionally load from specific agents.")
    print("--listen-transcript - Start with transcript listening enabled.")
    print("--all          - Read all transcripts at launch (use with --listen-transcript).")
    print("--web          - Run with web interface alongside CLI.")
    print("--web-only     - Run with web interface only.")
    print("--web-port PORT - Port for web interface (default: 5001).")
    print("--debug        - Enable debug logging.")

def format_chat_history(messages: List[Dict[str, Any]]) -> str:
    """Formats a list of message dicts into a string for saving."""
    chat_content = ""
    for msg in messages:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        if content: # Ensure content exists
             # Simple formatting, adjust as needed
             chat_content += f"**{role}:**\n{content}\n\n"
    return chat_content.strip()


# Main execution block
def main():
    global abort_requested # Allow external signals to stop
    try:
        # Load configuration using AppConfig
        config = AppConfig.from_env_and_args()

        # Setup logging based on config
        setup_logging(config.debug)
        logging.info(f"Starting agent '{config.agent_name}' with config: {config}")

        # Initialize session ID and chat filename if not already set by WebChat
        if not config.session_id:
             timestamp = datetime.now().strftime('%Y%m%d-T%H%M%S')
             config.session_id = timestamp
             event_id = config.event_id or '0000'
             current_chat_file = f"chat_D{timestamp}_aID-{config.agent_name}_eID-{event_id}.txt"
             logging.info(f"CLI Mode: Initialized chat filename: {current_chat_file}")
        else:
             # If session_id already set (likely by WebChat), derive filename
             event_id = config.event_id or '0000'
             current_chat_file = f"chat_D{config.session_id}_aID-{config.agent_name}_eID-{event_id}.txt"
             logging.info(f"Web/CLI Mode: Using existing session ID '{config.session_id}', chat file: {current_chat_file}")


        last_saved_index = 0
        last_archive_index = 0

        # Start web interface if requested (and not CLI only)
        web_thread = None
        if config.interface_mode in ['web', 'web_only']:
            try:
                web_interface = WebChat(config) # Pass config to WebChat
                web_thread = web_interface.run(port=config.web_port, debug=config.debug)
                logging.info(f"Web interface starting on http://127.0.0.1:{config.web_port}")
            except Exception as e:
                 logging.error(f"Failed to start web interface: {e}", exc_info=True)
                 if config.interface_mode == 'web_only':
                      print("Error: Failed to start web interface in web-only mode. Exiting.", file=sys.stderr)
                      sys.exit(1)
                 else:
                      print("Warning: Failed to start web interface, continuing in CLI mode.", file=sys.stderr)
                      config.interface_mode = 'cli' # Fallback to CLI

            if config.interface_mode == 'web_only':
                print("\nRunning in web-only mode. Press Ctrl+C in the console running Flask to exit.")
                # Keep main thread alive while Flask runs in its thread
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nShutting down web-only mode...")
                    # Perform any cleanup if needed
                return # Exit main function for web-only mode

        # --- CLI Mode Setup (or Web+CLI mode) ---
        if config.interface_mode != 'web_only':
            print(f"\nAgent '{config.agent_name}' running.")
            if config.interface_mode == 'web':
                 print(f"Web interface available at http://127.0.0.1:{config.web_port}")
            print("Enter message or type !help")

            # Initialize Anthropic client for CLI
            try:
                 client = Anthropic(api_key=config.anthropic_api_key)
                 logging.info("CLI: Anthropic client initialized.")
            except Exception as e:
                 logging.error(f"CLI: Failed to initialize Anthropic client: {e}", exc_info=True)
                 print("Error: Could not initialize AI client. Please check API key and network. Exiting.", file=sys.stderr)
                 sys.exit(1)

            # Initialize retrieval handler
            try:
                 retriever = RetrievalHandler(
                    index_name=config.index,
                    agent_name=config.agent_name,
                    session_id=config.session_id,
                    event_id=config.event_id
                 )
                 logging.info(f"CLI: RetrievalHandler initialized for index '{config.index}', agent '{config.agent_name}'.")
            except Exception as e:
                 logging.error(f"CLI: Failed to initialize RetrievalHandler: {e}", exc_info=True)
                 # Continue without retrieval? Or exit? Let's warn and continue.
                 retriever = None
                 print("Warning: Could not initialize document retrieval.", file=sys.stderr)

            # --- Load Initial System Prompt & Contexts for CLI ---
            conversation_history = [] # In-memory history for CLI session

            # Load base system prompt
            system_prompt = get_latest_system_prompt(config.agent_name)
            if not system_prompt:
                 logging.error("CLI: Failed to load system prompt. Using fallback.")
                 system_prompt = "You are a helpful assistant."

            # Prepend context/frameworks/docs to conversation history as system messages
            # (This keeps the main system prompt cleaner for core instructions)
            initial_context_messages = []
            frameworks = get_latest_frameworks(config.agent_name)
            if frameworks:
                 initial_context_messages.append({"role": "system", "content": f"## Frameworks\n{frameworks}"})
                 logging.info("CLI: Loaded frameworks.")
            context = get_latest_context(config.agent_name, config.event_id)
            if context:
                 initial_context_messages.append({"role": "system", "content": f"## Context\n{context}"})
                 logging.info("CLI: Loaded context.")
            docs = get_agent_docs(config.agent_name)
            if docs:
                 initial_context_messages.append({"role": "system", "content": f"## Agent Documentation\n{docs}"})
                 logging.info("CLI: Loaded agent documentation.")

            # Prepend these context messages to the history
            conversation_history.extend(initial_context_messages)

            # Load memory if enabled (appends to system_prompt)
            if config.memory:
                 system_prompt = reload_memory(config.agent_name, config.memory, system_prompt)
                 logging.info("CLI: Memory loaded.")


            # Log final system prompt parts for verification
            logging.debug("\n=== CLI: Final System Prompt & Initial Context ===")
            logging.debug(f"Core System Prompt Length: {len(system_prompt)} chars")
            for i, msg in enumerate(initial_context_messages):
                 logging.debug(f"Initial Context Message {i+1}: Role={msg['role']}, Length={len(msg['content'])}, Content='{msg['content'][:100]}...'")
            if "## Previous Chat History" in system_prompt:
                 logging.debug("Memory section appended to system prompt.")


            # --- Initialize Transcript Handling for CLI ---
            transcript_state = TranscriptState()
            last_transcript_check = time.time()
            TRANSCRIPT_CHECK_INTERVAL = 5  # seconds

            # Handle initial transcript load based on flags
            if config.listen_transcript:
                config.listen_transcript_enabled = True # Mark as enabled
                logging.info("CLI: Transcript listening enabled at startup.")
                print("Attempting to load initial transcript...")
                if check_transcript_updates(transcript_state, conversation_history, config.agent_name, config.event_id, read_all=config.read_all):
                     print("Initial transcript loaded.")
                     # The transcript is now in conversation_history
                else:
                     print("No initial transcript content found.")
                last_transcript_check = time.time() # Reset check timer after initial load
            else:
                 config.listen_transcript_enabled = False


            # --- Main CLI Chat Loop ---
            print("\nUser: ", end='', flush=True)
            while True:
                try:
                    # Check for transcript updates periodically if enabled
                    current_time = time.time()
                    if config.listen_transcript_enabled and (current_time - last_transcript_check > TRANSCRIPT_CHECK_INTERVAL):
                        if check_transcript_updates(transcript_state, conversation_history, config.agent_name, config.event_id, read_all=False): # read_all=False for rolling updates
                            logging.debug("New transcript content added to history.")
                            # Maybe notify user? print("\n[Transcript Updated]\nUser: ", end='', flush=True)
                        last_transcript_check = current_time

                    # Check for user input (non-blocking)
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        user_input = sys.stdin.readline().strip()

                        if not user_input: # Handle empty input
                             print("User: ", end='', flush=True)
                             continue

                        # --- Handle Commands ---
                        if user_input.startswith('!'):
                            command = user_input[1:].lower()
                            print(f"Executing command: !{command}") # Acknowledge command
                            if command == 'exit':
                                break
                            elif command == 'help':
                                display_help()
                            elif command == 'clear':
                                # Clear only the current session's message history
                                conversation_history = list(initial_context_messages) # Reset to initial context
                                last_saved_index = 0
                                last_archive_index = len(conversation_history) # Reset archive index too
                                print("Current session history cleared.")
                            elif command == 'save':
                                # Save messages since last manual save
                                messages_to_save = conversation_history[last_saved_index:]
                                if not messages_to_save:
                                     print("No new messages in this session to save.")
                                else:
                                     save_content = format_chat_history(messages_to_save)
                                     success, saved_filename = save_chat_to_s3(
                                         agent_name=config.agent_name,
                                         chat_content=save_content,
                                         event_id=config.event_id,
                                         is_saved=True, # Manual save to 'saved'
                                         filename=current_chat_file
                                     )
                                     if success:
                                          last_saved_index = len(conversation_history) # Update index
                                          print(f"Chat history manually saved to {saved_filename}")
                                     else:
                                          print("Error: Failed to save chat history manually.")
                            elif command == 'memory':
                                if not config.memory: # If memory is currently OFF (None or empty list)
                                     config.memory = [config.agent_name] # Turn ON, default to self
                                     system_prompt = reload_memory(config.agent_name, config.memory, system_prompt)
                                     print("Memory mode ACTIVATED. Previous chats loaded.")
                                else: # If memory is currently ON
                                     config.memory = [] # Turn OFF
                                     # Reload base system prompt without memory section
                                     system_prompt = get_latest_system_prompt(config.agent_name) or "You are a helpful assistant."
                                     print("Memory mode DEACTIVATED.")
                            elif command == 'listen-transcript':
                                 config.listen_transcript_enabled = not config.listen_transcript_enabled # Toggle
                                 if config.listen_transcript_enabled:
                                      print("Transcript listening ENABLED. Checking for initial/new content...")
                                      if check_transcript_updates(transcript_state, conversation_history, config.agent_name, config.event_id, read_all=config.read_all):
                                           print("Transcript content loaded/updated.")
                                      else:
                                           print("No new transcript content found.")
                                      last_transcript_check = time.time() # Reset timer
                                 else:
                                      print("Transcript listening DISABLED.")
                            else:
                                print(f"Unknown command: !{command}")

                            print("\nUser: ", end='', flush=True) # Re-prompt after command
                            continue # Skip LLM call for commands

                        # --- Process User Message ---
                        conversation_history.append({"role": "user", "content": user_input})

                        # --- Retrieval Step ---
                        retrieved_docs = []
                        if retriever:
                             retrieved_docs = retriever.get_relevant_context(user_input)

                        # Prepare context string for LLM prompt
                        context_for_prompt = ""
                        if retrieved_docs:
                             context_items = []
                             for i, doc in enumerate(retrieved_docs):
                                  source_file = doc.metadata.get('file_name', 'Unknown source')
                                  score = doc.metadata.get('score', 0.0)
                                  context_items.append(f"[Context {i+1} from {source_file} (Score: {score:.2f})]:\n{doc.page_content}")
                             context_for_prompt = "\n\n---\nRelevant Context Found:\n" + "\n\n".join(context_items)
                             logging.debug(f"CLI: Adding retrieved context to prompt ({len(context_for_prompt)} chars).")
                        else:
                             logging.debug("CLI: No relevant context retrieved.")

                        # Combine system prompt with retrieved context for this turn
                        current_turn_system_prompt = system_prompt + context_for_prompt

                        # --- Call LLM ---
                        print("Agent: ", end='', flush=True) # Indicate agent is thinking
                        response_text = analyze_with_claude(
                             client,
                             conversation_history, # Pass current history
                             current_turn_system_prompt, # Pass combined prompt
                             config.llm_model_name # Pass model name from config
                        )

                        if response_text:
                            print(response_text) # Print full response
                            conversation_history.append({"role": "assistant", "content": response_text})

                            # Auto-archive the turn
                            messages_to_archive = conversation_history[last_archive_index:]
                            if messages_to_archive:
                                 archive_content = format_chat_history(messages_to_archive)
                                 success, _ = save_chat_to_s3(
                                      agent_name=config.agent_name,
                                      chat_content=archive_content,
                                      event_id=config.event_id,
                                      is_saved=False, # Archive
                                      filename=current_chat_file
                                 )
                                 if success:
                                      last_archive_index = len(conversation_history) # Update archive index
                                      logging.debug(f"CLI: Auto-archived {len(messages_to_archive)} messages.")
                                 else:
                                      logging.error("CLI: Failed to auto-archive chat turn.")
                        else:
                            print("[Error processing request]") # Handle case where analyze_with_claude returns None


                        print("\nUser: ", end='', flush=True) # Prompt for next input

                except (EOFError, KeyboardInterrupt):
                    print("\nExiting chat.")
                    break
                except Exception as loop_e:
                     logging.error(f"Error in main chat loop: {loop_e}", exc_info=True)
                     print(f"\nAn unexpected error occurred: {loop_e}. Please check logs.", file=sys.stderr)
                     # Optionally add a small delay before re-prompting
                     time.sleep(1)
                     print("\nUser: ", end='', flush=True) # Re-prompt after error

    except Exception as e:
        logging.error(f"Fatal error during application startup or main execution: {e}", exc_info=True)
        print(f"\nA critical error occurred: {e}. Check claude_chat.log for details.", file=sys.stderr)
        sys.exit(1)
    finally:
         # Cleanup if needed
         logging.info("Application shutting down.")
         # Ensure web thread is handled if it exists? Flask might handle shutdown on Ctrl+C.

if __name__ == '__main__':
    main()