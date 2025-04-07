"""Utilities for interacting with AWS S3."""

import os
import boto3
import logging
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

# Configure logging for this module
logger = logging.getLogger(__name__)

# Global S3 client instance (lazy loaded)
s3_client = None
# Remove module-level reading of env vars
# aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
# aws_region = os.getenv('AWS_REGION')

def get_s3_client() -> Optional[boto3.client]:
    """Initializes and returns an S3 client, or None on failure.
       Reads env vars inside the function to ensure they are loaded after load_dotenv().
    """
    global s3_client
    if s3_client is None:
        # Read env vars here
        aws_region = os.getenv('AWS_REGION')
        aws_s3_bucket = os.getenv('AWS_S3_BUCKET') # Read bucket here too, mainly for logging the check

        if not aws_region or not aws_s3_bucket:
            logger.error("AWS_REGION or AWS_S3_BUCKET environment variables not found or not set.")
            return None
        try:
            s3_client = boto3.client(
                's3',
                region_name=aws_region
                # Credentials assumed to be handled by AWS SDK environment (IAM role, env vars, config file)
            )
            logger.info(f"S3 client initialized for region {aws_region}.")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}", exc_info=True)
            s3_client = None # Ensure it stays None on failure
    return s3_client

def read_file_content(file_key: str, description: str) -> Optional[str]:
    """Read content from S3 file, handling potential errors."""
    s3 = get_s3_client()
    # Read bucket name here as well, in case it changes or needs checking per call
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        logger.error(f"S3 client or bucket name not available for reading {description}.")
        return None
    try:
        logger.debug(f"Reading {description} from S3: s3://{aws_s3_bucket}/{file_key}")
        response = s3.get_object(Bucket=aws_s3_bucket, Key=file_key)
        content = response['Body'].read().decode('utf-8')
        # No need to log full content, length is sufficient for debug
        logger.debug(f"Successfully read {description} ({len(content)} chars)")
        return content
    except s3.exceptions.NoSuchKey:
        logger.warning(f"{description} file not found at S3 key: {file_key}")
        return None
    except Exception as e:
        logger.error(f"Error reading {description} from S3 key {file_key}: {e}", exc_info=True)
        return None

def find_file_any_extension(base_pattern: str, description: str) -> Optional[Tuple[str, str]]:
    """Find the most recent file matching base pattern with any extension in S3."""
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        logger.error(f"S3 client or bucket name not available for finding {description}.")
        return None

    try:
        prefix = ""
        base_name = base_pattern
        if '/' in base_pattern:
             prefix = base_pattern.rsplit('/', 1)[0] + '/'
             base_name = base_pattern.rsplit('/', 1)[1]

        logger.debug(f"Searching for {description} with prefix '{prefix}' and base name '{base_name}' in bucket '{aws_s3_bucket}'")
        paginator = s3.get_paginator('list_objects_v2')
        matching_files = []

        for page in paginator.paginate(Bucket=aws_s3_bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    relative_key = key[len(prefix):]
                    if '/' in relative_key: continue # Skip subdirs

                    filename_part = os.path.basename(key)
                    name_only, ext = os.path.splitext(filename_part)

                    if name_only == base_name:
                        matching_files.append(obj)

        if not matching_files:
             logger.warning(f"No files found matching base pattern '{base_pattern}' for {description}.")
             return None

        logger.debug(f"Found {len(matching_files)} potential files for {description} matching base '{base_name}'.")
        matching_files.sort(key=lambda obj: obj['LastModified'], reverse=True)
        latest_file_key = matching_files[0]['Key']

        logger.debug(f"Latest file found for {description}: {latest_file_key}")
        content = read_file_content(latest_file_key, description)
        if content is not None:
            logger.debug(f"Successfully loaded content for {description}, length: {len(content)}")
            return latest_file_key, content
        else:
            logger.error(f"Failed to read content from {latest_file_key} for {description}.")
            return None

    except Exception as e:
        logger.error(f"Error finding {description} file for pattern '{base_pattern}': {e}", exc_info=True)
        return None

# --- Functions moved from magic_chat.py ---

def get_latest_system_prompt(agent_name: Optional[str] = None) -> Optional[str]:
    """Get and combine system prompts from S3"""
    logger.debug(f"Getting system prompt (agent: {agent_name})")
    base_result = find_file_any_extension('_config/systemprompt_base', "base system prompt")
    base_prompt = base_result[1] if base_result else None

    if not base_prompt:
         logger.error("Base system prompt '_config/systemprompt_base' not found or failed to load.")
         return "You are a helpful assistant." # Critical fallback

    agent_prompt = ""
    if agent_name:
        agent_pattern = f'organizations/river/agents/{agent_name}/_config/systemprompt_aID-{agent_name}'
        agent_result = find_file_any_extension(agent_pattern, "agent system prompt")
        if agent_result:
            agent_prompt = agent_result[1]
            logger.info(f"Loaded agent-specific system prompt for '{agent_name}'.")
        else:
             logger.warning(f"No agent-specific system prompt found using pattern '{agent_pattern}'.")

    system_prompt = base_prompt
    if agent_prompt: system_prompt += "\n\n" + agent_prompt
    logger.info(f"Final system prompt length: {len(system_prompt)}")
    return system_prompt

def get_latest_frameworks(agent_name: Optional[str] = None) -> Optional[str]:
    """Get and combine frameworks from S3"""
    logger.debug(f"Getting frameworks (agent: {agent_name})")
    base_result = find_file_any_extension('_config/frameworks_base', "base frameworks")
    base_frameworks = base_result[1] if base_result else ""

    agent_frameworks = ""
    if agent_name:
        agent_pattern = f'organizations/river/agents/{agent_name}/_config/frameworks_aID-{agent_name}'
        agent_result = find_file_any_extension(agent_pattern, "agent frameworks")
        if agent_result:
            agent_frameworks = agent_result[1]
            logger.info(f"Loaded agent-specific frameworks for '{agent_name}'.")
        else:
             logger.warning(f"No agent-specific frameworks found using pattern '{agent_pattern}'.")

    frameworks = base_frameworks
    if agent_frameworks: frameworks += ("\n\n" + agent_frameworks) if frameworks else agent_frameworks

    if frameworks: logger.info(f"Loaded frameworks, total length: {len(frameworks)}")
    else: logger.warning("No base or agent-specific frameworks found.")
    return frameworks if frameworks else None

def get_latest_context(agent_name: str, event_id: Optional[str] = None) -> Optional[str]:
    """Get and combine organization and event contexts from S3"""
    logger.debug(f"Getting context (agent: {agent_name}, event: {event_id})")
    org_context = ""
    org_pattern = f'organizations/river/_config/context_oID-{agent_name}'
    logger.warning(f"Attempting org context load using pattern: '{org_pattern}'. Confirm logic.")
    org_result = find_file_any_extension(org_pattern, "organization context")
    if org_result: org_context = org_result[1]; logger.info("Loaded organization context.")
    else: logger.warning(f"No organization context found using pattern '{org_pattern}'.")

    event_context = ""
    if event_id and event_id != '0000':
        event_pattern = f'organizations/river/agents/{agent_name}/events/{event_id}/_config/context_aID-{agent_name}_eID-{event_id}'
        event_result = find_file_any_extension(event_pattern, "event context")
        if event_result: event_context = event_result[1]; logger.info(f"Loaded event context '{event_id}'.")
        else: logger.warning(f"No event context found using pattern '{event_pattern}'.")

    context = org_context
    if event_context: context += ("\n\n" + event_context) if context else event_context
    if context: logger.info(f"Loaded context, total length: {len(context)}")
    else: logger.warning("No organization or event context found.")
    return context if context else None

def get_agent_docs(agent_name: str) -> Optional[str]:
    """Get documentation files for the specified agent."""
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket: logger.error("S3 unavailable for getting agent docs."); return None

    try:
        prefix = f'organizations/river/agents/{agent_name}/docs/'
        logger.debug(f"Searching for agent docs in S3 prefix '{prefix}'")
        paginator = s3.get_paginator('list_objects_v2')
        docs = []
        for page in paginator.paginate(Bucket=aws_s3_bucket, Prefix=prefix):
             if 'Contents' in page:
                  for obj in page['Contents']:
                      key = obj['Key']
                      if key == prefix or key.endswith('/'): continue
                      filename = os.path.basename(key)
                      content = read_file_content(key, f'agent doc ({filename})')
                      if content: docs.append(f"--- START Doc: {filename} ---\n{content}\n--- END Doc: {filename} ---")

        if not docs: logger.warning(f"No documentation files found in '{prefix}'"); return None
        logger.info(f"Loaded {len(docs)} documentation files for agent '{agent_name}'.")
        return "\n\n".join(docs)
    except Exception as e: logger.error(f"Error getting agent docs for '{agent_name}': {e}", exc_info=True); return None

def save_chat_to_s3(agent_name: str, chat_content: str, event_id: Optional[str], is_saved: bool = False, filename: Optional[str] = None) -> tuple[bool, Optional[str]]:
    """Save chat content to S3 bucket (archive or saved folder)."""
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket: logger.error("S3 unavailable for saving chat."); return False, None
    event_id = event_id or '0000'

    try:
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d-T%H%M%S')
            filename = f"chat_D{timestamp}_aID-{agent_name}_eID-{event_id}.txt"
            logger.debug(f"Generated chat filename: {filename}")

        base_path = f"organizations/river/agents/{agent_name}/events/{event_id}/chats"
        archive_key = f"{base_path}/archive/{filename}"
        saved_key = f"{base_path}/saved/{filename}"

        if is_saved:
            logger.info(f"Saving chat: Copying from archive '{archive_key}' to saved '{saved_key}'")
            try:
                 s3.head_object(Bucket=aws_s3_bucket, Key=archive_key) # Check source
                 copy_source = {'Bucket': aws_s3_bucket, 'Key': archive_key}
                 s3.copy_object(CopySource=copy_source, Bucket=aws_s3_bucket, Key=saved_key)
                 logger.info(f"Successfully copied chat to saved: {saved_key}")
                 return True, filename
            except s3.exceptions.ClientError as e:
                 if e.response['Error']['Code'] == '404': logger.error(f"Cannot save: Archive {archive_key} missing.")
                 else: logger.error(f"S3 ClientError checking/copying {archive_key}: {e}", exc_info=True)
                 return False, None
            except Exception as e: logger.error(f"Error copying chat {archive_key}: {e}", exc_info=True); return False, None
        else:
            logger.info(f"Saving chat to archive: {archive_key}")
            full_content = chat_content.strip(); existing_content = ""
            try:
                existing_obj = s3.get_object(Bucket=aws_s3_bucket, Key=archive_key)
                existing_content = existing_obj['Body'].read().decode('utf-8')
                full_content = existing_content.strip() + '\n\n' + chat_content.strip()
                logger.debug(f"Appending {len(chat_content)} chars to archive.")
            except s3.exceptions.NoSuchKey: logger.debug(f"Archive {archive_key} not found. Creating new.")
            except Exception as get_e: logger.error(f"Error reading archive {archive_key}: {get_e}", exc_info=True); return False, None

            try:
                s3.put_object(Bucket=aws_s3_bucket, Key=archive_key, Body=full_content.encode('utf-8'), ContentType='text/plain; charset=utf-8')
                logger.info(f"Successfully saved to archive: {archive_key}")
                return True, filename
            except Exception as put_e: logger.error(f"Error writing to archive {archive_key}: {put_e}", exc_info=True); return False, None

    except Exception as e: logger.error(f"Error in save_chat_to_s3 ({filename}): {e}", exc_info=True); return False, None

def parse_text_chat(chat_content_str: str) -> List[Dict[str, str]]:
    """Parses chat content from text format (**User:**/**Agent:**)."""
    messages = []; current_role = None; current_content = []
    for line in chat_content_str.splitlines():
        line_strip = line.strip(); role_found = None; content_start = 0
        if line_strip.startswith('**User:**'): role_found = 'user'; content_start = len('**User:**')
        elif line_strip.startswith('**Agent:**'): role_found = 'assistant'; content_start = len('**Agent:**')
        if role_found:
            if current_role: messages.append({'role': current_role, 'content': '\n'.join(current_content).strip()})
            current_role = role_found; current_content = [line_strip[content_start:].strip()]
        elif current_role: current_content.append(line)
    if current_role: messages.append({'role': current_role, 'content': '\n'.join(current_content).strip()})
    return [msg for msg in messages if msg.get('content')]

def load_existing_chats_from_s3(agent_name: str, memory_agents: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Load chat history from S3 'saved' directory."""
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket: logger.error("S3 unavailable for loading chats."); return []
    chat_histories = []; agents_to_load = memory_agents or [agent_name]
    logger.info(f"Loading saved chat history for agents: {agents_to_load}")
    for agent in agents_to_load:
        prefix = f'organizations/river/agents/{agent}/events/0000/chats/saved/'
        logger.debug(f"Checking for saved chats in: {prefix}")
        try:
            paginator = s3.get_paginator('list_objects_v2'); chat_files = []
            for page in paginator.paginate(Bucket=aws_s3_bucket, Prefix=prefix):
                if 'Contents' in page: chat_files.extend(obj for obj in page['Contents'] if not obj['Key'].endswith('/'))
            if not chat_files: logger.debug(f"No saved chats for {agent}."); continue
            chat_files.sort(key=lambda obj: obj['LastModified'], reverse=True)
            logger.info(f"Found {len(chat_files)} saved chats for {agent}.")
            for chat_obj in chat_files:
                file_key = chat_obj['Key']; logger.debug(f"Reading saved chat: {file_key}")
                try:
                    content_str = read_file_content(file_key, f"saved chat {file_key}")
                    if not content_str: logger.warning(f"Empty chat file: {file_key}"); continue
                    messages = parse_text_chat(content_str) # Assume text for now
                    # Add JSON parsing logic if needed based on file extension
                    if messages: chat_histories.append({'agent': agent, 'file': file_key, 'messages': messages})
                    else: logger.warning(f"No messages extracted from {file_key}")
                except Exception as read_err: logger.error(f"Error reading/parsing chat {file_key}: {read_err}", exc_info=True)
        except Exception as list_err: logger.error(f"Error listing chats for {agent}: {list_err}", exc_info=True)
    logger.info(f"Loaded {len(chat_histories)} chat history files total.")
    return chat_histories

def format_chat_history(messages: List[Dict[str, Any]]) -> str:
    """Formats messages list into a string for saving."""
    chat_content = ""
    for msg in messages:
        role = msg.get("role", "unknown").capitalize(); content = msg.get("content", "")
        if content: chat_content += f"**{role}:**\n{content}\n\n"
    return chat_content.strip()