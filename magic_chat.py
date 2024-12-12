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

SESSION_START_TAG = '<session>'
SESSION_END_TAG = '</session>'
SESSION_END_MARKER = '\n### Chat Session End ###'

abort_requested = False

TOKEN_LIMIT = 4096
AVERAGE_TOKENS_PER_MESSAGE = 50
MAX_MESSAGES = TOKEN_LIMIT // AVERAGE_TOKENS_PER_MESSAGE

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
    
    logging.getLogger('anthropic').setLevel(logging.WARNING)

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

def get_latest_context(agent_name, event_id=None):
    """Get and combine contexts from S3"""
    try:
        s3_client = boto3.client('s3')
        
        # Get organization-specific context
        org_context = read_file_content(f'organizations/river/_config/context_oID-{agent_name}.md', "organization context")
        
        # Get event-specific context if event ID is provided
        event_context = ""
        if event_id:
            event_context_key = f'organizations/river/agents/{agent_name}/events/{event_id}/_config/context_aID-{agent_name}_eID-{event_id}.md'
            event_context = read_file_content(event_context_key, "event context")
        
        # Combine contexts
        context = org_context
        if event_context:
            context += "\n\n" + event_context
            
        return context
    except Exception as e:
        logging.error(f"Error getting contexts: {e}")
        return None

def read_file_content(file_key, description):
    try:
        response = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=file_key)
        content = response['Body'].read().decode('utf-8')
        if not content.strip():
            logging.debug(f"Empty content in {file_key}")
            return None
        logging.debug(f"Successfully read content from {file_key}, length: {len(content)}")
        return content
    except s3_client.exceptions.NoSuchKey:
        logging.error(f"No {description} file at '{file_key}' in S3.")
        return None
    except Exception as e:
        logging.error(f"Error reading {description} file '{file_key}' from S3: {e}")
        return None

def summarize_text(text, max_length=None):
    if max_length is None or len(text) <= max_length:
        return text
    else:
        return text[:max_length] + "..."

def analyze_with_claude(client, messages, system_prompt):
    global abort_requested
    max_retries = 5
    initial_backoff = 1
    backoff_factor = 2
    full_response = ""
    request_start_time = time.time()
    info_displayed = False
    response_received = threading.Event()

    def show_delay_message():
        if not response_received.is_set():
            print("[info] Response is taking longer than 7 seconds. Please wait...")

    for attempt in range(1, max_retries + 1):
        if abort_requested:
            abort_requested = False
            return None
        try:
            timer = threading.Timer(7, show_delay_message)
            timer.start()

            print("AI: ", end='', flush=True)  
            with client.messages.stream(
                model='claude-3-5-sonnet-20241022',
                max_tokens=4096,
                temperature=0.7,
                system=system_prompt,
                messages=messages
            ) as stream:
                for text in stream.text_stream:
                    response_received.set()
                    timer.cancel()
                    print(text, end='', flush=True)
                    full_response += text
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        user_input = sys.stdin.readline().strip()
                        if user_input.lower() == '!back':
                            abort_requested = True
                            print("\n[info] Aborting the current request...\n")
                            timer.cancel()
                            return None
            print()  
            return full_response
        except AnthropicError as e:
            timer.cancel()
            if e.status_code == 429:
                retry_after = e.response_headers.get('retry-after')
                if retry_after:
                    wait_time = int(retry_after)
                else:
                    wait_time = initial_backoff * (backoff_factor ** (attempt - 1))
                print(f"[info] Rate limit hit. Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"Anthropic API error: {e}")
                return None
        except Exception as e:
            timer.cancel()
            logging.error(f"Unexpected error: {e}")
            return None
    logging.error(f"Failed after {max_retries} attempts.")
    return None

def load_existing_chats_from_s3(agent_name, memory_agents=None):
    """Load chat history from S3 for the specified agent(s)"""
    try:
        chat_histories = []
        agents_to_load = [agent_name] if memory_agents is None else memory_agents

        for agent in agents_to_load:
            # Use default event '0000' since events are not yet implemented
            # Only load from saved directory when memory is enabled
            prefix = f'organizations/river/agents/{agent}/events/0000/chats/saved/chat_'
            
            try:
                response = s3_client.list_objects_v2(Bucket=AWS_S3_BUCKET, Prefix=prefix)
                if 'Contents' in response:
                    chat_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.txt')]
                    
                    for chat_file in chat_files:
                        try:
                            chat_obj = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=chat_file['Key'])
                            chat_content = chat_obj['Body'].read().decode('utf-8')
                            
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

def save_chat_to_s3(agent_name, chat_content, is_saved=False):
    """Save chat content to S3 in the new file structure"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"chat_{agent_name}_{timestamp}.txt"
        
        # Use default event '0000' since events are not yet implemented
        # Save to either 'saved' or 'archive' directory
        folder = 'saved' if is_saved else 'archive'
        s3_key = f"organizations/river/agents/{agent_name}/events/0000/chats/{folder}/{filename}"
        
        try:
            s3_client.put_object(
                Bucket=AWS_S3_BUCKET,
                Key=s3_key,
                Body=chat_content.encode('utf-8')
            )
            return True, filename
        except Exception as e:
            logging.error(f"Error saving chat file {filename}: {e}")
            return False, None
            
    except Exception as e:
        logging.error(f"Error preparing to save chat: {e}")
        return False, None

def reload_memory(agent_name, memory_agents, initial_system_prompt):
    """Reload memory from chat history files"""
    previous_chats = load_existing_chats_from_s3(agent_name, memory_agents)
    
    # Combine all chat content
    all_content = []
    for chat in previous_chats:
        for msg in chat['messages']:
            all_content.append(msg['content'])
    
    combined_content = "\n\n".join(all_content)  # Add extra newline between files
    logging.debug(f"Combined content length: {len(combined_content)}")
    summarized_content = summarize_text(combined_content, max_length=None)
    logging.debug(f"Summarized content length: {len(summarized_content) if summarized_content else 0}")
    
    # Add the summarized content to the system prompt with clear context
    if summarized_content:
        new_system_prompt = (
            initial_system_prompt + 
            "\n\n## Previous Chat History\nThe following is a summary of previous chat interactions:\n\n" + 
            summarized_content
        )
        logging.debug("Added chat history to system prompt")
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
                    obj for obj in response['Contents'] 
                    if obj['Key'].startswith(prefix) and obj['Key'] != prefix
                    and not obj['Key'].replace(prefix, '').strip('/').count('/')  # No additional folders
                    and obj['Key'].endswith('.txt')
                ]
                if transcript_files:
                    latest_file = max(transcript_files, key=lambda x: x['LastModified'])
                    logging.debug(f"Found latest transcript in agent folder: {latest_file['Key']}")
                    return latest_file['Key']
                else:
                    logging.debug(f"No transcript files found in agent folder: {prefix}")
        
        # Fallback to default transcripts folder
        prefix = '_files/transcripts/archive/'
        response = s3_client.list_objects_v2(Bucket=AWS_S3_BUCKET, Prefix=prefix, Delimiter='/')
        
        if 'Contents' in response:
            # Only consider files directly in this folder
            transcript_files = [
                obj for obj in response['Contents'] 
                if obj['Key'].startswith(prefix) and obj['Key'] != prefix
                and not obj['Key'].replace(prefix, '').strip('/').count('/')  # No additional folders
                and obj['Key'].endswith('.txt')
            ]
            if transcript_files:
                latest_file = max(transcript_files, key=lambda x: x['LastModified'])
                logging.debug(f"Found latest transcript in default folder: {latest_file['Key']}")
                return latest_file['Key']
            else:
                logging.debug(f"No transcript files found in default folder: {prefix}")
                
        logging.debug("No transcript files found in any location")
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
                system_prompt += "\n\n## Frameworks\n" + frameworks
                
            # Add context
            context = get_latest_context(config.agent_name)  # Note: event_id not implemented yet
            if context:
                system_prompt += "\n\n## Context\n" + context
            
            # Load memory if enabled
            if config.memory is not None:
                if len(config.memory) == 0:
                    config.memory = [config.agent_name]
                system_prompt = reload_memory(config.agent_name, config.memory, system_prompt)

            conversation_history = []

            org_id = 'River'  # Replace with actual organization ID as needed

            # Load initial content based on command line arguments
            if config.listen_transcript:
                transcript_key = get_latest_transcript_file(config.agent_name)
                if transcript_key:
                    transcript_content = read_file_content(transcript_key, "transcript")
                    if transcript_content:
                        print("Initial transcript loaded.")
                        system_prompt += f"\n\nTranscript update: {transcript_content}"
                    else:
                        print("No transcript content found.")
                else:
                    print("No transcript files found.")

            print("\nUser: ", end='', flush=True)  # Initial prompt
            
            # Main chat loop
            while True:
                try:
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
                                chat_content = format_chat_history(conversation_history)
                                save_chat_to_s3(config.agent_name, chat_content, is_saved=True)
                                print("Chat history saved successfully")
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
                                # Load transcript if needed
                                if command in ['listen', 'listen-all', 'listen-transcript']:
                                    transcript_key = get_latest_transcript_file(config.agent_name)
                                    if transcript_key:
                                        transcript_content = read_file_content(transcript_key, "transcript")
                                        if transcript_content:
                                            print("Transcript loaded and listening mode activated.")
                                            system_prompt += f"\n\nTranscript update: {transcript_content}"
                                        else:
                                            print("No transcript content found.")
                                    else:
                                        print("No transcript files found.")
                                
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
                            
                            # Save chat history to archive folder after each message
                            chat_content = format_chat_history(conversation_history)
                            save_chat_to_s3(config.agent_name, chat_content, is_saved=False)
                            
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
