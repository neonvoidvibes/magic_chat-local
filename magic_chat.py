import os
import sys
import logging
import time
import glob
import argparse
import select
from anthropic import Anthropic, AnthropicError
from datetime import datetime
import boto3
import io
import threading
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

def get_latest_summary_file():
    try:
        response = s3_client.list_objects_v2(Bucket=AWS_S3_BUCKET, Prefix='summary_')
        if 'Contents' not in response:
            return None
        summary_files = response['Contents']
        summary_files = [obj for obj in summary_files if obj['Key'].endswith('.txt')]
        if not summary_files:
            return None
        latest_file = max(summary_files, key=lambda x: x['LastModified'])
        latest_file_key = latest_file['Key']
        return latest_file_key
    except Exception as e:
        logging.error(f"Error finding summary files in S3: {e}")
        return None

def get_latest_transcript_file():
    try:
        response = s3_client.list_objects_v2(Bucket=AWS_S3_BUCKET, Prefix='transcript_')
        if 'Contents' not in response:
            return None
        transcript_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.txt')]
        if not transcript_files:
            return None
        latest_file = max(transcript_files, key=lambda x: x['LastModified'])
        return latest_file['Key']
    except Exception as e:
        logging.error(f"Error finding transcript files in S3: {e}")
        return None

def get_latest_insights_file():
    try:
        response = s3_client.list_objects_v2(Bucket=AWS_S3_BUCKET, Prefix='insights_')
        if 'Contents' not in response:
            return None
        insights_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.txt')]
        if not insights_files:
            return None
        latest_file = max(insights_files, key=lambda x: x['LastModified'])
        latest_file_key = latest_file['Key']
        return latest_file_key
    except Exception as e:
        logging.error(f"Error finding insights files in S3: {e}")
        return None

def read_file_content(file_key, description):
    try:
        response = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=file_key)
        content = response['Body'].read().decode('utf-8')
        if not content.strip():
            return None
        return content
    except s3_client.exceptions.NoSuchKey:
        logging.error(f"No {description} file at '{file_key}' in S3.")
        return None
    except Exception as e:
        logging.error(f"Error reading {description} file '{file_key}' from S3: {e}")
        return None

def read_file_content_local(file_path, description):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        if not content.strip():
            return None
    except FileNotFoundError:
        logging.error(f"No {description} file at '{file_path}'.")
        return None
    except Exception as e:
        logging.error(f"Error reading {description} file '{file_path}': {e}")
        return None
    return content

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

            print("\nAI: ", end='', flush=True)  
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
            print("\n")  
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

def append_to_chat_history(file_path, role, content, is_insight=False):
    try:
        with open(file_path, 'r+', encoding='utf-8') as f:
            lines = f.readlines()
            f.seek(0)
            new_lines = []
            insights_marker = "[Insights]"
            if is_insight:
                for line in lines:
                    if insights_marker not in line:
                        new_lines.append(line)
            else:
                new_lines = lines
            f.truncate(0)
            f.writelines(new_lines)
            if role.lower() == 'user':
                f.write(f"**User:**\n{content}\n\n")
            elif role.lower() == 'agent':
                f.write(f"**Agent:**\n{content}\n\n")
            else:
                f.write(f"{role.capitalize()}: {content}\n\n")
            f.write(f"{SESSION_END_TAG}\n")
            f.write(f"{SESSION_END_MARKER}\n")
    except Exception as e:
        logging.error(f"Error writing to chat history file '{file_path}': {e}")

def save_to_s3(content, agent_name, folder_path, filename=None):
    """Save content to S3 bucket in the specified folder path"""
    try:
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"chat_{agent_name}_{timestamp}.txt"
        
        s3_key = f"{folder_path}/{filename}"
        
        # Upload to S3
        s3_client.put_object(
            Bucket=AWS_S3_BUCKET,
            Key=s3_key,
            Body=content.encode('utf-8')
        )
        return s3_key
    except Exception as e:
        logging.error(f"Error saving to S3 at '{s3_key}': {e}")
        return None

def create_new_chat_file(folder, agent_name):
    timestamp = datetime.now()
    formatted_timestamp = timestamp.strftime('%Y%m%d_%H%M%S')
    display_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    filename = f"chat_{agent_name}_{formatted_timestamp}.txt"
    file_path = os.path.join(folder, filename)

    header = (
        f"### Chat Session Start ###\n\n"
        f"AI: {agent_name}\n"
        f"Date and Time (yyyy-mm-dd hh:mm:ss): {display_timestamp}\n\n"
        "System: You have access to the entire chat history in this file, including conversations from other agents you are listening to. Use this information to provide contextually relevant responses and reference past interactions when appropriate.\n\n"
    )

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(f"{SESSION_START_TAG}\n")
            f.write(f"{SESSION_END_TAG}\n")
            f.write(f"{SESSION_END_MARKER}\n")
    except Exception as e:
        logging.error(f"Error creating chat file '{file_path}': {e}")
        sys.exit(1)
    return file_path

def load_existing_chats(folder, agent_name, memory_agents):
    chats = []
    pattern = os.path.join(folder, f"chat_*_*.txt")
    for file in sorted(glob.glob(pattern), key=os.path.getctime):
        if agent_name in file or (memory_agents and any(agent in file for agent in memory_agents)):
            content = read_file_content_local(file, "own chat history" if agent_name in file else "listened chat history")
            if content:
                timestamp = extract_timestamp(content)
                messages = extract_messages(content, timestamp, agent_name if agent_name in file else get_agent_from_filename(file))
                if messages:
                    chats.append({"file": file, "messages": messages})
    return chats

def get_agent_from_filename(file_path):
    basename = os.path.basename(file_path)
    parts = basename.split('_')
    if len(parts) >= 3:
        return parts[1]
    return "unknown"

def extract_timestamp(chat_content):
    lines = chat_content.split('\n')
    for line in lines:
        if line.startswith("Date and Time"):
            return line.split(":", 1)[1].strip()
    return None

def extract_messages(chat_content, timestamp, agent_name):
    messages = []
    within_session = False
    for line in chat_content.split('\n'):
        line = line.strip()
        if line == SESSION_START_TAG:
            within_session = True
            continue
        if line == SESSION_END_TAG:
            within_session = False
            continue
        if within_session:
            if line.startswith("**User:**"):
                content = line[len("**User:**"):].strip()
                content = f"On {timestamp}, user said: {content}"
                messages.append({"role": "user", "content": content})
            elif line.startswith("**Agent:**"):
                content = line[len("**Agent:**"):].strip()
                content = f"On {timestamp}, agent ({agent_name}) said: {content}"
                messages.append({"role": "assistant", "content": content})
    return messages

def generate_summary_of_chats(chats):
    summaries = []
    for chat in chats:
        for message in chat['messages']:
            summaries.append(f"{message['content']}")
    return "\n".join(summaries)

def reload_memory(chat_history_folder, agent_name, memory_agents, initial_system_prompt):
    previous_chats = load_existing_chats(chat_history_folder, agent_name, memory_agents)
    chat_summary = generate_summary_of_chats(previous_chats)
    summarized_chat = summarize_text(chat_summary, max_length=None)
    
    insights = []
    for chat in previous_chats:
        for msg in chat['messages']:
            if msg['role'] == 'assistant' and msg['content'].startswith("[Insights]"):
                insights.append(msg['content'].replace("[Insights] ", ""))
    
    insights_summary = "\n".join(insights) if insights else ""
    
    if insights_summary:
        new_system_prompt = (
            initial_system_prompt +
            "\nSummary of past conversations:\n" + summarized_chat +
            "\n\nLatest Insights:\n" + insights_summary
        )
    else:
        new_system_prompt = initial_system_prompt + "\nSummary of past conversations:\n" + summarized_chat
    
    return new_system_prompt

def display_help():
    print("\nAvailable commands:")
    print("!help          - Display this help message")
    print("!exit          - Exit the chat")
    print("!clear         - Clear the chat history")
    print("!save          - Save current chat history to S3 saved folder")
    print("!listen        - Enable summary listening")
    print("!listen-all    - Enable all listening modes")
    print("!listen-deep   - Enable summary and insights listening")
    print("!listen-insights - Enable insights listening")
    print("!listen-transcript - Enable transcript listening")

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

            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            standard_prompt_file = os.path.join(script_dir, 'system_prompt_standard.txt')
            standard_system_prompt = read_file_content_local(standard_prompt_file, "standard system prompt")
            if not standard_system_prompt:
                print("Error: Standard system prompt is empty or unreadable. Check the log for details.")
                sys.exit(1)
            
            unique_prompt_file = os.path.join(script_dir, f'system_prompt_{config.agent_name}.txt')
            unique_system_prompt = read_file_content_local(unique_prompt_file, "unique system prompt")
            if not unique_system_prompt:
                print("Error: Unique system prompt is empty or unreadable. Check the log for details.")
                sys.exit(1)
            
            initial_system_prompt = standard_system_prompt + "\n" + unique_system_prompt

            chat_history_folder = os.path.join(script_dir, 'chat_history')
            os.makedirs(chat_history_folder, exist_ok=True)

            chat_summary = ""
            if config.memory is not None:
                if len(config.memory) == 0:
                    config.memory = [config.agent_name]
                system_prompt = reload_memory(chat_history_folder, config.agent_name, config.memory, initial_system_prompt)
            else:
                system_prompt = initial_system_prompt

            chat_history_file = create_new_chat_file(chat_history_folder, config.agent_name)

            conversation_history = []

            org_id = 'River'  # Replace with actual organization ID as needed
            frameworks_content = ""
            context_content = ""

            try:
                frameworks_key = 'frameworks/frameworks.txt'
                response = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=frameworks_key)
                frameworks_content = response['Body'].read().decode('utf-8')
            except Exception as e:
                logging.error(f"Error loading frameworks from S3: {e}")

            try:
                context_key = f'context/context_{org_id}.txt'
                response = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=context_key)
                context_content = response['Body'].read().decode('utf-8')
            except Exception as e:
                logging.error(f"Error loading context for {org_id}: {e}")

            while True:
                try:
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        print("\nUser: ", end='', flush=True)  
                        user_input = input().strip()
                        
                        if not user_input:
                            continue
                        
                        if user_input.startswith('!'):
                            command = user_input[1:].lower()
                            if command == 'exit':
                                break
                            elif command == 'help':
                                display_help()
                                continue
                            elif command == 'clear':
                                conversation_history = []
                                continue
                            elif command == 'save':
                                # Save chat history to saved folder
                                chat_content = ""
                                for msg in conversation_history:
                                    if msg["role"] == "user":
                                        chat_content += f"**User:**\n{msg['content']}\n\n"
                                    else:
                                        chat_content += f"**Agent:**\n{msg['content']}\n\n"
                                
                                save_to_s3(chat_content, config.agent_name, f"agents/{config.agent_name}/chat_history/saved")
                                print("Chat history saved successfully")
                                continue
                            elif command in ['listen', 'listen-all', 'listen-deep', 'listen-insights', 'listen-transcript']:
                                # Handle existing listen commands
                                pass
                        
                        # Process user message
                        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        user_content = f"On {current_timestamp}, user said: {user_input}"
                        conversation_history.append({"role": "user", "content": user_content})
                        
                        try:
                            response = analyze_with_claude(client, conversation_history, system_prompt)
                            conversation_history.append({"role": "assistant", "content": response})
                            
                            # Save chat history to archive folder after each message
                            chat_content = ""
                            for msg in conversation_history:
                                if msg["role"] == "user":
                                    chat_content += f"**User:**\n{msg['content']}\n\n"
                                else:
                                    chat_content += f"**Agent:**\n{msg['content']}\n\n"
                            
                            save_to_s3(chat_content, config.agent_name, f"agents/{config.agent_name}/chat_history/archive")
                            
                        except Exception as e:
                            logging.error(f"Error in chat interaction: {e}")
                            print(f"An error occurred: {e}")
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting the chat.")
                    break

    except Exception as e:
        logging.error(f"Error in main loop: {e}")

if __name__ == '__main__':
    main()