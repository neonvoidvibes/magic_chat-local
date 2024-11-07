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

SESSION_START_TAG = '<session>'
SESSION_END_TAG = '</session>'
SESSION_END_MARKER = '\n### Chat Session End ###'

abort_requested = False

TOKEN_LIMIT = 4096
AVERAGE_TOKENS_PER_MESSAGE = 50
MAX_MESSAGES = TOKEN_LIMIT // AVERAGE_TOKENS_PER_MESSAGE

# AWS S3 Configuration
region_name = 'eu-north-1'
bucket_name = 'aiademomagicaudio'
s3_client = boto3.client('s3', region_name=region_name)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a Claude agent instance.")
    parser.add_argument('--agent', required=True, help='Unique name for the agent.')
    parser.add_argument('--memory', nargs='*', help='Names of agents to load chat history from.', default=None)
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('--listen', action='store_true', help='Enable summary listening at startup.')
    return parser.parse_args()

def get_anthropic_api_key():
    try:
        with open('anthropic_api_key.txt', 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print("Error: File 'anthropic_api_key.txt' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading API key: {e}")
        sys.exit(1)

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
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='summary_')
        if 'Contents' not in response:
            return None
        summary_files = response['Contents']
        # Filter files that end with '.txt'
        summary_files = [obj for obj in summary_files if obj['Key'].endswith('.txt')]
        if not summary_files:
            return None
        latest_file = max(summary_files, key=lambda x: x['LastModified'])
        latest_file_key = latest_file['Key']
        return latest_file_key
    except Exception as e:
        logging.error(f"Error finding summary files in S3: {e}")
        return None

def read_file_content(file_key, description):
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
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

def summarize_text(text, max_length=500):
    if max_length and len(text) > max_length:
        return text[:max_length] + "..."
    else:
        return text

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
            # Start timer to show delay message after 7 seconds
            timer = threading.Timer(7, show_delay_message)
            timer.start()

            with client.messages.stream(
                model='claude-3-5-sonnet-20241022',
                max_tokens=4096,
                temperature=0.7,
                system=system_prompt,
                messages=messages
            ) as stream:
                for text in stream.text_stream:
                    response_received.set()  # Signal that response has started
                    timer.cancel()  # Cancel the delay message
                    print(text, end='', flush=True)
                    full_response += text
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        user_input = sys.stdin.readline().strip()
                        if user_input.lower() == '!back':
                            abort_requested = True
                            print("\n[info] Aborting the current request...\n")
                            timer.cancel()
                            return None
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

def append_to_chat_history(file_path, role, content):
    try:
        with open(file_path, 'r+', encoding='utf-8') as f:
            lines = f.readlines()
            f.seek(0)
            for line in lines:
                if line.strip() == SESSION_END_TAG:
                    break
                f.write(line)
            f.write(f"{role.capitalize()}: {content}\n\n")
            f.write(f"{SESSION_END_TAG}\n")
            f.write(f"{SESSION_END_MARKER}\n")
    except Exception as e:
        logging.error(f"Error writing to chat history file '{file_path}': {e}")

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

def read_file_content_local(file_path, description):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        if not content.strip():
            return None
        return content
    except FileNotFoundError:
        logging.error(f"No {description} file at '{file_path}'.")
        return None
    except Exception as e:
        logging.error(f"Error reading {description} file '{file_path}': {e}")
        return None

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
            if line.startswith("User: "):
                content = line[len("User: "):].strip()
                content = f"On {timestamp}, user said: {content}"
                messages.append({"role": "user", "content": content})
            elif line.startswith("Assistant: "):
                content = line[len("Assistant: "):].strip()
                content = f"On {timestamp}, assistant ({agent_name}) said: {content}"
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
    # Ensure the chat summary does not exceed a certain length
    summarized_chat = summarize_text(chat_summary, max_length=2000)
    new_system_prompt = initial_system_prompt + "\nSummary of past conversations:\n" + summarized_chat
    return new_system_prompt

def display_help():
    help_text = """
Available Commands:
!listen           - Start listening to summaries.
!silent           - Stop listening to summaries.
!reload-memory    - Reload chat histories of the agent and specified memory agents.
!memory [agents]  - Load chat history of same agent. Append multiple agent names option.
!back             - Abort the current message request.
!help             - Display this help message.
!exit             - Exit the chat.
"""
    print(help_text)

def truncate_conversation(conversation, max_tokens=TOKEN_LIMIT - 500):
    while True:
        estimated_tokens = len(conversation) * AVERAGE_TOKENS_PER_MESSAGE
        if estimated_tokens <= max_tokens:
            break
        if conversation:
            removed = conversation.pop(0)
            logging.debug(f"Truncated message: {removed}")
        else:
            break
    if len(conversation) * AVERAGE_TOKENS_PER_MESSAGE > max_tokens:
        logging.warning("Conversation history exceeds token limit even after truncation.")

def main():
    global abort_requested
    args = parse_arguments()
    agent_name = args.agent
    listening = args.listen
    memory_loaded = False

    setup_logging(args.debug)
    print(f"Chat agent '{agent_name}' is running. Enter your message or type '!help' for commands.\n")

    ANTHROPIC_API_KEY = get_anthropic_api_key()
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    standard_prompt_file = os.path.join(script_dir, 'system_prompt_standard.txt')
    standard_system_prompt = read_file_content_local(standard_prompt_file, "standard system prompt")
    if not standard_system_prompt:
        print("Error: Standard system prompt is empty or unreadable. Check the log for details.")
        sys.exit(1)
    
    unique_prompt_file = os.path.join(script_dir, f'system_prompt_{agent_name}.txt')
    unique_system_prompt = read_file_content_local(unique_prompt_file, "unique system prompt")
    if not unique_system_prompt:
        print("Error: Unique system prompt is empty or unreadable. Check the log for details.")
        sys.exit(1)
    
    initial_system_prompt = standard_system_prompt + "\n" + unique_system_prompt

    chat_history_folder = os.path.join(script_dir, 'chat_history')
    os.makedirs(chat_history_folder, exist_ok=True)

    chat_summary = ""
    memory_agents = []

    if args.memory is not None:
        if len(args.memory) == 0:
            memory_agents = [agent_name]
        else:
            memory_agents = args.memory
        system_prompt = reload_memory(chat_history_folder, agent_name, memory_agents, initial_system_prompt)
        memory_loaded = True
    else:
        system_prompt = initial_system_prompt

    chat_history_file = create_new_chat_file(chat_history_folder, agent_name)

    conversation_history = []

    while True:
        try:
            print("User: ", end='', flush=True)
            ready, _, _ = select.select([sys.stdin], [], [], None)
            if ready:
                user_input = sys.stdin.readline().strip()
            else:
                user_input = ''
        except (EOFError, KeyboardInterrupt):
            print("\nExiting the chat.")
            break
        if not user_input:
            continue
        if user_input.lower() == '!exit':
            print("Exiting the chat.")
            break
        if user_input.lower() == '!reload-memory':
            system_prompt = reload_memory(
                chat_history_folder, agent_name, memory_agents, initial_system_prompt
            )
            if args.debug:
                logging.debug("Memory reloaded.")
            print("Memory reloaded.\n")
            continue
        if user_input.lower() == '!help':
            display_help()
            continue
        if user_input.lower() == '!listen':
            listening = True
            print("Listening to summaries activated.\n")
            continue
        if user_input.lower() == '!silent':
            listening = False
            print("Listening to summaries paused.\n")
            continue
        if user_input.lower().startswith('!memory'):
            parts = user_input.split()
            agents_to_load = parts[1:] if len(parts) > 1 else [agent_name]
            if len(parts) == 1:
                agents_to_load = [agent_name]
            memory_agents = agents_to_load
            system_prompt = reload_memory(chat_history_folder, agent_name, memory_agents, initial_system_prompt)
            memory_loaded = True
            print("Memory loaded.\n")
            continue
        if user_input.lower() == '!back':
            abort_requested = True
            print("[info] No active request to abort.\n")
            continue

        summary = ""
        if listening:
            summary_file = get_latest_summary_file()
            if summary_file:
                summary = read_file_content(summary_file, "summary")
                if not summary:
                    print("Summary file is empty or unreadable.\n")
                if args.debug and summary:
                    logging.debug(f"Summary loaded: {summarize_text(summary)}")
            else:
                print("No summary file found.\n")

        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        user_content = f"On {current_timestamp}, user said: {user_input}"
        conversation_history.append({"role": "user", "content": user_content})
        append_to_chat_history(chat_history_file, "User", user_input)

        truncate_conversation(conversation_history, max_tokens=TOKEN_LIMIT - 500)

        messages = conversation_history.copy()
        
        if summary and memory_loaded:
            system_prompt = initial_system_prompt + "\nSummary of past conversations:\n" + chat_summary + "\n\nFull summary:\n" + summary
        elif summary:
            system_prompt = initial_system_prompt + "\nFull summary:\n" + summary
        elif memory_loaded:
            pass
        else:
            system_prompt = initial_system_prompt

        messages_to_send = [{'role': msg['role'], 'content': msg['content']} for msg in messages]

        invalid_roles = [msg['role'] for msg in messages_to_send if msg['role'] not in ['user', 'assistant']]
        if invalid_roles:
            continue

        print("\nAssistant:\n", end='')
        completion = analyze_with_claude(client, messages_to_send, system_prompt)
        if completion:
            assistant_response = completion.strip()
            assistant_content = f"On {current_timestamp}, assistant said: {assistant_response}"
            conversation_history.append({"role": "assistant", "content": assistant_content})
            append_to_chat_history(chat_history_file, "Assistant", assistant_response)
            print("\n")
        else:
            print("\nNo response received from model.\n")

if __name__ == '__main__':
    main()
