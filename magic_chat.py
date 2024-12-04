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
    help_text = """
Available Commands:
!listen                - Start listening to summaries.
!listen-transcript     - Start listening to transcripts.
!listen-insights       - Start listening to insights.
!listen-deep           - Start listening to summaries and insights.
!listen-all            - Start listening to summaries, transcripts, and insights.
!silent                - Stop listening to all files.
!reload-memory         - Reload chat histories of the agent and specified memory agents.
!memory [agents]       - Load chat history of same agent. Append multiple agent names option.
!back                  - Abort the current message request.
!help                  - Display this help message.
!exit                  - Exit the chat.
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

def generate_insights(transcript_content, frameworks_content, context_content):
    try:
        prompt = f"""
Using the following frameworks and context, analyze the conversation and provide comprehensive insights in the specified JSON schema.

Frameworks:
{frameworks_content}

Context:
{context_content}

Transcript:
{transcript_content}

Provide the output in JSON format according to the schema:
{InsightsOutput.schema_json(indent=4)}

**Important:** Provide the output as raw JSON without any code block formatting or additional text. Do not include markdown or any explanations.
"""

        import openai
        if not OPENAI_API_KEY:
            logging.error("OPENAI_API_KEY not set.")
            return None
        openai.api_key = OPENAI_API_KEY

        response = openai.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role': 'system', 'content': 'You are an AI assistant specialized in analyzing conversations to extract actionable insights that can drive improvements and inform decision-making. Focus on generating high-quality, specific, and relevant insights based on the provided frameworks and transcript.'},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )

        insights_json = response.choices[0].message.content

        insights_json = insights_json.strip()
        if insights_json.startswith('```json'):
            insights_json = insights_json[len('```json'):].strip()
        if insights_json.endswith('```'):
            insights_json = insights_json[:-3].strip()

        insights = InsightsOutput.parse_raw(insights_json)

        return insights
    except Exception as e:
        logging.error(f"Error generating insights: {e}")
        return None

def main():
    global abort_requested
    config = AppConfig.from_env_and_args()
    
    setup_logging(config.debug)
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
            print("User:\n", end='', flush=True)
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
                chat_history_folder, config.agent_name, config.memory, initial_system_prompt
            )
            if config.debug:
                logging.debug("Memory reloaded.")
            print("Memory reloaded.\n")
            continue
        if user_input.lower() == '!help':
            display_help()
            continue
        if user_input.lower() == '!listen':
            config.listen_summary = True
            print("Listening to summaries activated.\n")
            continue
        if user_input.lower() == '!listen-transcript':
            config.listen_transcript = True
            print("Listening to transcripts activated.\n")
            continue
        if user_input.lower() == '!listen-insights':
            config.listen_insights = True
            print("Listening to insights activated.\n")
            continue
        if user_input.lower() == '!listen-all':
            config.listen_summary = True
            config.listen_transcript = True
            config.listen_insights = True
            config.listen_all = True
            print("Listening to summaries, transcripts, and insights activated.\n")
            continue
        if user_input.lower() == '!listen-deep':
            config.listen_summary = True
            config.listen_insights = True
            config.listen_deep = True
            print("Listening to summaries and insights activated.\n")
            continue
        if user_input.lower().startswith('!memory'):
            parts = user_input.split()
            agents_to_load = parts[1:] if len(parts) > 1 else [config.agent_name]
            if len(parts) == 1:
                agents_to_load = [config.agent_name]
            config.memory = agents_to_load
            system_prompt = reload_memory(chat_history_folder, config.agent_name, config.memory, initial_system_prompt)
            print("Memory loaded.\n")
            continue
        if user_input.lower() == '!back':
            abort_requested = True
            print("[info] No active request to abort.\n")
            continue
        if user_input.lower() == '!insights':
            transcript_file = get_latest_transcript_file()
            if not transcript_file:
                print("<<No transcript file found in S3 to generate insights>>\n")
                continue
            transcript_content = read_file_content(transcript_file, "transcript")
            if not transcript_content:
                print("<<Transcript file is empty or unreadable>>\n")
                continue

            insights = generate_insights(transcript_content, frameworks_content, context_content)
            if insights:
                insights_filename = f"insights_{datetime.now().strftime('%Y%m%d-%H%M%S')}_uID-0112_oID-{org_id}_sID-{config.agent_name}.txt"
                try:
                    s3_key = f"live/insights/{insights_filename}"
                    s3_client.put_object(
                        Bucket=AWS_S3_BUCKET,
                        Key=s3_key,
                        Body=json.dumps(insights.dict(), indent=4),
                        ContentType='application/json'
                    )
                    print(f"Insights have been saved to S3 at key: {s3_key}\n")

                    insights_content = read_file_content(s3_key, "insights")
                    if insights_content:
                        agent_message = f"Insights:\n{insights_content}"
                        conversation_history.append({"role": "agent", "content": agent_message})
                        append_to_chat_history(chat_history_file, "Agent", agent_message, is_insight=True)
                        print("Insights have been appended to the conversation history.\n")
                    else:
                        print("Insights file is empty or unreadable.\n")
                except Exception as e:
                    logging.error(f"Error saving insights to S3: {e}")
                    print("Failed to save insights to S3.\n")
            else:
                print("No insights were generated.\n")
            continue

        content_pieces = []
        
        if config.listen_summary:
            summary_file = get_latest_summary_file()
            if summary_file:
                summary = read_file_content(summary_file, "summary")
                if summary:
                    content_pieces.append(f"\nLatest Summary:\n{summary}")
                    print("<<Latest summary loaded>>\n")
            else:
                print("<<No summary file found>>\n")
        
        if config.listen_transcript:
            transcript_file = get_latest_transcript_file()
            if transcript_file:
                transcript = read_file_content(transcript_file, "transcript")
                if transcript:
                    content_pieces.append("Transcript:\n" + transcript)
                else:
                    print("Transcript file is empty or unreadable.\n")
            else:
                print("<<No transcript file found>>\n")
        
        if config.listen_insights:
            insights_file = get_latest_insights_file()
            if insights_file:
                insights = read_file_content(insights_file, "insights")
                if insights:
                    content_pieces.append("Insights:\n" + insights)
                else:
                    print("Insights file is empty or unreadable.\n")
            else:
                print("<<No insights file found>>\n")
        
        combined_content = ""
        if content_pieces:
            combined_content = "\n\n".join(content_pieces)
            combined_content = summarize_text(combined_content, max_length=None)
        
        if combined_content and config.memory:
            system_prompt = initial_system_prompt + "\nSummary of past conversations:\n" + chat_summary + "\n\n" + combined_content
        elif combined_content:
            system_prompt = initial_system_prompt + "\n" + combined_content
        elif config.memory:
            system_prompt = initial_system_prompt + "\nSummary of past conversations:\n" + chat_summary
        else:
            system_prompt = initial_system_prompt

        if frameworks_content:
            system_prompt += f"\nFrameworks:\n{frameworks_content}"
        if context_content:
            system_prompt += f"\nContext:\n{context_content}"

        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        user_content = f"On {current_timestamp}, user said: {user_input}"
        conversation_history.append({"role": "user", "content": user_content})
        append_to_chat_history(chat_history_file, "User", user_input)

        truncate_conversation(conversation_history, max_tokens=TOKEN_LIMIT - 500)

        messages = conversation_history.copy()

        messages_to_send = [
            {'role': 'assistant' if msg['role'] == 'agent' else msg['role'], 'content': msg['content']}
            for msg in messages
        ]

        invalid_roles = [msg['role'] for msg in messages_to_send if msg['role'] not in ['user', 'assistant']]
        if invalid_roles:
            continue

        print("\nAgent:\n", end='')
        completion = analyze_with_claude(client, messages_to_send, system_prompt)
        if completion:
            agent_response = completion.strip()
            agent_content = f"On {current_timestamp}, agent said: {agent_response}"
            conversation_history.append({"role": "agent", "content": agent_content})
            append_to_chat_history(chat_history_file, "Agent", agent_response)
            print("\n")
        else:
            print("\nNo response received from model.\n")

if __name__ == '__main__':
    main()