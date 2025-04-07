import os
import sys
import logging
import time
import argparse
import select
import threading
from anthropic import Anthropic, AnthropicError
from datetime import datetime
import boto3 # Keep boto3 for potential direct use if needed elsewhere
import json
from models import InsightsOutput
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any

# Import S3 and chat parsing utilities from the new location
from utils.s3_utils import (
    get_latest_system_prompt,
    get_latest_frameworks,
    get_latest_context,
    get_agent_docs,
    load_existing_chats_from_s3,
    save_chat_to_s3,
    format_chat_history,
    # get_s3_client # Optional, if direct client needed
)
# Import other necessary utils
from utils.retrieval_handler import RetrievalHandler
from utils.transcript_utils import TranscriptState, get_latest_transcript_file, read_new_transcript_content, read_all_transcripts_in_folder
# Configuration and Web Interface (handle potential circular import)
from config import AppConfig
# Avoid direct import of WebChat here to prevent circular dependency
# from web.web_chat import WebChat

SESSION_START_TAG = '<session>'
SESSION_END_TAG = '</session>'
SESSION_END_MARKER = '\n### Chat Session End ###'

abort_requested = False

# Configure logger for this main script
logger = logging.getLogger(__name__)

def check_transcript_updates(transcript_state: TranscriptState, conversation_history: List[Dict[str, Any]], agent_name: str, event_id: str, read_all: bool = False) -> bool:
    """Checks for transcript updates and appends new content to history."""
    logger.debug("Checking for transcript updates...")
    new_content = read_new_transcript_content(transcript_state, agent_name, event_id, read_all=read_all)
    if new_content:
        logger.debug(f"Adding new transcript content: {new_content[:100]}...")
        conversation_history.append({"role": "user", "content": f"[LIVE TRANSCRIPT UPDATE]\n{new_content}"})
        return True
    return False

def setup_logging(debug: bool):
    """Sets up logging configuration."""
    log_filename = 'claude_chat.log'
    root_logger = logging.getLogger() # Get root logger
    log_level = logging.DEBUG if debug else logging.INFO
    root_logger.setLevel(log_level)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # File handler
    try:
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logger: {e}", file=sys.stderr)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_log_level = logging.DEBUG if debug else logging.INFO
    console_handler.setLevel(console_log_level)
    # Use different formatters if desired
    console_formatter = logging.Formatter('[%(levelname)-8s] %(message)s') # Example with level name aligned
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Set levels for noisy libraries
    logging.getLogger('anthropic').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('s3transfer').setLevel(logging.WARNING)
    logging.getLogger('pinecone').setLevel(logging.INFO)
    # Also set level for our own utils if needed
    logging.getLogger('utils').setLevel(log_level) # Match app debug level for utils

    logger.info(f"Logging setup complete. Level: {logging.getLevelName(log_level)}")


# S3 interaction functions are now imported from utils.s3_utils
# Definitions removed from here:
# - find_file_any_extension
# - get_latest_system_prompt
# - get_latest_frameworks
# - get_latest_context
# - get_agent_docs
# - load_existing_chats_from_s3
# - parse_text_chat (moved to s3_utils as helper for load_existing_chats)
# - read_file_content
# - save_chat_to_s3
# - format_chat_history


# analyze_with_claude needs the model_name parameter
def analyze_with_claude(client: Anthropic, messages: List[Dict[str, Any]], system_prompt: str, model_name: str) -> Optional[str]:
    """Process messages with Claude API, handling transcript updates appropriately"""
    logger.debug(f"\n=== Claude API Request ===")
    logger.debug(f"Using model: {model_name}")
    logger.debug(f"System prompt length: {len(system_prompt)} chars")

    formatted_messages = []
    for msg in messages:
         role = msg.get("role")
         content = msg.get("content", "")
         if not content or role == "system": continue
         api_role = "assistant" if role == "assistant" else "user"
         formatted_messages.append({"role": api_role, "content": content})

    transcript_instruction = "\nIMPORTANT: When you receive transcript updates (marked with [LIVE TRANSCRIPT UPDATE]), do not summarize them. Simply acknowledge that you've received the update and continue the conversation based on the new information."
    final_system_prompt = system_prompt + transcript_instruction

    logger.debug(f"Number of messages for API: {len(formatted_messages)}")
    if logger.isEnabledFor(logging.DEBUG): # Check level before complex logging
         for i, msg in enumerate(formatted_messages[-5:]):
              logger.debug(f"  Message {len(formatted_messages)-5+i}: Role={msg['role']}, Length={len(msg['content'])}, Content='{msg['content'][:100]}...'")

    try:
        response = client.messages.create(
            model=model_name,
            system=final_system_prompt,
            messages=formatted_messages,
            max_tokens=4096
        )
        response_text = response.content[0].text
        logger.debug("\n=== Claude API Response ===")
        logger.debug(f"Response length: {len(response_text)} chars")
        logger.debug(f"Response text (first 100): {response_text[:100]}...")
        return response_text
    except AnthropicError as e:
         logger.error(f"Anthropic API Error calling Claude model {model_name}: {e}")
         return f"Error communicating with AI: {e}"
    except Exception as e:
        logger.error(f"Unexpected error calling Claude model {model_name}: {e}", exc_info=True)
        return f"Unexpected error: {e}"


def reload_memory(agent_name: str, memory_agents: List[str], initial_system_prompt: str) -> str:
    """Reload memory from saved chat history files and append to system prompt."""
    # This function now primarily calls the implementation in s3_utils
    try:
        logger.debug("Reloading memory...")
        previous_chats = load_existing_chats_from_s3(agent_name, memory_agents) # Uses imported function

        if not previous_chats:
            logger.debug("No saved chat history found to load into memory.")
            return initial_system_prompt

        all_content_items = []
        for chat in previous_chats:
            file_info = f"(From file: {os.path.basename(chat.get('file', 'unknown'))})"
            logger.debug(f"Processing memory from {file_info}")
            for msg in chat.get('messages', []):
                role = msg.get('role', 'unknown').capitalize()
                content = msg.get('content', '')
                if content: all_content_items.append(f"{role} {file_info}: {content}")

        combined_content = "\n\n---\n\n".join(all_content_items)
        max_mem_len = 10000 # Example limit
        summarized_content = combined_content[:max_mem_len] + ("..." if len(combined_content) > max_mem_len else "")

        if summarized_content:
            memory_section = "\n\n## Previous Chat History (Memory)\n" + summarized_content
            logger.debug(f"Appending memory summary ({len(summarized_content)} chars) to system prompt.")
            if "## Previous Chat History (Memory)" not in initial_system_prompt:
                return initial_system_prompt + memory_section
            else:
                logger.warning("Memory section already present in system prompt, not appending again.")
                return initial_system_prompt
        else:
            logger.debug("No content extracted from previous chats for memory.")
            return initial_system_prompt

    except Exception as e:
        logger.error(f"Error reloading memory: {e}", exc_info=True)
        return initial_system_prompt # Return original on error


def display_help():
    """Prints available CLI commands and startup flags."""
    print("\nAvailable commands:")
    print("!help          - Display this help message")
    print("!exit          - Exit the chat")
    print("!clear         - Clear the current chat session history (in memory only)")
    print("!save          - Save current chat session to 'saved' folder in S3")
    print("!memory        - Toggle memory mode (loads chat history from 'saved' folder)")
    print("!listen-transcript - Toggle automatic transcript listening")
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

# format_chat_history is now imported from utils.s3_utils


# Main execution block
def main():
    global abort_requested
    try:
        config = AppConfig.from_env_and_args()
        setup_logging(config.debug)
        logger.info(f"Starting agent '{config.agent_name}' with config: {config}")

        if not config.session_id:
             timestamp = datetime.now().strftime('%Y%m%d-T%H%M%S')
             config.session_id = timestamp # Set session ID on config object
        event_id_str = config.event_id or '0000' # Ensure event_id is a string
        current_chat_file = f"chat_D{config.session_id}_aID-{config.agent_name}_eID-{event_id_str}.txt"
        logger.info(f"Chat session ID: {config.session_id}")
        logger.info(f"Chat filename: {current_chat_file}")

        last_saved_index = 0
        last_archive_index = 0
        web_thread = None

        # Start web interface if requested
        if config.interface_mode in ['web', 'web_only']:
            try:
                 # Import WebChat here locally to avoid top-level circular dependency issues
                 from web.web_chat import WebChat
                 web_interface = WebChat(config) # Pass the fully initialized config
                 web_thread = web_interface.run(port=config.web_port, debug=config.debug)
                 logger.info(f"Web interface starting on http://127.0.0.1:{config.web_port}")
            except ImportError as e:
                 logger.error(f"Failed to import WebChat: {e}. Web interface unavailable.", exc_info=True)
                 if config.interface_mode == 'web_only': print("Error: Web interface cannot start. Exiting.", file=sys.stderr); sys.exit(1)
                 else: print("Warning: Web interface failed to start, running CLI only.", file=sys.stderr); config.interface_mode = 'cli'
            except Exception as e:
                 logger.error(f"Failed to start web interface: {e}", exc_info=True)
                 if config.interface_mode == 'web_only': print("Error: Failed to start web interface. Exiting.", file=sys.stderr); sys.exit(1)
                 else: print("Warning: Failed to start web interface, running CLI only.", file=sys.stderr); config.interface_mode = 'cli'

            if config.interface_mode == 'web_only':
                print("\nRunning in web-only mode. Press Ctrl+C in the console running Flask to exit.")
                try:
                     while True: time.sleep(1) # Keep main thread alive
                except KeyboardInterrupt: print("\nShutting down web-only mode...")
                return # Exit main function for web-only

        # --- CLI Mode ---
        if config.interface_mode != 'web_only':
            print(f"\nAgent '{config.agent_name}' running.")
            if config.interface_mode == 'web': print(f"Web interface available at http://127.0.0.1:{config.web_port}")
            print("Enter message or type !help")

            try: client = Anthropic(api_key=config.anthropic_api_key); logger.info("CLI: Anthropic client initialized.")
            except Exception as e: logger.error(f"CLI: Failed to initialize Anthropic client: {e}", exc_info=True); print("Error: AI client init failed.", file=sys.stderr); sys.exit(1)

            retriever = None
            try:
                 retriever = RetrievalHandler(
                    index_name=config.index, agent_name=config.agent_name,
                    session_id=config.session_id, event_id=config.event_id
                 )
                 logger.info(f"CLI: RetrievalHandler initialized for index '{config.index}'.")
            except Exception as e: logger.error(f"CLI: Failed to initialize RetrievalHandler: {e}", exc_info=True); print("Warning: Document retrieval unavailable.", file=sys.stderr)

            conversation_history = []
            system_prompt = get_latest_system_prompt(config.agent_name) or "You are a helpful assistant."
            initial_context_messages = []
            frameworks = get_latest_frameworks(config.agent_name); context = get_latest_context(config.agent_name, config.event_id); docs = get_agent_docs(config.agent_name)
            if frameworks: initial_context_messages.append({"role": "system", "content": f"## Frameworks\n{frameworks}"}); logger.info("CLI: Loaded frameworks.")
            if context: initial_context_messages.append({"role": "system", "content": f"## Context\n{context}"}); logger.info("CLI: Loaded context.")
            if docs: initial_context_messages.append({"role": "system", "content": f"## Agent Documentation\n{docs}"}); logger.info("CLI: Loaded agent documentation.")
            conversation_history.extend(initial_context_messages)

            if config.memory: system_prompt = reload_memory(config.agent_name, config.memory, system_prompt); logger.info("CLI: Memory loaded.")

            logger.debug("\n=== CLI: Final System Prompt & Initial Context ===")
            logger.debug(f"Core System Prompt Length: {len(system_prompt)} chars")
            for i, msg in enumerate(initial_context_messages): logger.debug(f"Initial Context Message {i+1}: Role={msg['role']}, Length={len(msg['content'])}, Content='{msg['content'][:100]}...'")
            if "## Previous Chat History" in system_prompt: logger.debug("Memory section appended to system prompt.")

            transcript_state = TranscriptState()
            last_transcript_check = time.time(); TRANSCRIPT_CHECK_INTERVAL = 5
            if config.listen_transcript:
                config.listen_transcript_enabled = True; logger.info("CLI: Transcript listening enabled.")
                print("Attempting to load initial transcript...");
                if check_transcript_updates(transcript_state, conversation_history, config.agent_name, config.event_id, read_all=config.read_all): print("Initial transcript loaded.")
                else: print("No initial transcript content found.")
                last_transcript_check = time.time()
            else: config.listen_transcript_enabled = False

            print("\nUser: ", end='', flush=True)
            while True:
                try:
                    current_time = time.time()
                    if config.listen_transcript_enabled and (current_time - last_transcript_check > TRANSCRIPT_CHECK_INTERVAL):
                        if check_transcript_updates(transcript_state, conversation_history, config.agent_name, config.event_id, read_all=False): logger.debug("New transcript content added.")
                        last_transcript_check = current_time

                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        user_input = sys.stdin.readline().strip()
                        if not user_input: print("User: ", end='', flush=True); continue

                        if user_input.startswith('!'):
                            command = user_input[1:].lower()
                            print(f"Executing command: !{command}")
                            if command == 'exit': break
                            elif command == 'help': display_help()
                            elif command == 'clear':
                                conversation_history = list(initial_context_messages); last_saved_index = 0; last_archive_index = len(conversation_history)
                                print("Current session history cleared.")
                            elif command == 'save':
                                messages_to_save = conversation_history[last_saved_index:]
                                if not messages_to_save: print("No new messages to save.")
                                else:
                                     save_content = format_chat_history(messages_to_save)
                                     success, saved_filename = save_chat_to_s3(config.agent_name, save_content, config.event_id, is_saved=True, filename=current_chat_file)
                                     if success: last_saved_index = len(conversation_history); print(f"Chat manually saved to {saved_filename}")
                                     else: print("Error: Failed to save chat manually.")
                            elif command == 'memory':
                                if not config.memory:
                                     config.memory = [config.agent_name]; system_prompt = reload_memory(config.agent_name, config.memory, system_prompt)
                                     print("Memory mode ACTIVATED.")
                                else:
                                     config.memory = []; system_prompt = get_latest_system_prompt(config.agent_name) or "You are a helpful assistant."
                                     print("Memory mode DEACTIVATED.")
                            elif command == 'listen-transcript':
                                 config.listen_transcript_enabled = not config.listen_transcript_enabled
                                 if config.listen_transcript_enabled:
                                      print("Transcript listening ENABLED. Checking...");
                                      if check_transcript_updates(transcript_state, conversation_history, config.agent_name, config.event_id, read_all=config.read_all): print("Transcript loaded/updated.")
                                      else: print("No new transcript content.")
                                      last_transcript_check = time.time()
                                 else: print("Transcript listening DISABLED.")
                            else: print(f"Unknown command: !{command}")
                            print("\nUser: ", end='', flush=True); continue

                        conversation_history.append({"role": "user", "content": user_input})
                        retrieved_docs = retriever.get_relevant_context(user_input) if retriever else []

                        context_for_prompt = ""
                        if retrieved_docs:
                             context_items = []
                             for i, doc in enumerate(retrieved_docs):
                                  source_file = doc.metadata.get('file_name', 'Unknown source'); score = doc.metadata.get('score', 0.0)
                                  context_items.append(f"[Context {i+1} from {source_file} (Score: {score:.2f})]:\n{doc.page_content}")
                             context_for_prompt = "\n\n---\nRelevant Context Found:\n" + "\n\n".join(context_items)
                             logger.debug(f"CLI: Adding retrieved context ({len(context_for_prompt)} chars).")
                        else: logger.debug("CLI: No relevant context retrieved.")

                        current_turn_system_prompt = system_prompt + context_for_prompt
                        print("Agent: ", end='', flush=True)
                        response_text = analyze_with_claude(client, conversation_history, current_turn_system_prompt, config.llm_model_name)

                        if response_text:
                            print(response_text)
                            conversation_history.append({"role": "assistant", "content": response_text})
                            messages_to_archive = conversation_history[last_archive_index:]
                            if messages_to_archive:
                                 archive_content = format_chat_history(messages_to_archive)
                                 success, _ = save_chat_to_s3(config.agent_name, archive_content, config.event_id, is_saved=False, filename=current_chat_file)
                                 if success: last_archive_index = len(conversation_history); logger.debug("CLI: Auto-archived turn.")
                                 else: logger.error("CLI: Failed to auto-archive chat turn.")
                        else: print("[Error processing request]")
                        print("\nUser: ", end='', flush=True)

                except (EOFError, KeyboardInterrupt): print("\nExiting chat."); break
                except Exception as loop_e: logger.error(f"Error in main chat loop: {loop_e}", exc_info=True); print(f"\nError: {loop_e}. Check logs.", file=sys.stderr); time.sleep(1); print("\nUser: ", end='', flush=True)

    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nA critical error occurred: {e}. Check log for details.", file=sys.stderr)
        sys.exit(1)
    finally:
         logging.info("Application shutting down.")

if __name__ == '__main__':
    main()