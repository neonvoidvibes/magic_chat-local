import os
import sys
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

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
from typing import Optional, List, Dict, Any

# Import shared utilities
from utils.s3_utils import (
    get_latest_system_prompt, get_latest_frameworks, get_latest_context,
    get_agent_docs, load_existing_chats_from_s3, save_chat_to_s3, format_chat_history
)
from utils.retrieval_handler import RetrievalHandler
from utils.transcript_utils import TranscriptState, get_latest_transcript_file, read_new_transcript_content, read_all_transcripts_in_folder
from config import AppConfig

logger = logging.getLogger(__name__)

# --- check_transcript_updates function ---
def check_transcript_updates(transcript_state: TranscriptState, conversation_history: List[Dict[str, Any]], agent_name: str, event_id: str, read_all: bool = False) -> bool:
    """Checks for transcript updates and appends new content to history."""
    logger.debug("Checking for transcript updates...")
    new_content = read_new_transcript_content(transcript_state, agent_name, event_id, read_all=read_all)
    if new_content:
        label = "[REAL-TIME Meeting Transcript Update]"
        logger.debug(f"Adding new transcript content: {new_content[:100]}...")
        conversation_history.append({"role": "user", "content": f"{label}\n{new_content}"})
        return True
    return False

# --- setup_logging function ---
def setup_logging(debug: bool):
    """Sets up logging configuration."""
    log_filename = 'claude_chat.log'; root_logger = logging.getLogger(); log_level = logging.DEBUG if debug else logging.INFO
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
    try:
        fh = logging.FileHandler(log_filename, encoding='utf-8'); fh.setLevel(log_level)
        ff = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'); fh.setFormatter(ff); root_logger.addHandler(fh)
    except Exception as e: print(f"File logger error: {e}", file=sys.stderr)
    ch = logging.StreamHandler(sys.stdout); cl = logging.DEBUG if debug else logging.INFO; ch.setLevel(cl)
    cf = logging.Formatter('[%(levelname)-8s] %(message)s'); ch.setFormatter(cf); root_logger.addHandler(ch)
    for lib in ['anthropic', 'httpx', 'boto3', 'botocore', 'urllib3', 's3transfer']: logging.getLogger(lib).setLevel(logging.WARNING)
    logging.getLogger('pinecone').setLevel(logging.INFO); logging.getLogger('utils').setLevel(log_level)
    logger.info(f"Logging setup complete. Level: {logging.getLevelName(log_level)}")

# --- analyze_with_claude function ---
def analyze_with_claude(client: Anthropic, messages: List[Dict[str, Any]], system_prompt: str, model_name: str, max_tokens: int) -> Optional[str]:
    """Process messages with Claude API."""
    logger.debug(f"\n=== Claude API Request === Model: {model_name}, MaxTokens: {max_tokens}")
    logger.debug(f"System prompt length: {len(system_prompt)} chars")
    formatted_messages = []
    for msg in messages:
         role = msg.get("role"); content = msg.get("content", "")
         if not content or role == "system": continue
         api_role = "assistant" if role == "assistant" else "user"
         formatted_messages.append({"role": api_role, "content": content})
    realtime_instruction = "\n\nIMPORTANT: When asked about what is happening 'now' or for live updates, prioritize information marked with [REAL-TIME Meeting Transcript Update] over historical context from the vector database."
    final_system_prompt = system_prompt + realtime_instruction
    logger.debug(f"Final system prompt length (with instructions): {len(final_system_prompt)} chars")
    logger.debug(f"Number of messages for API: {len(formatted_messages)}")
    if logger.isEnabledFor(logging.DEBUG):
        for i, msg in enumerate(formatted_messages[-5:]): logger.debug(f"  Msg {len(formatted_messages)-5+i}: Role={msg['role']}, Len={len(msg['content'])}, Content='{msg['content'][:100]}...'")
    try:
        response = client.messages.create(model=model_name, system=final_system_prompt, messages=formatted_messages, max_tokens=max_tokens)
        response_text = response.content[0].text
        logger.debug("\n=== Claude API Response ==="); logger.debug(f"Len: {len(response_text)}. Start: {response_text[:100]}...")
        return response_text
    except AnthropicError as e: logger.error(f"Anthropic API Error ({model_name}): {e}"); return f"Error: {e}"
    except Exception as e: logger.error(f"Claude Error ({model_name}): {e}", exc_info=True); return f"Error: {e}"

# --- reload_memory function ---
def reload_memory(agent_name: str, memory_agents: List[str], initial_system_prompt: str) -> str:
    """Reload memory and append to system prompt."""
    try:
        logger.debug("Reloading memory..."); chats = load_existing_chats_from_s3(agent_name, memory_agents)
        if not chats: return initial_system_prompt
        items = [f"{msg.get('role','?').capitalize()} (File:{os.path.basename(c.get('file','?'))}): {msg.get('content','')}" for c in chats for msg in c.get('messages',[]) if msg.get('content')]
        combined = "\n\n---\n\n".join(items); max_len = 10000; summary = combined[:max_len] + ("..." if len(combined) > max_len else "")
        if summary:
            mem_section = "\n\n## Previous Chat History (Memory)\n" + summary; logger.debug(f"Appending memory ({len(summary)} chars).")
            if "## Previous Chat History" not in initial_system_prompt: return initial_system_prompt + mem_section
            else: logger.warning("Memory section exists."); return initial_system_prompt
        else: return initial_system_prompt
    except Exception as e: logger.error(f"Memory reload error: {e}", exc_info=True); return initial_system_prompt

# --- display_help function ---
def display_help():
    """Prints CLI help."""
    print("\nCommands: !help, !exit, !clear, !save, !memory, !listen-transcript")
    print("Flags: --agent, --index, --event, --memory, --listen-transcript, --all, --web, --web-only, --web-port, --max-tokens, --debug")

# --- Main execution block ---
def main():
    global abort_requested
    try:
        config = AppConfig.from_env_and_args(); setup_logging(config.debug)
        logger.info(f"Starting agent '{config.agent_name}' with config: {config}")
        if not config.session_id: config.session_id = datetime.now().strftime('%Y%m%d-T%H%M%S')
        event_id_str = config.event_id or '0000'
        current_chat_file = f"chat_D{config.session_id}_aID-{config.agent_name}_eID-{event_id_str}.txt"
        logger.info(f"Session: {config.session_id}, File: {current_chat_file}")
        last_saved_idx=0; last_archive_idx=0; web_thread=None

        if config.interface_mode in ['web', 'web_only']:
            try:
                 from web.web_chat import WebChat # Import locally
                 web_interface = WebChat(config)
                 web_thread = web_interface.run(port=config.web_port, debug=config.debug)
                 logger.info(f"Web interface starting: http://127.0.0.1:{config.web_port}")
            except Exception as e:
                 logger.error(f"Web start failed: {e}", exc_info=True)
                 if config.interface_mode == 'web_only': print("Error: Web failed. Exiting.", file=sys.stderr); sys.exit(1)
                 else: print("Warning: Web failed, CLI only.", file=sys.stderr); config.interface_mode = 'cli'

            if config.interface_mode == 'web_only':
                print("\nWeb-only mode. Ctrl+C in Flask console to exit.")
                try:
                     # Correct indentation fixed again
                     while True:
                          time.sleep(1)
                except KeyboardInterrupt:
                     print("\nExiting web-only...")
                return # Exit main for web_only

        if config.interface_mode != 'web_only':
            print(f"\nAgent '{config.agent_name}' running (CLI).")
            if config.interface_mode == 'web': print(f"Web UI: http://127.0.0.1:{config.web_port}")
            print("Enter message or type !help")
            try: client = Anthropic(api_key=config.anthropic_api_key); logger.info("CLI: Anthropic client ok.")
            except Exception as e: logger.error(f"CLI: Anthropic client failed: {e}", exc_info=True); print("Error: AI client failed.", file=sys.stderr); sys.exit(1)
            retriever = None
            try: retriever = RetrievalHandler(index_name=config.index, agent_name=config.agent_name, session_id=config.session_id, event_id=config.event_id); logger.info(f"CLI: Retriever ok.")
            except Exception as e: logger.error(f"CLI: Retriever failed: {e}", exc_info=True); print("Warning: Doc retrieval unavailable.", file=sys.stderr)

            conversation_history = []
            system_prompt = get_latest_system_prompt(config.agent_name) or "Assistant."
            logger.info(f"Base system prompt loaded ({len(system_prompt)} chars).")
            initial_context_messages = []
            frameworks=get_latest_frameworks(config.agent_name); context=get_latest_context(config.agent_name, config.event_id); docs=get_agent_docs(config.agent_name)
            if frameworks: initial_context_messages.append({"role": "system", "content": f"## Frameworks\n{frameworks}"}); logger.info("CLI: Frameworks loaded.")
            if context: initial_context_messages.append({"role": "system", "content": f"## Context\n{context}"}); logger.info("CLI: Context loaded.")
            if docs: initial_context_messages.append({"role": "system", "content": f"## Agent Docs\n{docs}"}); logger.info("CLI: Docs loaded.")
            conversation_history = initial_context_messages # Start history with system context blocks

            if config.memory: system_prompt = reload_memory(config.agent_name, config.memory, system_prompt); logger.info("CLI: Memory loaded.")

            logger.debug("\n=== CLI: Final System Prompt & Initial Context ===")
            logger.debug(f"Core Sys Prompt Len: {len(system_prompt)}")
            for i, msg in enumerate(initial_context_messages): logger.debug(f"Init Ctx Msg {i+1}: Role={msg['role']}, Len={len(msg['content'])}, Content='{msg['content'][:100]}...'")
            if "## Previous Chat History" in system_prompt: logger.debug("Memory section appended.")

            transcript_state = TranscriptState()
            last_transcript_check = time.time(); TRANSCRIPT_CHECK_INTERVAL = 5
            if config.listen_transcript:
                config.listen_transcript_enabled = True; logger.info("CLI: Tx listening enabled.")
                print("Checking initial transcript...");
                if check_transcript_updates(transcript_state, conversation_history, config.agent_name, config.event_id, True): print("Initial tx loaded.")
                else: print("No initial tx found.")
                last_transcript_check = time.time()
            else: config.listen_transcript_enabled = False

            print("\nUser: ", end='', flush=True)
            while True:
                try:
                    current_time = time.time()
                    if config.listen_transcript_enabled and (current_time - last_transcript_check > TRANSCRIPT_CHECK_INTERVAL):
                        if check_transcript_updates(transcript_state, conversation_history, config.agent_name, config.event_id, False): logger.debug("New tx added.")
                        last_transcript_check = current_time

                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        user_input = sys.stdin.readline().strip()
                        if not user_input: print("User: ", end='', flush=True); continue

                        if user_input.startswith('!'):
                            # Command handling logic... (kept concise)
                            command = user_input[1:].lower(); print(f"Cmd: !{command}")
                            if command == 'exit': break
                            elif command == 'help': display_help()
                            elif command == 'clear': conversation_history = list(initial_context_messages); last_saved_idx=0; last_archive_idx=len(conversation_history); print("Session history cleared.")
                            elif command == 'save':
                                msgs = conversation_history[last_saved_idx:];
                                if not msgs: print("No new messages.")
                                else: content=format_chat_history(msgs); success, fname=save_chat_to_s3(config.agent_name, content, config.event_id, True, current_chat_file);
                                if success: last_saved_idx = len(conversation_history); print(f"Saved as {fname}")
                                else: print("Error saving.")
                            elif command == 'memory':
                                if not config.memory: config.memory = [config.agent_name]; system_prompt = reload_memory(config.agent_name, config.memory, system_prompt); print("Memory ON.")
                                else: config.memory = []; system_prompt = get_latest_system_prompt(config.agent_name) or "Assistant."; print("Memory OFF.")
                            elif command == 'listen-transcript':
                                 config.listen_transcript_enabled = not config.listen_transcript_enabled; status = "ENABLED" if config.listen_transcript_enabled else "DISABLED"
                                 print(f"Tx listening {status}.")
                                 if config.listen_transcript_enabled:
                                      if check_transcript_updates(transcript_state, conversation_history, config.agent_name, config.event_id, True): print("Tx loaded/updated.")
                                      else: print("No new tx found."); last_transcript_check = time.time()
                            else: print(f"Unknown command: !{command}")
                            print("\nUser: ", end='', flush=True); continue

                        # --- Prepare for LLM ---
                        current_turn_history = list(conversation_history) # Copy history
                        # Add context block *before* user message if retrieved
                        retrieved_docs = retriever.get_relevant_context(user_input) if retriever else []
                        if retrieved_docs:
                             items = [f"[Ctx {i+1} from {d.metadata.get('file_name','?')}({d.metadata.get('score',0):.2f})]:\n{d.page_content}" for i, d in enumerate(retrieved_docs)]
                             context_block = "\n\n---\nRetrieved Context:\n" + "\n\n".join(items)
                             logger.debug(f"CLI: Adding context ({len(context_block)} chars).")
                             current_turn_history.append({"role": "system", "content": context_block})
                        else: logger.debug("CLI: No context found.")
                        # Add actual user message
                        current_turn_history.append({"role": "user", "content": user_input})
                        # Transcript updates are added automatically by check_transcript_updates if enabled

                        # Call LLM
                        print("Agent: ", end='', flush=True)
                        response_text = analyze_with_claude(
                             client, current_turn_history, system_prompt, # Pass base system prompt
                             config.llm_model_name, config.llm_max_output_tokens
                        )

                        # Process response
                        if response_text:
                            print(response_text)
                            # Add user message and assistant response to persistent history
                            conversation_history.append({"role": "user", "content": user_input}) # Add the user input that led to this
                            conversation_history.append({"role": "assistant", "content": response_text})
                            # Archive logic...
                            msgs_archive = conversation_history[last_archive_idx:]
                            if msgs_archive:
                                 content = format_chat_history(msgs_archive)
                                 success, _ = save_chat_to_s3(config.agent_name, content, config.event_id, False, current_chat_file)
                                 if success: last_archive_idx = len(conversation_history); logger.debug("CLI: Auto-archived.")
                                 else: logger.error("CLI: Failed archive.")
                        else: print("[Error]")
                        print("\nUser: ", end='', flush=True)

                except (EOFError, KeyboardInterrupt): print("\nExiting."); break
                except Exception as loop_e: logger.error(f"Loop error: {loop_e}", exc_info=True); print(f"\nError: {loop_e}", file=sys.stderr); time.sleep(1); print("\nUser: ", end='', flush=True)

    except Exception as e: logging.error(f"Fatal error: {e}", exc_info=True); print(f"\nCritical error: {e}.", file=sys.stderr); sys.exit(1)
    finally: logging.info("App shutdown.")

if __name__ == '__main__':
    main()