from flask import Flask, request, jsonify, render_template, current_app, Response
from typing import Optional, List, Dict, Any
import threading
import os
import logging
from datetime import datetime, time as dt_time, timezone # Added timezone
import json
import time
import traceback

# Import tenacity for retries
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

from config import AppConfig
from utils.retrieval_handler import RetrievalHandler
from utils.transcript_utils import TranscriptState, read_new_transcript_content, read_all_transcripts_in_folder, get_latest_transcript_file
from utils.s3_utils import (
    get_latest_system_prompt, get_latest_frameworks, get_latest_context,
    get_agent_docs, load_existing_chats_from_s3, save_chat_to_s3, format_chat_history,
    get_s3_client
)
from anthropic import Anthropic, APIStatusError # Import specific error type

logger = logging.getLogger(__name__)

# --- Tenacity Retry Configuration ---
def log_retry_error(retry_state):
    """Log retry attempts."""
    logger.warning(f"Retrying Anthropic API call (attempt {retry_state.attempt_number}): {retry_state.outcome.exception()}")

# Define retry strategy: Retry on APIStatusError (like Overloaded), wait exponentially, max 3 attempts
# Wait 2^x * 1 seconds between each retry, starting with x=1 (2s), then 4s. Max wait 10s. Stop after 3 attempts.
retry_strategy = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_error_callback=log_retry_error,
    retry=(retry_if_exception_type(APIStatusError)) # Only retry on specific API status errors
)
# --- End Tenacity Config ---


class WebChat:
    def __init__(self, config: AppConfig):
        self.config = config
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        self.app.config['SECRET_KEY'] = os.urandom(24)
        self.setup_routes()
        self.chat_history = []
        self.client = None
        self.system_prompt = "Default system prompt."
        self.retriever = None
        self.transcript_state = TranscriptState()
        self.last_saved_index = 0 # Tracks index in chat_history *for saving*
        self.last_archive_index = 0 # Tracks index in chat_history *for archiving*
        self.current_chat_file = None # Base filename for archive
        self.initial_transcript_content: Optional[str] = None
        self.initial_transcript_total_bytes: int = 0
        self.initial_transcript_sent: bool = False
        self.pending_transcript_update: Optional[str] = None
        self.transcript_lock = threading.Lock()

        try:
             if not hasattr(config, 'session_id') or not config.session_id: config.session_id = datetime.now().strftime('%Y%m%d-T%H%M%S')
             event_id = config.event_id or '0000'
             self.current_chat_file = f"chat_D{config.session_id}_aID-{config.agent_name}_eID-{event_id}.txt"
             logger.info(f"WebChat: Session {config.session_id}, Chat file: {self.current_chat_file}")
             self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
             logger.info("WebChat: Anthropic client initialized.")
             self.load_resources()
        except Exception as e: logger.error(f"WebChat: Init error: {e}", exc_info=True); raise RuntimeError("WebChat initialization failed") from e

    def get_document_context(self, query: str) -> Optional[List[Any]]:
        if not self.retriever: logger.error("WebChat: Retriever missing."); return None
        try:
            logger.debug(f"WebChat: Getting context for query: {query[:100]}...")
            is_transcript = any(word in query.lower() for word in ['transcript', 'conversation', 'meeting', 'session', 'said'])
            contexts = self.retriever.get_relevant_context(query=query, top_k=10, is_transcript=is_transcript)
            if not contexts: logger.info("WebChat: No relevant context found."); return None
            logger.info(f"WebChat: Retrieved {len(contexts)} relevant context docs.")
            return contexts
        except Exception as e: logger.error(f"WebChat: Error getting context: {e}", exc_info=True); return None

    def load_resources(self):
        if not self.client: logger.error("WebChat load_resources: Client missing."); self.system_prompt = "Error: Client missing."; return
        try:
            logger.info("WebChat: Loading resources...")
            base_prompt = get_latest_system_prompt(self.config.agent_name) or "Assistant."
            logger.info(f"Base prompt loaded ({len(base_prompt)} chars).")
            # Combine instructions into the base prompt permanently during loading
            source_instr = "\n\n## Source Attribution Requirements\n1. ALWAYS specify the exact source file..."
            realtime_instr = "\n\nIMPORTANT: Prioritize [REAL-TIME...] updates..."
            synth_instr = "\nSynthesizing from Context: When answering... combine related pieces... Do not state incomplete..."
            self.system_prompt = base_prompt + source_instr + realtime_instr + synth_instr

            frameworks=get_latest_frameworks(self.config.agent_name); context=get_latest_context(self.config.agent_name, self.config.event_id); docs=get_agent_docs(self.config.agent_name)
            if frameworks: self.system_prompt += "\n\n## Frameworks\n" + frameworks; logger.info("Frameworks loaded.")
            if context: self.system_prompt += "\n\n## Context\n" + context; logger.info("Context loaded.")
            if docs: self.system_prompt += "\n\n## Agent Documentation\n" + docs; logger.info("Docs loaded.")

            self.retriever = RetrievalHandler(
                index_name=self.config.index, agent_name=self.config.agent_name,
                session_id=self.config.session_id, event_id=self.config.event_id,
                anthropic_client=self.client
            )
            logger.info("Retriever initialized.")
            if self.config.memory is not None: self.reload_memory(); logger.info("Memory loaded.")
            if self.config.listen_transcript: self.load_initial_transcript()
            logger.info("WebChat: Resources loaded ok.")
            logger.debug(f"Final system prompt len (before time inject): {len(self.system_prompt)} chars.")
        except Exception as e:
            logger.error(f"WebChat: Error loading resources: {e}", exc_info=True)
            if not self.system_prompt or self.system_prompt == "Default system prompt.": self.system_prompt = "Error loading config."


    def reload_memory(self):
        logger.debug("WebChat: Reloading memory...")
        try:
             agents_to_load = self.config.memory if self.config.memory else [self.config.agent_name]
             previous_chats = load_existing_chats_from_s3(self.config.agent_name, agents_to_load)
             if not previous_chats: logger.debug("No memory files found."); return
             all_items = [f"{msg.get('role','?').capitalize()} (File:{os.path.basename(c.get('file','?'))}): {msg.get('content','')}" for c in previous_chats for msg in c.get('messages',[]) if msg.get('content')]
             combined = "\n\n---\n\n".join(all_items); max_len = 10000
             summary = combined[:max_len] + ("..." if len(combined) > max_len else "")
             if summary:
                 mem_section = "\n\n## Previous Chat History (Memory)\n" + summary
                 start_index = self.system_prompt.find("## Previous Chat History")
                 if start_index != -1:
                     # Find the start of the *next* section or end of string
                     next_section_index = self.system_prompt.find("\n\n## ", start_index + len("## Previous Chat History"))
                     end_index = next_section_index if next_section_index != -1 else len(self.system_prompt)
                     self.system_prompt = self.system_prompt[:start_index].strip() + self.system_prompt[end_index:].strip()
                     logger.info(f"Replacing memory section.")
                 self.system_prompt += mem_section; logger.info(f"Appended/Replaced memory ({len(summary)} chars).")
             else: logger.debug("No content for memory.")
        except Exception as e: logger.error(f"Memory reload error: {e}", exc_info=True)

    # --- Helper for API call with Retries ---
    @retry_strategy
    def _call_anthropic_stream_with_retry(self, model, max_tokens, system, messages):
        """Calls Anthropic stream API with configured retry logic."""
        # The actual API call is now wrapped by the tenacity decorator
        return self.client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages
        )
    # --- End Helper ---

    def setup_routes(self):
        @self.app.route('/')
        def index():
            tmpl = 'index.html'
            logger.debug(f"Rendering template: {tmpl}")
            return render_template(tmpl, agent_name=self.config.agent_name)

        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            try:
                data = request.json
                if not data or 'message' not in data: return jsonify({'error': 'No message'}), 400
                user_msg_content = data['message']
                logger.info(f"WebChat: Received msg: {user_msg_content[:100]}...")

                turn_system_prompt = self.system_prompt
                llm_messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in self.chat_history
                    if m.get("role") in ["user", "assistant"]
                ]
                logger.debug(f"Building context from history ({len(llm_messages)} msgs).")

                retrieved_docs = self.get_document_context(user_msg_content)
                if retrieved_docs:
                     items = [f"[Ctx {i+1} from {d.metadata.get('file_name','?')}({d.metadata.get('score',0):.2f})]:\n{d.page_content}" for i, d in enumerate(retrieved_docs)]
                     context_block = "\n\n---\nRetrieved Context:\n" + "\n\n".join(items)
                     logger.debug(f"Adding context ({len(context_block)} chars) to sys prompt.")
                     turn_system_prompt += context_block
                else: logger.debug("No context retrieved.")

                transcript_content_to_add = "";
                if self.config.listen_transcript:
                    if not self.initial_transcript_sent:
                        if self.initial_transcript_content:
                            label = "[FULL INITIAL TRANSCRIPT]"; transcript_content_to_add = f"{label}\n{self.initial_transcript_content}"
                            self.initial_transcript_sent = True; logger.info(f"Adding FULL initial tx ({len(self.initial_transcript_content)} chars).")
                        else: logger.warning("Tx listening on, but no initial tx."); self.initial_transcript_sent = True
                    else:
                        with self.transcript_lock:
                            if self.pending_transcript_update:
                                label = "[REAL-TIME Meeting Transcript Update]"; transcript_content_to_add = f"{label}\n{self.pending_transcript_update}"
                                logger.info(f"Adding PENDING real-time tx ({len(self.pending_transcript_update)} chars).")
                                self.pending_transcript_update = None
                if transcript_content_to_add:
                     llm_messages.append({'role': 'user', 'content': transcript_content_to_add})
                     self.chat_history.append({'role': 'user', 'content': transcript_content_to_add, 'type': 'transcript_update'})

                llm_messages.append({'role': 'user', 'content': user_msg_content})
                self.chat_history.append({'role': 'user', 'content': user_msg_content})
                logger.debug(f"Appended user message(s). History size: {len(self.chat_history)}")

                if not self.client: return jsonify({'error': 'AI client missing'}), 500
                model=self.config.llm_model_name; max_tokens=self.config.llm_max_output_tokens
                logger.debug(f"WebChat: Using LLM: {model}, MaxTokens: {max_tokens}")

                now_utc = datetime.now(timezone.utc)
                time_str = now_utc.strftime('%A, %Y-%m-%d %H:%M:%S %Z')
                time_context = f"Current Time Context: {time_str}\n"
                final_system_prompt_for_turn = time_context + "\n" + turn_system_prompt
                logger.debug(f"Final sys prompt for turn includes time. Len: {len(final_system_prompt_for_turn)}")

                def generate():
                    response_content = ""; stream_error = None
                    try:
                        logger.debug(f"LLM call (with retry). Msgs: {len(llm_messages)}, Final SysPromptLen: {len(final_system_prompt_for_turn)}")
                        if logger.isEnabledFor(logging.DEBUG): last_msg=llm_messages[-1]; logger.debug(f" -> Last Msg: Role={last_msg['role']}, Len={len(last_msg['content'])}, Starts='{last_msg['content'][:150]}...'")

                        # Use the retry-enabled helper method
                        with self._call_anthropic_stream_with_retry(
                            model=model,
                            max_tokens=max_tokens,
                            system=final_system_prompt_for_turn,
                            messages=llm_messages
                        ) as stream:
                            for text in stream.text_stream:
                                response_content += text
                                yield f"data: {json.dumps({'delta': text})}\n\n"
                        logger.info(f"LLM response ok ({len(response_content)} chars).")

                    except RetryError as e: # Catch error if all retries fail
                         logger.error(f"Anthropic API call failed after multiple retries: {e}", exc_info=True)
                         stream_error = "Assistant is currently unavailable after multiple retries. Please try again later."
                    except APIStatusError as e: # Catch specific API status errors not retried
                        logger.error(f"Anthropic API Status Error (non-retryable or final attempt): {e}", exc_info=True)
                        # Check if it's overload specifically for a better message
                        if 'overloaded_error' in str(e).lower():
                            stream_error = "Assistant is busy, please try again shortly."
                        else:
                            stream_error = f"API Error: {e.message}" if hasattr(e, 'message') else str(e)
                    except Exception as e: # Catch other potential errors
                         if "aborted" in str(e).lower() or "cancel" in str(e).lower():
                              logger.warning(f"LLM stream aborted or cancelled: {e}")
                              stream_error="Stream stopped."
                         else:
                              logger.error(f"LLM stream error: {e}", exc_info=True)
                              stream_error = str(e)

                    if stream_error: yield f"data: {json.dumps({'error': f'Error: {stream_error}'})}\n\n"

                    if not stream_error and response_content:
                        self.chat_history.append({'role': 'assistant', 'content': response_content})
                        logger.debug(f"Appended assistant response. History size: {len(self.chat_history)}")

                    archive_msgs = self.chat_history[self.last_archive_index:]
                    if archive_msgs:
                        content_to_save = format_chat_history(archive_msgs)
                        if content_to_save:
                             success, _ = save_chat_to_s3(self.config.agent_name, content_to_save, self.config.event_id, False, self.current_chat_file)
                             if success: self.last_archive_index = len(self.chat_history); logger.debug("Auto-archived.")
                             else: logger.error("Failed auto-archive.")
                    yield f"data: {json.dumps({'done': True})}\n\n"

                return Response(generate(), mimetype='text/event-stream')
            except Exception as e: logger.error(f"/api/chat error: {e}", exc_info=True); return jsonify({'error': 'Server error'}), 500

        @self.app.route('/api/status', methods=['GET'])
        def status():
             mem=getattr(self.config,'memory',None) is not None; tx=getattr(self.config,'listen_transcript',False)
             init_tx_proc=self.initial_transcript_sent; proc_perc=100 if init_tx_proc else 0
             with self.transcript_lock: has_pending = self.pending_transcript_update is not None
             return jsonify({'agent_name': self.config.agent_name, 'listen_transcript': tx, 'memory_enabled': mem,
                            'initial_transcript_processed': init_tx_proc, 'initial_transcript_progress_percent': proc_perc,
                            'has_pending_transcript_update': has_pending})

        @self.app.route('/api/save', methods=['POST'])
        def save_chat_api():
            logger.info("Received request to /api/save")
            try:
                raw_msgs_to_save = self.chat_history[self.last_saved_index:]
                filtered_msgs_to_save = [
                    m for m in raw_msgs_to_save
                    if m.get('role') in ['user', 'assistant'] and m.get('type') != 'transcript_update'
                ]
                logger.debug(f"API Save: Found {len(raw_msgs_to_save)} raw msgs since last save, filtered to {len(filtered_msgs_to_save)} user/assistant msgs.")

                if not filtered_msgs_to_save:
                    logger.info("API Save: No new user/assistant messages to save.")
                    return jsonify({'message': 'No new messages to save.'}), 200

                content = format_chat_history(filtered_msgs_to_save)
                timestamp = datetime.now().strftime('%Y%m%d-T%H%M%S')
                clean_save_filename = f"chat_D{timestamp}_aID-{self.config.agent_name}_eID-{self.config.event_id}_SAVED.txt"
                logger.info(f"API Save: Attempting to save clean chat to: {clean_save_filename}")

                s3_client = get_s3_client()
                aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
                save_success = False
                saved_key = None

                if s3_client and aws_s3_bucket:
                     try:
                         base_path = f"organizations/river/agents/{self.config.agent_name}/events/{self.config.event_id or '0000'}/chats/saved"
                         saved_key = f"{base_path}/{clean_save_filename}"
                         s3_client.put_object(Bucket=aws_s3_bucket, Key=saved_key, Body=content.encode('utf-8'), ContentType='text/plain; charset=utf-8')
                         save_success = True
                         logger.info(f"API Save: Successfully saved clean chat to: {saved_key}")
                     except Exception as put_e:
                         logger.error(f"API Save: Error directly saving clean chat to {saved_key}: {put_e}", exc_info=True)
                         save_success = False

                if save_success:
                    self.last_saved_index = self.last_saved_index + len(raw_msgs_to_save)
                    return jsonify({'message': f'Saved chat as {clean_save_filename}'}), 200
                else:
                    return jsonify({'error': 'Error saving chat to S3.'}), 500

            except Exception as e:
                logger.error(f"Error in /api/save route: {e}", exc_info=True)
                return jsonify({'error': 'Internal server error during save.'}), 500

        @self.app.route('/api/command', methods=['POST'])
        def command():
            data = request.json; cmd = data.get('command','').lower(); msg = f"Cmd: !{cmd}"; code=200; resp={}
            if not cmd: return jsonify({'error': 'No command'}), 400
            logger.info(f"WebChat: Cmd: !{cmd}")
            try:
                # Command logic...
                if cmd == 'help': msg = "Cmds: !help, !clear, !save, !memory, !listen-transcript"
                elif cmd == 'clear':
                    self.chat_history=[]; self.last_saved_index=0; self.last_archive_index=0
                    self.initial_transcript_content=None; self.initial_transcript_sent=False; self.transcript_state = TranscriptState()
                    with self.transcript_lock: self.pending_transcript_update = None
                    if self.config.listen_transcript: self.load_initial_transcript(); msg='History/Tx state cleared. Initial tx reloaded.'
                    else: msg='History/Tx state cleared.'
                elif cmd == 'save':
                    # Replicate the logic from /api/save for consistency
                    raw_msgs_to_save = self.chat_history[self.last_saved_index:]
                    filtered_msgs_to_save = [
                        m for m in raw_msgs_to_save
                        if m.get('role') in ['user', 'assistant'] and m.get('type') != 'transcript_update'
                    ]
                    logger.debug(f"!save Cmd: Found {len(raw_msgs_to_save)} raw msgs, filtered to {len(filtered_msgs_to_save)} user/assistant msgs.")

                    if not filtered_msgs_to_save:
                        msg='Nothing new to save (only non-chat messages found).'
                        code = 200
                    else:
                        content = format_chat_history(filtered_msgs_to_save)
                        timestamp = datetime.now().strftime('%Y%m%d-T%H%M%S')
                        clean_save_filename = f"chat_D{timestamp}_aID-{self.config.agent_name}_eID-{self.config.event_id}_SAVED_CMD.txt" # Added _CMD suffix
                        logger.info(f"!save Cmd: Attempting to save clean chat to: {clean_save_filename}")

                        s3_client = get_s3_client()
                        aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
                        save_success = False
                        saved_key = None

                        if s3_client and aws_s3_bucket:
                             try:
                                 base_path = f"organizations/river/agents/{self.config.agent_name}/events/{self.config.event_id or '0000'}/chats/saved"
                                 saved_key = f"{base_path}/{clean_save_filename}"
                                 s3_client.put_object(Bucket=aws_s3_bucket, Key=saved_key, Body=content.encode('utf-8'), ContentType='text/plain; charset=utf-8')
                                 save_success = True
                                 logger.info(f"!save Cmd: Successfully saved clean chat to: {saved_key}")
                             except Exception as put_e:
                                 logger.error(f"!save Cmd: Error directly saving clean chat to {saved_key}: {put_e}", exc_info=True)
                                 save_success = False

                        if save_success:
                            self.last_saved_index = self.last_saved_index + len(raw_msgs_to_save)
                            msg = f'Saved chat as {clean_save_filename}'
                        else:
                            msg = 'Error saving chat.'; code = 500
                elif cmd == 'memory':
                     if self.config.memory is None: self.config.memory = [self.config.agent_name]; self.reload_memory(); msg='Memory ON.'
                     else: self.config.memory = None; self.load_resources(); msg='Memory OFF.' # Reload resources to remove memory section
                elif cmd == 'listen-transcript':
                     self.config.listen_transcript = not self.config.listen_transcript; status = "ENABLED" if self.config.listen_transcript else "DISABLED"
                     if self.config.listen_transcript: loaded = self.load_initial_transcript(); msg = f"Tx listening {status}." + (" Initial tx loaded." if loaded else " No initial tx found.")
                     else: self.initial_transcript_content=None; self.initial_transcript_sent=False; self.transcript_state=TranscriptState(); msg = f"Tx listening {status}."
                else: msg = f"Unknown cmd: !{cmd}"; code = 400

                resp['message'] = msg
                if code == 200:
                    init_tx_processed_flag = self.initial_transcript_sent
                    processed_percent = 100 if init_tx_processed_flag else 0
                    with self.transcript_lock:
                         has_pending_update = self.pending_transcript_update is not None
                    resp['status'] = {
                        'listen_transcript': self.config.listen_transcript,
                        'memory_enabled': self.config.memory is not None,
                        'initial_transcript_processed': init_tx_processed_flag,
                        'initial_transcript_progress_percent': processed_percent,
                        'has_pending_transcript_update': has_pending_update
                    }
                else:
                    resp['error'] = msg # Assign error message if code is not 200
                return jsonify(resp), code
            except Exception as e: logger.error(f"Cmd !{cmd} error: {e}", exc_info=True); return jsonify({'error': str(e)}), 500


    def load_initial_transcript(self):
        self.initial_transcript_content=None; self.initial_transcript_sent=False; self.transcript_state=TranscriptState();
        with self.transcript_lock: self.pending_transcript_update=None
        logger.info("Loading initial transcript & initializing S3 state...")
        s3=get_s3_client(); bucket=os.getenv('AWS_S3_BUCKET'); key=None; s3_size=0; s3_mod=None
        if not s3 or not bucket: logger.error("S3 unavailable for tx init."); return False
        try: key=get_latest_transcript_file(self.config.agent_name, self.config.event_id, s3)
        except Exception as e: logger.error(f"Error getting latest tx key: {e}"); key=None
        if key:
            try: meta=s3.head_object(Bucket=bucket, Key=key); s3_size=meta['ContentLength']; s3_mod=meta['LastModified']; logger.info(f"Latest S3 tx: '{key}', Size: {s3_size}, Mod: {s3_mod}")
            except Exception as e: logger.error(f"Failed head object for '{key}': {e}"); key=None
        else: logger.warning("Could not find latest S3 tx key.")
        try: full_content=read_all_transcripts_in_folder(self.config.agent_name, self.config.event_id)
        except Exception as e: logger.error(f"Error reading all transcripts: {e}"); full_content=None
        if isinstance(full_content, str):
            self.initial_transcript_content=full_content; self.initial_transcript_total_bytes=len(full_content.encode('utf-8'))
            logger.info(f"Stored initial tx ({len(full_content)} chars / {self.initial_transcript_total_bytes} bytes).")
            if key and s3_mod: self.transcript_state.file_positions[key]=s3_size; self.transcript_state.last_modified[key]=s3_mod; self.transcript_state.current_latest_key=key; logger.info(f"Initialized TxState for S3 key '{key}'.")
            else: logger.warning("Could not initialize TxState S3 info.")
            return True
        else: logger.error("Loaded initial tx is not a string or failed."); self.initial_transcript_content=None; return False

    def check_transcript_updates(self):
        if not self.config.listen_transcript: return
        try:
            new_content = read_new_transcript_content(self.transcript_state, self.config.agent_name, self.config.event_id, False)
            if new_content:
                 with self.transcript_lock:
                      if self.pending_transcript_update is None: self.pending_transcript_update = new_content; logger.info(f"BG check INITIALIZED pending tx ({len(new_content)} chars).")
                      else: self.pending_transcript_update += "\n" + new_content; logger.info(f"BG check APPENDED to pending tx ({len(new_content)} chars). Total: {len(self.pending_transcript_update)} chars.")
        except Exception as e: logger.error(f"BG tx check error: {e}", exc_info=True)

    def run(self, host: str = '127.0.0.1', port: int = 5001, debug: bool = False):
        def transcript_update_loop():
             logger.info("Starting background transcript state update loop.")
             while True: time.sleep(5); self.check_transcript_updates()
        if self.config.interface_mode != 'cli':
            update_thread = threading.Thread(target=transcript_update_loop, daemon=True); update_thread.start()
        logger.info(f"Starting Flask server on {host}:{port}, Debug: {debug}")
        try: self.app.run(host=host, port=port, debug=debug, use_reloader=False)
        except Exception as e: logger.critical(f"Flask server failed: {e}", exc_info=True)