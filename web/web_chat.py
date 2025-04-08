from flask import Flask, request, jsonify, render_template, current_app, Response
from typing import Optional, List, Dict, Any
import threading
import os
import logging
from datetime import datetime
import json
import time
import traceback
import math

from config import AppConfig
from utils.retrieval_handler import RetrievalHandler
# Import get_latest_transcript_file directly
from utils.transcript_utils import TranscriptState, read_new_transcript_content, read_all_transcripts_in_folder, get_latest_transcript_file
# Import S3 client getter
from utils.s3_utils import get_s3_client
from utils.s3_utils import (
    get_latest_system_prompt, get_latest_frameworks, get_latest_context,
    get_agent_docs, load_existing_chats_from_s3, save_chat_to_s3, format_chat_history
)
from anthropic import Anthropic

logger = logging.getLogger(__name__)

# Constants removed

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
        self.transcript_state = TranscriptState() # State for read_new_transcript_content
        self.scheduler_thread = None
        self.last_saved_index = 0
        self.last_archive_index = 0
        self.current_chat_file = None

        # State for transcript processing
        self.initial_transcript_content: Optional[str] = None
        self.initial_transcript_total_bytes: int = 0
        self.initial_transcript_processed_bytes: int = 0 # 0 = not sent, >0 = sent

        try:
             if not hasattr(config, 'session_id') or not config.session_id: config.session_id = datetime.now().strftime('%Y%m%d-T%H%M%S')
             event_id = config.event_id or '0000'
             self.current_chat_file = f"chat_D{config.session_id}_aID-{config.agent_name}_eID-{event_id}.txt"
             logger.info(f"WebChat: Session {config.session_id}, Chat file: {self.current_chat_file}")
             self.load_resources()
             self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
             logger.info("WebChat: Anthropic client initialized.")
        except Exception as e: logger.error(f"WebChat: Init error: {e}", exc_info=True); raise RuntimeError("WebChat initialization failed") from e

    def get_document_context(self, query: str) -> Optional[List[Any]]:
        """Get relevant document context using the RetrievalHandler."""
        if not self.retriever: logger.error("WebChat: Retriever missing."); return None
        try:
            logger.debug(f"WebChat: Getting context for query: {query[:100]}...")
            is_transcript = any(word in query.lower() for word in ['transcript', 'conversation', 'meeting', 'session', 'said'])
            contexts = self.retriever.get_relevant_context(query=query, top_k=3, is_transcript=is_transcript)
            if not contexts: logger.info("WebChat: No relevant context found."); return None
            logger.info(f"WebChat: Retrieved {len(contexts)} relevant context docs.")
            return contexts
        except Exception as e: logger.error(f"WebChat: Error getting context: {e}", exc_info=True); return None

    def load_resources(self):
        """Load prompts, context, docs, init retriever, handle memory."""
        try:
            logger.info("WebChat: Loading resources...")
            base_prompt = get_latest_system_prompt(self.config.agent_name) or "Assistant."
            logger.info(f"Base prompt loaded ({len(base_prompt)} chars).")
            source_instr = "\n\n## Source Attribution Requirements\n1. ALWAYS specify the exact source file when quoting...\n...(omitted for brevity)..."
            realtime_instr = "\n\nIMPORTANT: Prioritize transcript content from `[REAL-TIME Meeting Transcript Update]` and `[FULL INITIAL TRANSCRIPT]` labels found within the user messages in the conversation history for 'now' or 'current state' queries."
            self.system_prompt = base_prompt + source_instr + realtime_instr

            frameworks = get_latest_frameworks(self.config.agent_name)
            if frameworks: self.system_prompt += "\n\n## Frameworks\n" + frameworks; logger.info("Frameworks loaded.")
            context = get_latest_context(self.config.agent_name, self.config.event_id)
            if context: self.system_prompt += "\n\n## Context\n" + context; logger.info("Context loaded.")
            docs = get_agent_docs(self.config.agent_name)
            if docs: self.system_prompt += "\n\n## Agent Documentation\n" + docs; logger.info("Docs loaded.")

            self.retriever = RetrievalHandler(
                index_name=self.config.index, agent_name=self.config.agent_name,
                session_id=self.config.session_id, event_id=self.config.event_id
            )
            logger.info("Retriever initialized.")
            if self.config.memory is not None: self.reload_memory(); logger.info("Memory loaded.")

            if self.config.listen_transcript:
                self.load_initial_transcript() # This initializes self.transcript_state

            logger.info("WebChat: Resources loaded ok.")
            logger.debug(f"Final system prompt len: {len(self.system_prompt)} chars (excluding transcript).")
        except Exception as e:
            logger.error(f"WebChat: Error loading resources: {e}", exc_info=True)
            if not self.system_prompt: self.system_prompt = "Error loading config."

    def reload_memory(self):
        """Append memory summary using imported function."""
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
                 if "## Previous Chat History" not in self.system_prompt:
                      self.system_prompt += mem_section
                      logger.info(f"Appended memory ({len(summary)} chars).")
                 else:
                     start_index = self.system_prompt.find("## Previous Chat History")
                     if start_index != -1:
                         end_index = self.system_prompt.find("\n\n## ", start_index + 1)
                         if end_index == -1: end_index = len(self.system_prompt)
                         self.system_prompt = self.system_prompt[:start_index] + self.system_prompt[end_index:]
                         self.system_prompt += mem_section
                         logger.info(f"Replaced memory section ({len(summary)} chars).")
                     else:
                         self.system_prompt += mem_section
                         logger.info(f"Appended memory ({len(summary)} chars) - fallback.")
             else: logger.debug("No content for memory.")
        except Exception as e: logger.error(f"Memory reload error: {e}", exc_info=True)

    def setup_routes(self):
        @self.app.route('/')
        def index():
            tmpl = 'index_yggdrasil.html' if self.config.agent_name == 'yggdrasil' else 'index.html'
            logger.debug(f"Rendering template: {tmpl}")
            return render_template(tmpl, agent_name=self.config.agent_name)

        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            try:
                data = request.json
                if not data or 'message' not in data: return jsonify({'error': 'No message'}), 400
                user_msg_content = data['message'] # The original user query
                logger.info(f"WebChat: Received msg: {user_msg_content[:100]}...")

                # --- Prepare for LLM ---
                current_sys_prompt = self.system_prompt
                llm_messages = list(self.chat_history)
                logger.debug(f"Building LLM context from history with {len(llm_messages)} existing messages.")

                retrieved_docs = self.get_document_context(user_msg_content)
                if retrieved_docs:
                     items = [f"[Ctx {i+1} from {d.metadata.get('file_name','?')}({d.metadata.get('score',0):.2f})]:\n{d.page_content}" for i, d in enumerate(retrieved_docs)]
                     context_block = "\n\n---\nRetrieved Context:\n" + "\n\n".join(items)
                     logger.debug(f"Adding retrieved context block ({len(context_block)} chars) to system prompt.")
                     current_sys_prompt += context_block
                else: logger.debug("No retrieved context.")

                # --- Transcript Handling ---
                transcript_content_to_add = ""
                logger.debug(f"Transcript state before processing request: processed_bytes={self.initial_transcript_processed_bytes}, total_bytes={self.initial_transcript_total_bytes}, listening={self.config.listen_transcript}")

                # 1. Check if initial transcript needs to be sent (only happens once)
                if self.config.listen_transcript and self.initial_transcript_content and self.initial_transcript_processed_bytes == 0:
                    label = "[FULL INITIAL TRANSCRIPT]"
                    transcript_content_to_add = f"{label}\n{self.initial_transcript_content}"
                    # Mark as processed immediately by setting processed = total
                    self.initial_transcript_processed_bytes = self.initial_transcript_total_bytes
                    logger.info(f"Adding FULL initial transcript ({len(self.initial_transcript_content)} chars / {self.initial_transcript_total_bytes} bytes). Marked as processed.")

                # 2. If initial transcript is already sent, check for real-time updates
                elif self.config.listen_transcript:
                     tx_update = self.check_transcript_updates() # Uses read_new_transcript_content with state
                     if tx_update:
                         label = "[REAL-TIME Meeting Transcript Update]"
                         transcript_content_to_add = f"{label}\n{tx_update}"
                         logger.info(f"Adding real-time transcript update ({len(tx_update)} chars).")

                # Combine original user message with the transcript content for THIS turn's input
                final_user_content_for_llm = user_msg_content
                if transcript_content_to_add:
                    final_user_content_for_llm += f"\n\n{transcript_content_to_add}"

                # Add THIS turn's combined message to the list being sent to the LLM
                llm_messages.append({'role': 'user', 'content': final_user_content_for_llm})

                # Store the augmented message in persistent history *before* generating response
                self.chat_history.append({'role': 'user', 'content': final_user_content_for_llm})
                logger.debug(f"Appended augmented user message to self.chat_history. History size: {len(self.chat_history)}")


                # --- Call LLM ---
                if not self.client: return jsonify({'error': 'AI client missing'}), 500
                model = self.config.llm_model_name
                max_tokens = self.config.llm_max_output_tokens
                logger.debug(f"WebChat: Using LLM: {model}, MaxTokens: {max_tokens}")

                def generate():
                    response_content = ""; stream_error = None
                    try:
                        logger.debug(f"LLM call. Msgs: {len(llm_messages)}, Final SysPromptLen: {len(current_sys_prompt)}")
                        if logger.isEnabledFor(logging.DEBUG):
                             last_msg = llm_messages[-1]
                             logger.debug(f" -> Last Msg Sent to LLM: Role={last_msg['role']}, Len={len(last_msg['content'])}, Starts='{last_msg['content'][:150]}...'")

                        with self.client.messages.stream(model=model, max_tokens=max_tokens, system=current_sys_prompt, messages=llm_messages) as stream:
                            for text in stream.text_stream:
                                response_content += text
                                yield f"data: {json.dumps({'delta': text})}\n\n"
                        logger.info(f"LLM response ok ({len(response_content)} chars).")
                    except Exception as e:
                        logger.error(f"LLM stream error: {e}", exc_info=True)
                        stream_error = str(e)

                    if stream_error: yield f"data: {json.dumps({'error': f'Error: {stream_error}'})}\n\n"

                    # Add assistant response to persistent history
                    if not stream_error and response_content:
                        self.chat_history.append({'role': 'assistant', 'content': response_content})
                        logger.debug(f"Appended assistant response to self.chat_history. History size: {len(self.chat_history)}")

                    # Archive logic
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

        # Status endpoint
        @self.app.route('/api/status', methods=['GET'])
        def status():
             mem = getattr(self.config, 'memory', None) is not None
             tx = getattr(self.config, 'listen_transcript', False)
             init_tx_processed_flag = bool(self.initial_transcript_content and self.initial_transcript_processed_bytes >= self.initial_transcript_total_bytes and self.initial_transcript_total_bytes > 0)
             processed_percent = 100 if init_tx_processed_flag else 0

             return jsonify({
                 'agent_name': self.config.agent_name,
                 'listen_transcript': tx,
                 'memory_enabled': mem,
                 'initial_transcript_processed': init_tx_processed_flag,
                 'initial_transcript_progress_percent': processed_percent
             })

        # Command endpoint
        @self.app.route('/api/command', methods=['POST'])
        def command():
            data = request.json; cmd = data.get('command','').lower(); msg = f"Cmd: !{cmd}"; code=200; resp={}
            if not cmd: return jsonify({'error': 'No command'}), 400
            logger.info(f"WebChat: Cmd: !{cmd}")
            try:
                if cmd == 'help': msg = "Cmds: !help, !clear, !save, !memory, !listen-transcript"
                elif cmd == 'clear':
                    self.chat_history=[]; self.last_saved_index=0; self.last_archive_index=0
                    self.initial_transcript_content = None
                    self.initial_transcript_total_bytes = 0
                    self.initial_transcript_processed_bytes = 0
                    self.transcript_state = TranscriptState()
                    if self.config.listen_transcript:
                         self.load_initial_transcript() # Reloads and resets state
                         msg='History and transcript state cleared. Initial transcript reloaded.'
                    else:
                         msg='History and transcript state cleared.'
                elif cmd == 'save':
                    msgs = self.chat_history[self.last_saved_index:];
                    if not msgs: msg='Nothing new to save.'
                    else:
                        content_to_save = format_chat_history(msgs)
                        success, fname = save_chat_to_s3(self.config.agent_name, content_to_save, self.config.event_id, True, self.current_chat_file)
                        if success: self.last_saved_index = len(self.chat_history); msg = f'Saved as {fname}'
                        else: msg = 'Error saving.'; code = 500
                elif cmd == 'memory':
                     if self.config.memory is None: self.config.memory = [self.config.agent_name]; self.reload_memory(); msg='Memory ON. System prompt updated.'
                     else: self.config.memory = None; self.load_resources(); msg='Memory OFF. System prompt updated.'
                elif cmd == 'listen-transcript':
                     self.config.listen_transcript = not self.config.listen_transcript
                     status = "ENABLED" if self.config.listen_transcript else "DISABLED"
                     if self.config.listen_transcript:
                          loaded = self.load_initial_transcript() # Loads/reloads and resets state
                          msg = f"Transcript listening {status}." + (" Initial transcript loaded/reloaded." if loaded else " Failed to load initial transcript.")
                     else:
                          self.initial_transcript_content = None
                          self.initial_transcript_total_bytes = 0
                          self.initial_transcript_processed_bytes = 0
                          self.transcript_state = TranscriptState()
                          msg = f"Transcript listening {status}. State cleared."
                else: msg = f"Unknown cmd: !{cmd}"; code = 400
                resp['message'] = msg
                if code == 200:
                    # Recalculate status flags after command execution
                    init_tx_processed_flag = bool(self.initial_transcript_content and self.initial_transcript_processed_bytes >= self.initial_transcript_total_bytes and self.initial_transcript_total_bytes > 0)
                    processed_percent = 100 if init_tx_processed_flag else 0
                    resp['status'] = {
                        'listen_transcript': self.config.listen_transcript,
                        'memory_enabled': self.config.memory is not None,
                        'initial_transcript_processed': init_tx_processed_flag,
                        'initial_transcript_progress_percent': processed_percent
                     }
                else: resp['error'] = msg
                return jsonify(resp), code
            except Exception as e: logger.error(f"Cmd !{cmd} error: {e}", exc_info=True); return jsonify({'error': str(e)}), 500

    def load_initial_transcript(self):
        """Load initial transcript, store locally, AND initialize TranscriptState."""
        self.initial_transcript_content = None
        self.initial_transcript_total_bytes = 0
        self.initial_transcript_processed_bytes = 0 # Reset processed flag
        self.transcript_state = TranscriptState() # Reset S3 read state too
        logger.info("Attempting to load initial transcript and initialize S3 state...")

        s3_client = get_s3_client()
        aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
        if not s3_client or not aws_s3_bucket:
            logger.error("S3 client or bucket not available for initializing transcript state.")
            return False

        try:
            # Load the full content first
            full_content = read_all_transcripts_in_folder(self.config.agent_name, self.config.event_id)
            if full_content and isinstance(full_content, str):
                self.initial_transcript_content = full_content
                self.initial_transcript_total_bytes = len(full_content.encode('utf-8'))
                logger.info(f"Stored initial transcript ({len(full_content)} chars / {self.initial_transcript_total_bytes} bytes).")

                # NOW, find the latest actual S3 file to initialize TranscriptState
                latest_key = get_latest_transcript_file(self.config.agent_name, self.config.event_id, s3_client)
                if latest_key:
                    try:
                        metadata = s3_client.head_object(Bucket=aws_s3_bucket, Key=latest_key)
                        current_s3_size = metadata['ContentLength']
                        current_s3_modified = metadata['LastModified']

                        # Initialize the state to match the end of the file we just loaded
                        self.transcript_state.file_positions[latest_key] = current_s3_size
                        self.transcript_state.last_modified[latest_key] = current_s3_modified
                        self.transcript_state.current_latest_key = latest_key
                        logger.info(f"Initialized TranscriptState for S3 key '{latest_key}' (Size: {current_s3_size}, Mod: {current_s3_modified}).")
                        return True # Successful load and state initialization

                    except Exception as head_e:
                        logger.error(f"Failed to get S3 metadata for key '{latest_key}' to initialize state: {head_e}")
                        # Proceeding without state initialization, real-time updates might be incorrect
                        return True # Content was loaded, but state init failed
                else:
                    logger.warning("Could not find latest S3 transcript key to initialize state after loading content.")
                    # Proceeding without state initialization
                    return True # Content was loaded, but state init failed
            elif not full_content:
                logger.warning("No initial transcript content found or read.")
                return False
            else: # Not a string
                logger.error("Loaded transcript content is not a string.")
                self.initial_transcript_content = None
                return False

        except Exception as e:
            logger.error(f"Error loading all transcripts or initializing state: {e}", exc_info=True)
            return False


    # check_transcript_updates checks S3 for new bytes AFTER initial load is sent
    def check_transcript_updates(self) -> Optional[str]:
        """Check for new transcript content *after* initial load is processed/sent."""
        if not self.config.listen_transcript: return None
        # Check if initial load exists AND has been marked as processed/sent
        initial_load_sent = bool(self.initial_transcript_content and self.initial_transcript_processed_bytes >= self.initial_transcript_total_bytes)
        if not initial_load_sent:
            return None

        try:
            # Uses TranscriptState which should now be initialized correctly
            new_content = read_new_transcript_content(self.transcript_state, self.config.agent_name, self.config.event_id, False)
            if new_content:
                # Log when new content IS detected by this check
                logger.info(f"Real-time check found new transcript content via S3 state ({len(new_content)} chars).")
                return new_content
            # No need to log every time no new content is found, reduces noise
            # else:
            #     logger.debug("Real-time check: No new content detected via S3 state.")
            return None
        except Exception as e:
            logger.error(f"Real-time transcript check error: {e}", exc_info=True)
            return None

    # run method remains the same
    def run(self, host: str = '127.0.0.1', port: int = 5001, debug: bool = False):
        """Run the Flask web server."""
        def transcript_update_loop():
             logger.info("Starting background transcript state update loop.")
             while True:
                  time.sleep(5)
                  if self.config.listen_transcript:
                      # This call primarily updates self.transcript_state for read_new_transcript_content
                      self.check_transcript_updates()

        if self.config.interface_mode != 'cli':
            update_thread = threading.Thread(target=transcript_update_loop, daemon=True)
            update_thread.start()

        logger.info(f"Starting Flask server on {host}:{port}, Debug: {debug}")
        try:
            if not debug:
                 from waitress import serve
                 logger.info("Running with Waitress production server.")
                 serve(self.app, host=host, port=port)
            else:
                 logger.warning("Running with Flask development server (debug=True).")
                 self.app.run(host=host, port=port, debug=debug, use_reloader=False)
        except ImportError:
             logger.warning("Waitress not found. Falling back to Flask development server.")
             self.app.run(host=host, port=port, debug=debug, use_reloader=False)
        except Exception as e:
            logger.critical(f"Flask server failed: {e}", exc_info=True)