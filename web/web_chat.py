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
from utils.transcript_utils import TranscriptState, read_new_transcript_content, read_all_transcripts_in_folder
from utils.s3_utils import (
    get_latest_system_prompt, get_latest_frameworks, get_latest_context,
    get_agent_docs, load_existing_chats_from_s3, save_chat_to_s3, format_chat_history
)
from anthropic import Anthropic

logger = logging.getLogger(__name__)

# Constants
TRANSCRIPT_CHUNK_SIZE_BYTES = 4000 # Approx 1000 tokens

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
        self.transcript_state = TranscriptState() # For tracking *new* bytes after initial load
        self.scheduler_thread = None
        self.last_saved_index = 0
        self.last_archive_index = 0
        self.current_chat_file = None

        # State for incremental transcript processing
        self.initial_transcript_content: Optional[str] = None
        self.initial_transcript_total_bytes: int = 0
        self.initial_transcript_processed_bytes: int = 0
        self.transcript_chunk_size: int = TRANSCRIPT_CHUNK_SIZE_BYTES

        try:
             if not hasattr(config, 'session_id') or not config.session_id: config.session_id = datetime.now().strftime('%Y%m%d-T%H%M%S')
             event_id = config.event_id or '0000'
             self.current_chat_file = f"chat_D{config.session_id}_aID-{config.agent_name}_eID-{event_id}.txt"
             logger.info(f"WebChat: Session {config.session_id}, Chat file: {self.current_chat_file}")
             self.load_resources() # This will call load_initial_transcript if needed
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
            # Ensure instructions mention the new chunk label as well
            source_instr = "\n\n## Source Attribution Requirements\n1. ALWAYS specify the exact source file when quoting...\n...(omitted for brevity)..."
            realtime_instr = "\n\nIMPORTANT: Prioritize `[REAL-TIME Meeting Transcript Update]` and `[INITIAL TRANSCRIPT CHUNK ...]` content for 'now' or 'current state' queries."
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

            # Load initial transcript content *into state*, not system prompt
            if self.config.listen_transcript:
                self.load_initial_transcript() # This now stores content locally

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
                 # Check if memory section exists before adding
                 if "## Previous Chat History" not in self.system_prompt:
                      self.system_prompt += mem_section
                      logger.info(f"Appended memory ({len(summary)} chars).")
                 else:
                     # Replace existing memory section if reloading is intended
                     start_index = self.system_prompt.find("## Previous Chat History")
                     if start_index != -1:
                         end_index = self.system_prompt.find("\n\n## ", start_index + 1)
                         if end_index == -1: end_index = len(self.system_prompt)
                         self.system_prompt = self.system_prompt[:start_index] + self.system_prompt[end_index:]
                         self.system_prompt += mem_section # Append new memory
                         logger.info(f"Replaced memory section ({len(summary)} chars).")
                     else: # Fallback
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
                user_msg_content = data['message']
                logger.info(f"WebChat: Received msg: {user_msg_content[:100]}...")

                # --- Prepare for LLM ---
                current_sys_prompt = self.system_prompt # Base system prompt (no transcript)
                llm_messages = [m for m in self.chat_history if m.get('role') in ['user', 'assistant']] # History

                # Add retrieved document context to system prompt string
                retrieved_docs = self.get_document_context(user_msg_content)
                if retrieved_docs:
                     items = [f"[Ctx {i+1} from {d.metadata.get('file_name','?')}({d.metadata.get('score',0):.2f})]:\n{d.page_content}" for i, d in enumerate(retrieved_docs)]
                     context_block = "\n\n---\nRetrieved Context:\n" + "\n\n".join(items)
                     logger.debug(f"Adding retrieved context block ({len(context_block)} chars) to system prompt.")
                     current_sys_prompt += context_block
                else: logger.debug("No retrieved context.")

                # --- Transcript Handling ---
                transcript_chunk_to_add = ""
                # ADDED LOGGING HERE: Verify state persistence
                logger.debug(f"Transcript state before processing request: processed={self.initial_transcript_processed_bytes}, total={self.initial_transcript_total_bytes}, listening={self.config.listen_transcript}")

                # 1. Check if initial transcript needs processing
                if self.config.listen_transcript and self.initial_transcript_content and self.initial_transcript_processed_bytes < self.initial_transcript_total_bytes:
                    start_byte = self.initial_transcript_processed_bytes
                    end_byte = min(start_byte + self.transcript_chunk_size, self.initial_transcript_total_bytes)
                    # Slice the encoded bytes
                    chunk_content_bytes = self.initial_transcript_content.encode('utf-8')[start_byte:end_byte]
                    # Decode the sliced bytes safely
                    chunk_content = chunk_content_bytes.decode('utf-8', errors='ignore')

                    # Ensure we didn't get an empty string after decoding potentially broken characters at the boundary
                    if chunk_content:
                        total_chunks = math.ceil(self.initial_transcript_total_bytes / self.transcript_chunk_size)
                        current_chunk_num = (start_byte // self.transcript_chunk_size) + 1
                        label = f"[INITIAL TRANSCRIPT CHUNK {current_chunk_num}/{total_chunks}]"
                        transcript_chunk_to_add = f"{label}\n{chunk_content}"
                        # Update processed bytes based on the *actual bytes decoded successfully*
                        # This is tricky if errors='ignore' was used. Re-encoding the result is the safest way.
                        actual_bytes_in_chunk = len(chunk_content.encode('utf-8'))
                        # Advance pointer by the intended chunk size OR total if it's the last chunk
                        self.initial_transcript_processed_bytes = end_byte
                        logger.info(f"Adding initial transcript chunk {current_chunk_num}/{total_chunks} ({len(chunk_content)} chars / {actual_bytes_in_chunk} decoded bytes from {len(chunk_content_bytes)} byte slice). New processed count: {self.initial_transcript_processed_bytes}/{self.initial_transcript_total_bytes} bytes.")
                    else:
                         logger.warning(f"Calculated initial transcript chunk (bytes {start_byte}-{end_byte}) was empty or undecodable, advancing tracker by chunk size.")
                         self.initial_transcript_processed_bytes = end_byte # Still advance

                # 2. If initial transcript is done, check for real-time updates
                elif self.config.listen_transcript:
                    tx_update = self.check_transcript_updates()
                    if tx_update:
                        label = "[REAL-TIME Meeting Transcript Update]"
                        transcript_chunk_to_add = f"{label}\n{tx_update}"
                        logger.info(f"Adding real-time transcript update ({len(tx_update)} chars).")

                # Combine user message with the determined transcript chunk/update
                final_user_content = user_msg_content
                if transcript_chunk_to_add:
                    final_user_content += f"\n\n{transcript_chunk_to_add}"

                # Add the combined message to the list for the LLM
                llm_messages.append({'role': 'user', 'content': final_user_content})

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
                             logger.debug(f" -> Last Msg Role: {last_msg['role']}, Len: {len(last_msg['content'])}, Starts: '{last_msg['content'][:150]}...'")

                        with self.client.messages.stream(model=model, max_tokens=max_tokens, system=current_sys_prompt, messages=llm_messages) as stream:
                            for text in stream.text_stream:
                                response_content += text
                                yield f"data: {json.dumps({'delta': text})}\n\n"
                        logger.info(f"LLM response ok ({len(response_content)} chars).")
                    except Exception as e:
                        logger.error(f"LLM stream error: {e}", exc_info=True)
                        stream_error = str(e)

                    if stream_error: yield f"data: {json.dumps({'error': f'Error: {stream_error}'})}\n\n"

                    # Add the original user message and assistant response to persistent history
                    self.chat_history.append({'role': 'user', 'content': user_msg_content})
                    if not stream_error and response_content:
                        self.chat_history.append({'role': 'assistant', 'content': response_content})

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
             # Check if content exists and if processed bytes >= total bytes
             init_tx_done = not self.initial_transcript_content or self.initial_transcript_processed_bytes >= self.initial_transcript_total_bytes
             processed_percent = 0
             if self.initial_transcript_content and self.initial_transcript_total_bytes > 0:
                 processed_percent = round((self.initial_transcript_processed_bytes / self.initial_transcript_total_bytes) * 100)

             return jsonify({
                 'agent_name': self.config.agent_name,
                 'listen_transcript': tx,
                 'memory_enabled': mem,
                 'initial_transcript_processed': init_tx_done,
                 'initial_transcript_progress_percent': processed_percent if tx and self.initial_transcript_content else 0 # Show progress only if listening and content exists
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
                    # Reset transcript state fully
                    self.initial_transcript_content = None
                    self.initial_transcript_total_bytes = 0
                    self.initial_transcript_processed_bytes = 0
                    self.transcript_state = TranscriptState()
                    # Optionally reload initial transcript if listening is still enabled
                    if self.config.listen_transcript:
                         self.load_initial_transcript()
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
                     else: self.config.memory = None; self.load_resources(); msg='Memory OFF. System prompt updated.' # Reload resources to ensure memory removed
                elif cmd == 'listen-transcript':
                     self.config.listen_transcript = not self.config.listen_transcript
                     status = "ENABLED" if self.config.listen_transcript else "DISABLED"
                     if self.config.listen_transcript:
                          loaded = self.load_initial_transcript() # Load/reload initial transcript
                          msg = f"Transcript listening {status}." + (" Initial transcript loaded/reloaded." if loaded else " Failed to load initial transcript.")
                     else:
                          # Clear transcript state when turning off
                          self.initial_transcript_content = None
                          self.initial_transcript_total_bytes = 0
                          self.initial_transcript_processed_bytes = 0
                          self.transcript_state = TranscriptState()
                          msg = f"Transcript listening {status}. State cleared."
                else: msg = f"Unknown cmd: !{cmd}"; code = 400
                resp['message'] = msg
                if code == 200:
                    # Recalculate status flags after command execution
                    init_tx_done = not self.initial_transcript_content or self.initial_transcript_processed_bytes >= self.initial_transcript_total_bytes
                    processed_percent = 0
                    if self.initial_transcript_content and self.initial_transcript_total_bytes > 0:
                         processed_percent = round((self.initial_transcript_processed_bytes / self.initial_transcript_total_bytes) * 100)
                    resp['status'] = {
                        'listen_transcript': self.config.listen_transcript,
                        'memory_enabled': self.config.memory is not None,
                        'initial_transcript_processed': init_tx_done,
                        'initial_transcript_progress_percent': processed_percent if self.config.listen_transcript and self.initial_transcript_content else 0
                     }
                else: resp['error'] = msg
                return jsonify(resp), code
            except Exception as e: logger.error(f"Cmd !{cmd} error: {e}", exc_info=True); return jsonify({'error': str(e)}), 500

    def load_initial_transcript(self):
        """Load initial transcript content into state, don't add to system prompt."""
        self.initial_transcript_content = None
        self.initial_transcript_total_bytes = 0
        self.initial_transcript_processed_bytes = 0
        self.transcript_state = TranscriptState() # Reset S3 read state too
        logger.info("Attempting to load initial transcript...")
        try:
            full_content = read_all_transcripts_in_folder(self.config.agent_name, self.config.event_id)
            if full_content:
                # Ensure content is treated as string before encoding
                if isinstance(full_content, str):
                    self.initial_transcript_content = full_content
                    self.initial_transcript_total_bytes = len(full_content.encode('utf-8'))
                    logger.info(f"Stored initial transcript ({len(full_content)} chars / {self.initial_transcript_total_bytes} bytes). Ready for chunking.")
                    return True
                else:
                    logger.error("Loaded transcript content is not a string.")
                    self.initial_transcript_content = None # Ensure it's None if not string
                    return False
            else:
                logger.warning("No initial transcript content found or read.")
                return False
        except Exception as e:
            logger.error(f"Error loading all transcripts: {e}", exc_info=True)
            return False

    def check_transcript_updates(self) -> Optional[str]:
        """Check for new transcript content *after* initial load is processed."""
        if not self.config.listen_transcript: return None
        # Check if content exists and if processed bytes >= total bytes
        initial_done = not self.initial_transcript_content or self.initial_transcript_processed_bytes >= self.initial_transcript_total_bytes
        if not initial_done:
            # logger.debug("Initial transcript processing not complete, skipping real-time check.") # Reduce log noise
            return None

        try:
            new_content = read_new_transcript_content(self.transcript_state, self.config.agent_name, self.config.event_id, False)
            if new_content:
                # logger.debug(f"Real-time check found new transcript content ({len(new_content)} chars).") # Reduce log noise
                return new_content
            return None
        except Exception as e:
            logger.error(f"Real-time transcript check error: {e}", exc_info=True)
            return None

    def run(self, host: str = '127.0.0.1', port: int = 5001, debug: bool = False):
        """Run the Flask web server."""
        def transcript_update_loop():
             logger.info("Starting background transcript state update loop.")
             while True:
                  time.sleep(5) # Check every 5 seconds
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