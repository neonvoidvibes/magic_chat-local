from flask import Flask, request, jsonify, render_template, current_app, Response
from typing import Optional, List, Dict, Any
import threading
import os
import logging
from datetime import datetime, time as dt_time # Keep dt_time import if used elsewhere
import json
import time
import traceback

from config import AppConfig
from utils.retrieval_handler import RetrievalHandler
from utils.transcript_utils import TranscriptState, read_new_transcript_content, read_all_transcripts_in_folder, get_latest_transcript_file
# Corrected import path for S3 utils
from utils.s3_utils import (
    get_latest_system_prompt, get_latest_frameworks, get_latest_context,
    get_agent_docs, load_existing_chats_from_s3, save_chat_to_s3, format_chat_history,
    get_s3_client # Import get_s3_client if needed
)
from anthropic import Anthropic

logger = logging.getLogger(__name__)

class WebChat:
    def __init__(self, config: AppConfig):
        self.config = config
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        self.app.config['SECRET_KEY'] = os.urandom(24)
        self.setup_routes()
        self.chat_history = []
        self.client = None # Anthropic client
        self.system_prompt = "Default system prompt."
        self.retriever = None
        self.transcript_state = TranscriptState()
        self.last_saved_index = 0
        self.last_archive_index = 0
        self.current_chat_file = None
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

             # Initialize Anthropic client FIRST
             self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
             logger.info("WebChat: Anthropic client initialized.")

             # Now load resources, which initializes Retriever and passes the client
             self.load_resources()

        except Exception as e: logger.error(f"WebChat: Init error: {e}", exc_info=True); raise RuntimeError("WebChat initialization failed") from e

    def get_document_context(self, query: str) -> Optional[List[Any]]:
        """Get relevant document context using the RetrievalHandler."""
        if not self.retriever: logger.error("WebChat: Retriever missing."); return None
        try:
            logger.debug(f"WebChat: Getting context for query: {query[:100]}...")
            is_transcript = any(word in query.lower() for word in ['transcript', 'conversation', 'meeting', 'session', 'said'])
            # Query transformation happens inside get_relevant_context now
            contexts = self.retriever.get_relevant_context(query=query, top_k=10, is_transcript=is_transcript) # Use increased top_k default or allow override
            if not contexts: logger.info("WebChat: No relevant context found."); return None
            logger.info(f"WebChat: Retrieved {len(contexts)} relevant context docs.")
            return contexts
        except Exception as e: logger.error(f"WebChat: Error getting context: {e}", exc_info=True); return None

    def load_resources(self):
        """Load prompts, context, docs, init retriever, handle memory."""
        # Requires self.client to be initialized *before* calling this if retriever needs it
        if not self.client:
             logger.error("WebChat load_resources: Anthropic client not initialized yet.")
             # Handle error appropriately, maybe raise or set default prompt
             self.system_prompt = "Error: Client not ready."
             return

        try:
            logger.info("WebChat: Loading resources...")
            base_prompt = get_latest_system_prompt(self.config.agent_name) or "Assistant."
            logger.info(f"Base prompt loaded ({len(base_prompt)} chars).")
            source_instr = "\n\n## Source Attribution Requirements\n1. ALWAYS specify the exact source file..." # Shortened
            realtime_instr = "\n\nIMPORTANT: Prioritize [REAL-TIME...] updates for 'now' queries." # Shortened
            self.system_prompt = base_prompt + source_instr + realtime_instr

            frameworks = get_latest_frameworks(self.config.agent_name); context = get_latest_context(self.config.agent_name, self.config.event_id); docs = get_agent_docs(self.config.agent_name)
            if frameworks: self.system_prompt += "\n\n## Frameworks\n" + frameworks; logger.info("Frameworks loaded.")
            if context: self.system_prompt += "\n\n## Context\n" + context; logger.info("Context loaded.")
            if docs: self.system_prompt += "\n\n## Agent Documentation\n" + docs; logger.info("Docs loaded.")

            # Initialize Retriever, passing the client instance
            self.retriever = RetrievalHandler(
                index_name=self.config.index, agent_name=self.config.agent_name,
                session_id=self.config.session_id, event_id=self.config.event_id,
                anthropic_client=self.client # Pass client here
            )
            logger.info("Retriever initialized.")

            if self.config.memory is not None: self.reload_memory(); logger.info("Memory loaded.")
            if self.config.listen_transcript: self.load_initial_transcript()
            logger.info("WebChat: Resources loaded ok.")
            logger.debug(f"Final system prompt len: {len(self.system_prompt)} chars.")
        except Exception as e:
            logger.error(f"WebChat: Error loading resources: {e}", exc_info=True)
            if not self.system_prompt or self.system_prompt == "Default system prompt.": self.system_prompt = "Error loading configuration."


    def reload_memory(self):
        """Append memory summary using imported function."""
        logger.debug("WebChat: Reloading memory...")
        # (Keep existing implementation - relies on imported s3_utils functions)
        try:
             agents_to_load = self.config.memory if self.config.memory else [self.config.agent_name]
             previous_chats = load_existing_chats_from_s3(self.config.agent_name, agents_to_load)
             if not previous_chats: logger.debug("No memory files found."); return
             all_items = [f"{msg.get('role','?').capitalize()} (File:{os.path.basename(c.get('file','?'))}): {msg.get('content','')}" for c in previous_chats for msg in c.get('messages',[]) if msg.get('content')]
             combined = "\n\n---\n\n".join(all_items); max_len = 10000
             summary = combined[:max_len] + ("..." if len(combined) > max_len else "")
             if summary:
                 mem_section = "\n\n## Previous Chat History (Memory)\n" + summary
                 # Replace existing memory section if present, otherwise append
                 start_index = self.system_prompt.find("## Previous Chat History")
                 if start_index != -1:
                     end_index = self.system_prompt.find("\n\n## ", start_index + 1)
                     if end_index == -1: end_index = len(self.system_prompt)
                     self.system_prompt = self.system_prompt[:start_index] + self.system_prompt[end_index:]
                     logger.info(f"Replacing memory section.")
                 self.system_prompt += mem_section
                 logger.info(f"Appended/Replaced memory summary ({len(summary)} chars).")

             else: logger.debug("No content for memory.")
        except Exception as e: logger.error(f"Memory reload error: {e}", exc_info=True)


    def setup_routes(self):
        @self.app.route('/')
        def index():
            tmpl = 'index.html' # Defaulting to index.html for now
            logger.debug(f"Rendering template: {tmpl}")
            return render_template(tmpl, agent_name=self.config.agent_name)

        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            # (Keep existing chat endpoint logic, ensuring it uses self.client and self.config correctly)
            try:
                data = request.json
                if not data or 'message' not in data: return jsonify({'error': 'No message'}), 400
                user_msg_content = data['message']
                logger.info(f"WebChat: Received msg: {user_msg_content[:100]}...")

                current_sys_prompt = self.system_prompt
                # Prepare messages list for LLM (user/assistant turns only initially)
                llm_messages = [m for m in self.chat_history if m.get('role') in ['user', 'assistant']]
                logger.debug(f"Building LLM context from history with {len(llm_messages)} existing messages.")

                # Get retrieved context and add to system prompt for this turn
                retrieved_docs = self.get_document_context(user_msg_content)
                if retrieved_docs:
                     items = [f"[Ctx {i+1} from {d.metadata.get('file_name','?')}({d.metadata.get('score',0):.2f})]:\n{d.page_content}" for i, d in enumerate(retrieved_docs)]
                     context_block = "\n\n---\nRetrieved Context:\n" + "\n\n".join(items)
                     logger.debug(f"Adding retrieved context block ({len(context_block)} chars) to system prompt.")
                     current_sys_prompt += context_block
                else: logger.debug("No retrieved context.")

                # Handle transcript updates (initial or pending)
                transcript_content_to_add = ""
                final_user_content_for_llm = user_msg_content # Start with original query

                if self.config.listen_transcript:
                    if not self.initial_transcript_sent:
                        if self.initial_transcript_content:
                            label = "[FULL INITIAL TRANSCRIPT]"; transcript_content_to_add = f"{label}\n{self.initial_transcript_content}"
                            self.initial_transcript_sent = True; logger.info(f"Adding FULL initial transcript ({len(self.initial_transcript_content)} chars).")
                        else: logger.warning("Tx listening on, but no initial tx content."); self.initial_transcript_sent = True # Mark as processed anyway
                    else: # Initial already sent, check pending
                        with self.transcript_lock:
                            if self.pending_transcript_update:
                                label = "[REAL-TIME Meeting Transcript Update]"; transcript_content_to_add = f"{label}\n{self.pending_transcript_update}"
                                logger.info(f"Adding PENDING real-time transcript ({len(self.pending_transcript_update)} chars).")
                                self.pending_transcript_update = None # Clear pending
                            # else: logger.debug("No pending tx update.") # Can be noisy

                # Add transcript content (if any) and user message to LLM messages list
                if transcript_content_to_add:
                     llm_messages.append({'role': 'user', 'content': transcript_content_to_add}) # Add transcript first? Or after user msg? Let's try before.
                llm_messages.append({'role': 'user', 'content': user_msg_content}) # Add user message last

                # Add messages to persistent history (transcript separate, user original)
                if transcript_content_to_add: self.chat_history.append({'role': 'user', 'content': transcript_content_to_add})
                self.chat_history.append({'role': 'user', 'content': user_msg_content})
                logger.debug(f"Appended user message(s) to self.chat_history. History size: {len(self.chat_history)}")


                if not self.client: return jsonify({'error': 'AI client missing'}), 500
                model = self.config.llm_model_name; max_tokens = self.config.llm_max_output_tokens
                logger.debug(f"WebChat: Using LLM: {model}, MaxTokens: {max_tokens}")

                def generate():
                    response_content = ""; stream_error = None
                    try:
                        logger.debug(f"LLM call. Msgs: {len(llm_messages)}, Final SysPromptLen: {len(current_sys_prompt)}")
                        if logger.isEnabledFor(logging.DEBUG): last_msg = llm_messages[-1]; logger.debug(f" -> Last Msg: Role={last_msg['role']}, Len={len(last_msg['content'])}, Starts='{last_msg['content'][:150]}...'")
                        with self.client.messages.stream(model=model, max_tokens=max_tokens, system=current_sys_prompt, messages=llm_messages) as stream:
                            for text in stream.text_stream: response_content += text; yield f"data: {json.dumps({'delta': text})}\n\n"
                        logger.info(f"LLM response ok ({len(response_content)} chars).")
                    except Exception as e: logger.error(f"LLM stream error: {e}", exc_info=True); stream_error = str(e)
                    if stream_error: yield f"data: {json.dumps({'error': f'Error: {stream_error}'})}\n\n"

                    if not stream_error and response_content: self.chat_history.append({'role': 'assistant', 'content': response_content}); logger.debug(f"Appended assistant response. History size: {len(self.chat_history)}")

                    archive_msgs = self.chat_history[self.last_archive_index:]
                    if archive_msgs:
                        content_to_save = format_chat_history(archive_msgs)
                        if content_to_save: success, _ = save_chat_to_s3(self.config.agent_name, content_to_save, self.config.event_id, False, self.current_chat_file)
                        if success: self.last_archive_index = len(self.chat_history); logger.debug("Auto-archived.")
                        else: logger.error("Failed auto-archive.")
                    yield f"data: {json.dumps({'done': True})}\n\n"

                return Response(generate(), mimetype='text/event-stream')
            except Exception as e: logger.error(f"/api/chat error: {e}", exc_info=True); return jsonify({'error': 'Server error'}), 500

        # Status endpoint
        @self.app.route('/api/status', methods=['GET'])
        def status():
            mem=getattr(self.config,'memory',None) is not None; tx=getattr(self.config,'listen_transcript',False)
            init_tx_proc=self.initial_transcript_sent; proc_perc=100 if init_tx_proc else 0
            with self.transcript_lock: has_pending = self.pending_transcript_update is not None
            return jsonify({'agent_name': self.config.agent_name, 'listen_transcript': tx, 'memory_enabled': mem,
                            'initial_transcript_processed': init_tx_proc, 'initial_transcript_progress_percent': proc_perc,
                            'has_pending_transcript_update': has_pending})

        # Command endpoint
        @self.app.route('/api/command', methods=['POST'])
        def command():
            # (Keep existing command logic)
            data = request.json; cmd = data.get('command','').lower(); msg = f"Cmd: !{cmd}"; code=200; resp={}
            if not cmd: return jsonify({'error': 'No command'}), 400
            logger.info(f"WebChat: Cmd: !{cmd}")
            try:
                if cmd == 'help': msg = "Cmds: !help, !clear, !save, !memory, !listen-transcript"
                elif cmd == 'clear':
                    self.chat_history=[]; self.last_saved_index=0; self.last_archive_index=0
                    self.initial_transcript_content=None; self.initial_transcript_sent=False; self.transcript_state = TranscriptState()
                    with self.transcript_lock: self.pending_transcript_update = None
                    if self.config.listen_transcript: self.load_initial_transcript(); msg='History/Tx state cleared. Initial tx reloaded.'
                    else: msg='History/Tx state cleared.'
                elif cmd == 'save':
                    msgs = self.chat_history[self.last_saved_index:];
                    if not msgs: msg='Nothing new.'
                    else: content = format_chat_history(msgs); success, fname = save_chat_to_s3(self.config.agent_name, content, self.config.event_id, True, self.current_chat_file);
                    if success: self.last_saved_index = len(self.chat_history); msg = f'Saved as {fname}'
                    else: msg = 'Error saving.'; code = 500
                elif cmd == 'memory':
                     if self.config.memory is None: self.config.memory = [self.config.agent_name]; self.reload_memory(); msg='Memory ON.'
                     else: self.config.memory = None; self.load_resources(); msg='Memory OFF.'
                elif cmd == 'listen-transcript':
                     self.config.listen_transcript = not self.config.listen_transcript; status = "ENABLED" if self.config.listen_transcript else "DISABLED"
                     if self.config.listen_transcript: loaded = self.load_initial_transcript(); msg = f"Tx listening {status}." + (" Initial tx loaded." if loaded else "")
                     else: self.initial_transcript_content=None; self.initial_transcript_sent=False; self.transcript_state=TranscriptState(); msg = f"Tx listening {status}."
                else: msg = f"Unknown cmd: !{cmd}"; code = 400
                resp['message'] = msg
                if code == 200:
                    init_tx_proc=self.initial_transcript_sent; proc_perc=100 if init_tx_proc else 0; with self.transcript_lock: has_pending=self.pending_transcript_update is not None
                    resp['status'] = {'listen_transcript': self.config.listen_transcript, 'memory_enabled': self.config.memory is not None,
                                      'initial_transcript_processed': init_tx_proc, 'initial_transcript_progress_percent': proc_perc,
                                      'has_pending_transcript_update': has_pending}
                else: resp['error'] = msg
                return jsonify(resp), code
            except Exception as e: logger.error(f"Cmd !{cmd} error: {e}", exc_info=True); return jsonify({'error': str(e)}), 500


    # load_initial_transcript and check_transcript_updates remain the same
    def load_initial_transcript(self):
        """Load initial transcript content."""
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
        """Check for new transcript content and APPEND it to pending_transcript_update."""
        if not self.config.listen_transcript: return
        if not self.initial_transcript_sent: return # Don't check S3 if initial load hasn't been sent/processed
        try:
            new_content = read_new_transcript_content(self.transcript_state, self.config.agent_name, self.config.event_id, False)
            if new_content:
                 with self.transcript_lock:
                      if self.pending_transcript_update is None: self.pending_transcript_update = new_content; logger.info(f"BG check INITIALIZED pending tx ({len(new_content)} chars).")
                      else: self.pending_transcript_update += "\n" + new_content; logger.info(f"BG check APPENDED to pending tx ({len(new_content)} chars). Total: {len(self.pending_transcript_update)} chars.")
        except Exception as e: logger.error(f"BG tx check error: {e}", exc_info=True)

    # run method with background thread remains the same
    def run(self, host: str = '127.0.0.1', port: int = 5001, debug: bool = False):
        """Run the Flask web server and background transcript checker."""
        def transcript_update_loop():
             logger.info("Starting background transcript state update loop.")
             while True: time.sleep(5); self.check_transcript_updates()
        if self.config.interface_mode != 'cli':
            update_thread = threading.Thread(target=transcript_update_loop, daemon=True); update_thread.start()
        logger.info(f"Starting Flask server on {host}:{port}, Debug: {debug}")
        try:
            # Removed waitress for simplicity during debugging
            # if not debug: from waitress import serve; logger.info("Running Waitress"); serve(self.app, host=host, port=port)
            # else: logger.warning("Running Flask dev server"); self.app.run(host=host, port=port, debug=debug, use_reloader=False)
            self.app.run(host=host, port=port, debug=debug, use_reloader=False) # Always use dev server for now
        except Exception as e: logger.critical(f"Flask server failed: {e}", exc_info=True)