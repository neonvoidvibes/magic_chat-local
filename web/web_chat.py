from flask import Flask, request, jsonify, render_template, current_app, Response
from typing import Optional, List, Dict, Any
import threading
import os
import logging
from datetime import datetime
import json
import time
import traceback

from config import AppConfig
from utils.retrieval_handler import RetrievalHandler
from utils.transcript_utils import TranscriptState, read_new_transcript_content, read_all_transcripts_in_folder
from utils.s3_utils import (
    get_latest_system_prompt, get_latest_frameworks, get_latest_context,
    get_agent_docs, load_existing_chats_from_s3, save_chat_to_s3, format_chat_history
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
        self.client = None
        self.system_prompt = "Default system prompt."
        self.retriever = None
        self.transcript_state = TranscriptState()
        self.scheduler_thread = None
        self.last_saved_index = 0
        self.last_archive_index = 0
        self.current_chat_file = None

        try:
             if not hasattr(config, 'session_id') or not config.session_id: config.session_id = datetime.now().strftime('%Y%m%d-T%H%M%S')
             event_id = config.event_id or '0000'
             self.current_chat_file = f"chat_D{config.session_id}_aID-{config.agent_name}_eID-{event_id}.txt"
             logger.info(f"WebChat: Session {config.session_id}, Chat file: {self.current_chat_file}")
             self.load_resources()
             self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
             logger.info("WebChat: Anthropic client initialized.")
        except Exception as e: logger.error(f"WebChat: Init error: {e}", exc_info=True); raise RuntimeError("WebChat initialization failed") from e

    def get_document_context(self, query: str) -> Optional[List[Any]]: # Return type from retriever is List[Document] now
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
            # Added realtime instruction to base prompt load
            source_instr = "\n\n## Source Attribution Requirements\n..." # Keep existing instructions short for log
            realtime_instr = "\n\nIMPORTANT: When asked about what is happening 'now' or for live updates, prioritize information marked with [REAL-TIME Meeting Transcript Update] over historical context from the vector database."
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
            if self.config.listen_transcript: self.load_initial_transcript()
            logger.info("WebChat: Resources loaded ok.")
            logger.debug(f"Final system prompt len: {len(self.system_prompt)} chars.")
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
                 if "## Previous Chat History" not in self.system_prompt: self.system_prompt += mem_section; logger.info(f"Appended memory ({len(summary)} chars).")
                 else: logger.warning("Memory section exists, skipping.")
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
                user_msg = data['message']
                logger.info(f"WebChat: Received msg: {user_msg[:100]}...")

                # Prepare messages for LLM
                # Start with a copy of persistent history *excluding* last assistant response if any
                llm_messages = [m for m in self.chat_history if m.get('role') != 'assistant']

                # Get retrieved context
                retrieved_docs = self.get_document_context(user_msg)
                context_block = ""
                if retrieved_docs:
                     items = [f"[Ctx {i+1} from {d.metadata.get('file_name','?')}({d.metadata.get('score',0):.2f})]:\n{d.page_content}" for i, d in enumerate(retrieved_docs)]
                     context_block = "\n\n---\nRetrieved Context:\n" + "\n\n".join(items)
                     logger.debug(f"Adding context block ({len(context_block)} chars).")
                     # Add context as a system message before the user query
                     llm_messages.append({"role": "system", "content": context_block}) # Use system role for context block
                else: logger.debug("No context retrieved.")

                # Add current user message
                llm_messages.append({'role': 'user', 'content': user_msg})

                # Check for and add transcript update *after* user message
                tx_chunk = self.check_transcript_updates()
                if tx_chunk:
                     label = "[REAL-TIME Meeting Transcript Update]"
                     logger.debug(f"Adding tx chunk ({len(tx_chunk)} chars).")
                     llm_messages.append({"role": "user", "content": f"{label}\n{tx_chunk}"})

                if not self.client: return jsonify({'error': 'AI client missing'}), 500

                model = self.config.llm_model_name
                max_tokens = self.config.llm_max_output_tokens
                logger.debug(f"WebChat: Using LLM: {model}, MaxTokens: {max_tokens}")

                def generate():
                    response = ""; stream_error = None
                    try:
                        # Correct variable name here for logging
                        logger.debug(f"LLM call. Msgs: {len(llm_messages)}, SysPromptLen: {len(self.system_prompt)}")
                        with self.client.messages.stream(model=model, max_tokens=max_tokens, system=self.system_prompt, messages=llm_messages) as stream:
                            for text in stream.text_stream: response += text; yield f"data: {json.dumps({'delta': text})}\n\n"
                        logger.info(f"LLM response ok ({len(response)} chars).")
                    except Exception as e: logger.error(f"LLM stream error: {e}", exc_info=True); stream_error = str(e)
                    if stream_error: yield f"data: {json.dumps({'error': f'Error: {stream_error}'})}\n\n"

                    # Add actual user message and assistant response to persistent history
                    # Ensure user message isn't added twice if it was already part of llm_messages preparation
                    if not any(m['role'] == 'user' and m['content'] == user_msg for m in self.chat_history[-2:]): # Basic check to avoid double add
                         self.chat_history.append({'role': 'user', 'content': user_msg})
                    if not stream_error and response: self.chat_history.append({'role': 'assistant', 'content': response})

                    # Archive logic
                    archive_msgs = self.chat_history[self.last_archive_index:]
                    if archive_msgs:
                        content = format_chat_history(archive_msgs)
                        if content:
                            success, _ = save_chat_to_s3(self.config.agent_name, content, self.config.event_id, False, self.current_chat_file)
                            if success: self.last_archive_index = len(self.chat_history); logger.debug("Auto-archived.")
                            else: logger.error("Failed auto-archive.")
                    yield f"data: {json.dumps({'done': True})}\n\n"

                return Response(generate(), mimetype='text/event-stream')
            except Exception as e: logger.error(f"/api/chat error: {e}", exc_info=True); return jsonify({'error': 'Server error'}), 500

        @self.app.route('/api/status', methods=['GET'])
        def status():
             mem = getattr(self.config, 'memory', None) is not None; tx = getattr(self.config, 'listen_transcript', False)
             return jsonify({'agent_name': self.config.agent_name, 'listen_transcript': tx, 'memory_enabled': mem})

        @self.app.route('/api/command', methods=['POST'])
        def command():
            data = request.json; cmd = data.get('command','').lower(); msg = f"Cmd: !{cmd}"; code=200; resp={}
            if not cmd: return jsonify({'error': 'No command'}), 400
            logger.info(f"WebChat: Cmd: !{cmd}")
            try:
                if cmd == 'help': msg = "Cmds: !help, !clear, !save, !memory, !listen-transcript"
                elif cmd == 'clear': self.chat_history=[]; self.last_saved_index=0; self.last_archive_index=0; msg='History cleared.'
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
                     else: msg = f"Tx listening {status}."
                else: msg = f"Unknown cmd: !{cmd}"; code = 400
                resp['message'] = msg
                if code == 200: resp['status'] = {'listen_transcript': self.config.listen_transcript, 'memory_enabled': self.config.memory is not None}
                else: resp['error'] = msg
                return jsonify(resp), code
            except Exception as e: logger.error(f"Cmd !{cmd} error: {e}", exc_info=True); return jsonify({'error': str(e)}), 500

    def load_initial_transcript(self):
        """Load initial transcript content."""
        try: content = read_all_transcripts_in_folder(self.config.agent_name, self.config.event_id)
        except Exception as e: logger.error(f"Error checking all transcripts: {e}"); content = None
        if content:
             logger.info(f"Loaded initial tx ({len(content)} chars).")
             if "[INITIAL TRANSCRIPT]" not in self.system_prompt: self.system_prompt += f"\n\n[TX Start]\n{content[:3000]}..."
             self.transcript_state = TranscriptState(); return True
        else: logger.info("No initial tx found."); return False

    def check_transcript_updates(self) -> Optional[str]:
        """Check for new transcript content."""
        if not self.config.listen_transcript: return None
        try:
            if not hasattr(self, 'transcript_state') or self.transcript_state is None: self.transcript_state = TranscriptState(); logger.warning("TxState re-init.")
            new = read_new_transcript_content(self.transcript_state, self.config.agent_name, self.config.event_id, False)
            if new: logger.debug(f"New tx found ({len(new)} chars)."); return new
            return None
        except Exception as e: logger.error(f"Tx check error: {e}", exc_info=True); return None

    def run(self, host: str = '127.0.0.1', port: int = 5001, debug: bool = False):
        """Run the Flask web server."""
        def transcript_update_loop():
             logger.info("Starting internal tx update loop.")
             while True: time.sleep(5); self.check_transcript_updates()
        if self.config.interface_mode != 'cli':
            update_thread = threading.Thread(target=transcript_update_loop, daemon=True); update_thread.start()
        logger.info(f"Starting Flask server on {host}:{port}, Debug: {debug}")
        try: self.app.run(host=host, port=port, debug=debug, use_reloader=False)
        except Exception as e: logger.critical(f"Flask server failed: {e}", exc_info=True)