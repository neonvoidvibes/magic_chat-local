from flask import Flask, request, jsonify, render_template, current_app, Response
from typing import Optional, List, Dict, Any
import threading
import os
import logging
from datetime import datetime
import json
import time
import traceback

# Import shared utilities
from config import AppConfig
from utils.retrieval_handler import RetrievalHandler
from utils.transcript_utils import TranscriptState, read_new_transcript_content, read_all_transcripts_in_folder
from utils.s3_utils import ( # Import S3 functions from the utility file
    get_latest_system_prompt,
    get_latest_frameworks,
    get_latest_context,
    get_agent_docs,
    load_existing_chats_from_s3,
    save_chat_to_s3,
    format_chat_history
)

# Import Anthropic client library
from anthropic import Anthropic

# Initialize logger for this module
logger = logging.getLogger(__name__)


class WebChat:
    def __init__(self, config: AppConfig):
        self.config = config
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        self.app.config['SECRET_KEY'] = os.urandom(24)
        self.setup_routes()
        self.chat_history = [] # List of {'role': 'user'/'assistant', 'content': '...'}
        self.client = None
        self.system_prompt = "Default system prompt."
        self.retriever = None
        self.transcript_state = TranscriptState() # Each WebChat instance has its own state
        self.scheduler_thread = None # Transcript scheduler thread (if managed internally)
        self.last_saved_index = 0 # For manual !save command
        self.last_archive_index = 0 # For auto-archiving per turn
        self.current_chat_file = None

        try:
             # Ensure session_id exists from the start
             if not hasattr(config, 'session_id') or not config.session_id:
                 config.session_id = datetime.now().strftime('%Y%m%d-T%H%M%S')
             event_id = config.event_id or '0000'
             self.current_chat_file = f"chat_D{config.session_id}_aID-{config.agent_name}_eID-{event_id}.txt"
             logger.info(f"WebChat: Session {config.session_id}, Chat file: {self.current_chat_file}")

             # Load prompts, context, etc., and initialize retriever
             self.load_resources()

             # Initialize Anthropic client
             self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
             logger.info("WebChat: Anthropic client initialized.")

        except Exception as e:
             logger.error(f"WebChat: Error during initialization: {e}", exc_info=True)
             # Prevent Flask from starting if basic setup failed?
             raise RuntimeError("WebChat initialization failed") from e


    def get_document_context(self, query: str) -> Optional[List[Dict]]:
        """Get relevant document context using the RetrievalHandler."""
        if not self.retriever:
             logger.error("WebChat: Retriever not initialized.")
             return None
        try:
            logger.debug(f"WebChat: Getting document context for query: {query[:100]}...")
            is_transcript = any(word in query.lower() for word in ['transcript', 'conversation', 'meeting', 'session', 'said'])
            logger.debug(f"WebChat: Is transcript query hint: {is_transcript}")

            # Use the handler instance to get context
            # The handler uses its own config (agent_name, event_id) for filtering
            contexts = self.retriever.get_relevant_context(
                query=query, top_k=3, is_transcript=is_transcript
            )

            if not contexts:
                logger.info(f"WebChat: No relevant context found by retriever.")
                return None

            logger.info(f"WebChat: Retrieved {len(contexts)} relevant context documents.")
            # Return the list of Document objects provided by the handler
            return contexts

        except Exception as e:
            logger.error(f"WebChat: Error retrieving document context: {e}", exc_info=True)
            return None

    def load_resources(self):
        """Load prompts, context, docs, init retriever, handle memory."""
        try:
            logger.info("WebChat: Loading resources...")
            # Use imported S3 functions
            base_system_prompt = get_latest_system_prompt(self.config.agent_name)
            if not base_system_prompt:
                 logger.error("WebChat: Failed to load base system prompt! Using fallback.")
                 base_system_prompt = "You are a helpful assistant."

            source_instructions = "\n\n## Source Attribution Requirements\n..." # Keep existing instructions
            self.system_prompt = base_system_prompt + source_instructions

            frameworks = get_latest_frameworks(self.config.agent_name)
            if frameworks: self.system_prompt += "\n\n## Frameworks\n" + frameworks; logger.info("WebChat: Loaded frameworks.")
            context = get_latest_context(self.config.agent_name, self.config.event_id)
            if context: self.system_prompt += "\n\n## Context\n" + context; logger.info("WebChat: Loaded context.")
            docs = get_agent_docs(self.config.agent_name)
            if docs: self.system_prompt += "\n\n## Agent Documentation\n" + docs; logger.info("WebChat: Loaded agent documentation.")

            # Init Retriever
            self.retriever = RetrievalHandler(
                index_name=self.config.index, agent_name=self.config.agent_name,
                session_id=self.config.session_id, event_id=self.config.event_id
            )
            logger.info(f"WebChat: RetrievalHandler initialized.")

            # Handle memory (appends to self.system_prompt)
            if self.config.memory is not None:
                self.reload_memory() # Calls internal method which calls imported function
                logger.info("WebChat: Memory loaded/reloaded.")

            if self.config.listen_transcript: self.load_initial_transcript()

            logger.info("WebChat: Resources loaded successfully.")
            logger.debug(f"WebChat: Final system prompt length: {len(self.system_prompt)} chars.")

        except Exception as e:
            logger.error(f"WebChat: Error loading resources: {e}", exc_info=True)
            if not self.system_prompt: self.system_prompt = "Error loading configuration."


    def reload_memory(self):
        """Append memory summary to the system prompt using imported function."""
        logger.debug("WebChat: Reloading memory...")
        # Use imported reload function (from s3_utils via magic_chat import for now)
        # Need to ensure reload_memory logic is correctly placed in s3_utils
        # Assuming load_existing_chats_from_s3 is available via import:
        try:
             agents_to_load = self.config.memory if self.config.memory else [self.config.agent_name]
             previous_chats = load_existing_chats_from_s3(self.config.agent_name, agents_to_load)
             if not previous_chats: logger.debug("WebChat: No memory files found."); return

             all_content_items = []
             for chat in previous_chats:
                  file_info = f"(Memory file: {os.path.basename(chat.get('file', 'unknown'))})"
                  for msg in chat.get('messages', []):
                       role = msg.get('role', 'unknown').capitalize()
                       content = msg.get('content', '')
                       if content: all_content_items.append(f"{role} {file_info}: {content}")

             combined_content = "\n\n---\n\n".join(all_content_items)
             max_mem_len = 10000 # Truncate if needed
             summarized_content = combined_content[:max_mem_len] + ("..." if len(combined_content) > max_mem_len else "")

             if summarized_content:
                 memory_section = "\n\n## Previous Chat History (Memory)\n" + summarized_content
                 if "## Previous Chat History (Memory)" not in self.system_prompt:
                      self.system_prompt += memory_section
                      logger.info(f"WebChat: Appended memory summary ({len(summarized_content)} chars).")
                 else: logger.warning("WebChat: Memory section already in prompt, skipping append.")
             else: logger.debug("WebChat: No content extracted for memory.")

        except Exception as e:
            logger.error(f"WebChat: Error during memory reload: {e}", exc_info=True)


    def setup_routes(self):
        @self.app.route('/')
        def index():
            template_name = 'index_yggdrasil.html' if self.config.agent_name == 'yggdrasil' else 'index.html'
            logger.debug(f"Rendering template: {template_name}")
            return render_template(template_name, agent_name=self.config.agent_name)

        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            try:
                data = request.json
                if not data or 'message' not in data: return jsonify({'error': 'No message provided'}), 400

                user_message_content = data['message']
                logger.info(f"WebChat: Received message: {user_message_content[:100]}...")
                self.chat_history.append({'role': 'user', 'content': user_message_content})

                current_system_prompt = self.system_prompt
                llm_messages = list(self.chat_history)

                retrieved_docs = self.get_document_context(user_message_content)
                context_text_for_prompt = ""
                if retrieved_docs:
                     context_items = []
                     for i, doc in enumerate(retrieved_docs):
                          source_file = doc.metadata.get('file_name', 'Unknown source'); score = doc.metadata.get('score', 0.0)
                          context_items.append(f"[Context {i+1} from {source_file} (Score: {score:.2f})]:\n{doc.page_content}")
                     context_text_for_prompt = "\n\n---\nRelevant Context Found:\n" + "\n\n".join(context_items)
                     logger.debug(f"WebChat: Adding retrieved context ({len(context_text_for_prompt)} chars).")
                     current_system_prompt += context_text_for_prompt
                else: logger.debug("WebChat: No relevant context retrieved.")

                new_transcript_chunk = self.check_transcript_updates()
                if new_transcript_chunk:
                     logger.debug(f"WebChat: Adding transcript chunk ({len(new_transcript_chunk)} chars).")
                     llm_messages.append({"role": "user", "content": f"[LIVE TRANSCRIPT UPDATE]\n{new_transcript_chunk}"})

                if not self.client: return jsonify({'error': 'Chat client not available'}), 500

                model_to_use = self.config.llm_model_name
                logger.debug(f"WebChat: Using LLM model: {model_to_use}")

                def generate():
                    full_response = ""
                    try:
                        logger.debug(f"WebChat: Calling LLM. Msgs: {len(llm_messages)}, SysPromptLen: {len(current_system_prompt)}")
                        with self.client.messages.stream(
                            model=model_to_use, max_tokens=1024,
                            system=current_system_prompt, messages=llm_messages
                        ) as stream:
                            for text in stream.text_stream:
                                full_response += text
                                yield f"data: {json.dumps({'delta': text})}\n\n"
                        logger.info(f"WebChat: LLM response received ({len(full_response)} chars).")

                        self.chat_history.append({'role': 'assistant', 'content': full_response})

                        new_messages_to_archive = self.chat_history[self.last_archive_index:]
                        if new_messages_to_archive:
                            try: archive_content = format_chat_history(new_messages_to_archive)
                            except Exception: archive_content = ""; logger.warning("Fallback used for format_chat_history")
                            if archive_content:
                                success, _ = save_chat_to_s3( # Use imported save
                                    agent_name=self.config.agent_name, chat_content=archive_content.strip(),
                                    event_id=self.config.event_id or '0000', is_saved=False,
                                    filename=self.current_chat_file
                                )
                                if success: self.last_archive_index = len(self.chat_history); logger.debug("WebChat: Auto-archived turn.")
                                else: logger.error("WebChat: Failed to auto-archive.")
                    except Exception as stream_e:
                        logger.error(f"WebChat: Error during LLM stream/archive: {stream_e}", exc_info=True)
                        yield f"data: {json.dumps({'error': 'Error generating response.'})}\n\n"
                    finally:
                        yield f"data: {json.dumps({'done': True})}\n\n"

                return Response(generate(), mimetype='text/event-stream')

            except Exception as e:
                logger.error(f"WebChat: Error in /api/chat: {e}", exc_info=True)
                return jsonify({'error': 'Internal server error'}), 500

        @self.app.route('/api/status', methods=['GET'])
        def status():
             # Simplified status
             memory_enabled = getattr(self.config, 'memory', None) is not None
             listen_transcript = getattr(self.config, 'listen_transcript', False)
             return jsonify({
                 'agent_name': self.config.agent_name,
                 'listen_transcript': listen_transcript,
                 'memory_enabled': memory_enabled
             })

        @self.app.route('/api/command', methods=['POST'])
        def command():
            data = request.json
            if not data or 'command' not in data: return jsonify({'error': 'No command provided'}), 400
            cmd = data['command'].lower(); message = f"Executing command: !{cmd}"
            logger.info(f"WebChat: Received command: !{cmd}")
            status_code = 200; response_data = {}

            try:
                if cmd == 'help': message = "Commands: !help, !clear, !save, !memory, !listen-transcript"
                elif cmd == 'clear': self.chat_history = []; self.last_saved_index=0; self.last_archive_index=0; message = 'Chat history cleared.'
                elif cmd == 'save':
                    new_msgs = self.chat_history[self.last_saved_index:]
                    if not new_msgs: message = 'No new messages to save.'
                    else:
                        content = format_chat_history(new_msgs)
                        success, fname = save_chat_to_s3(self.config.agent_name, content, self.config.event_id, True, self.current_chat_file)
                        if success: self.last_saved_index = len(self.chat_history); message = f'Chat saved as {fname}'
                        else: message = 'Error saving chat.'; status_code = 500
                elif cmd == 'memory':
                     if self.config.memory is None: self.config.memory = [self.config.agent_name]; self.reload_memory(); message = 'Memory ACTIVATED.'
                     else: self.config.memory = None; self.load_resources(); message = 'Memory DEACTIVATED.' # Reload resources to remove memory prompt
                elif cmd == 'listen-transcript':
                     self.config.listen_transcript = not self.config.listen_transcript
                     if self.config.listen_transcript: loaded = self.load_initial_transcript(); message = f"Transcript listening ENABLED." + (" Initial transcript loaded." if loaded else "")
                     else: message = "Transcript listening DISABLED."
                else: message = f"Unknown command: !{cmd}"; status_code = 400

                response_data['message'] = message
                if status_code == 200: response_data['status'] = {'listen_transcript': self.config.listen_transcript, 'memory_enabled': self.config.memory is not None}
                else: response_data['error'] = message
                return jsonify(response_data), status_code
            except Exception as e: logger.error(f"Command !{cmd} error: {e}", exc_info=True); return jsonify({'error': str(e)}), 500

    def load_initial_transcript(self):
        """Load initial transcript content."""
        try:
             all_content = read_all_transcripts_in_folder(self.config.agent_name, self.config.event_id)
             if all_content:
                 logger.info(f"WebChat: Loaded initial transcript ({len(all_content)} chars).")
                 if "[INITIAL TRANSCRIPT]" not in self.system_prompt: self.system_prompt += f"\n\n[INITIAL TRANSCRIPT]\n{all_content[:3000]}..."
                 self.transcript_state = TranscriptState()
                 return True
             else: logger.info("WebChat: No initial transcript found."); return False
        except Exception as e: logger.error(f"WebChat: Error loading initial transcript: {e}", exc_info=True); return False

    def check_transcript_updates(self) -> Optional[str]:
        """Check for new transcript content."""
        if not self.config.listen_transcript: return None
        try:
            if not hasattr(self, 'transcript_state') or self.transcript_state is None: self.transcript_state = TranscriptState(); logger.warning("WebChat: TranscriptState re-init.")
            new_content = read_new_transcript_content(self.transcript_state, self.config.agent_name, self.config.event_id, read_all=False)
            if new_content: logger.debug(f"WebChat: New transcript found ({len(new_content)} chars)."); return new_content
            else: return None
        except Exception as e: logger.error(f"WebChat: Error checking transcript: {e}", exc_info=True); return None

    def run(self, host: str = '127.0.0.1', port: int = 5001, debug: bool = False):
        """Run the Flask web server."""
        def transcript_update_loop():
             logger.info("WebChat: Starting internal transcript update loop.")
             while True:
                  time.sleep(5)
                  if self.config.listen_transcript: self.check_transcript_updates() # Call check, but result handled by chat endpoint

        if self.config.interface_mode != 'cli':
            update_thread = threading.Thread(target=transcript_update_loop, daemon=True); update_thread.start()

        logger.info(f"WebChat: Starting Flask server on {host}:{port}, Debug: {debug}")
        try: self.app.run(host=host, port=port, debug=debug, use_reloader=False)
        except Exception as e: logger.critical(f"WebChat: Flask server failed: {e}", exc_info=True)