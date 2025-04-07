from flask import Flask, request, jsonify, render_template, current_app, Response
from typing import Optional, List, Dict, Any # Added List, Dict, Any
from utils.transcript_utils import TranscriptState, get_latest_transcript_file, read_new_transcript_content, read_all_transcripts_in_folder
from utils.retrieval_handler import RetrievalHandler
import threading
from config import AppConfig
import os
import boto3
import logging # Import the logging module
from datetime import datetime
import json
import time
import traceback # Import traceback for detailed error logging

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Assuming magic_chat contains necessary S3 functions & classes
# Need to be careful about circular imports if magic_chat imports WebChat
try:
    # Import specific functions needed, avoid importing magic_chat itself if possible
    from magic_chat import (
        get_latest_system_prompt,
        get_latest_frameworks,
        get_latest_context,
        get_agent_docs,
        load_existing_chats_from_s3,
        save_chat_to_s3 as save_chat_to_s3_magic, # Alias to avoid name clash
        format_chat_history,
        TranscriptState, # Import TranscriptState class
        reload_memory as reload_memory_magic # Import reload_memory
    )
    magic_imports_ok = True
except ImportError as e:
    logger.error(f"Failed to import from magic_chat: {e}. Some functionalities might be limited. Check for circular imports.")
    magic_imports_ok = False
    # Define fallbacks if needed
    def get_latest_system_prompt(agent_name=None): return "Default System Prompt: Error loading from S3."
    def get_latest_frameworks(agent_name=None): return None
    def get_latest_context(agent_name, event_id=None): return None
    def get_agent_docs(agent_name): return None
    def load_existing_chats_from_s3(agent_name, memory_agents=None): return []
    def save_chat_to_s3_magic(agent_name, chat_content, event_id, is_saved=False, filename=None): return False, None
    def format_chat_history(messages): return ""
    class TranscriptState: pass
    def reload_memory_magic(agent_name, memory_agents, initial_system_prompt): return initial_system_prompt


# WebChat Class Definition
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
        self.transcript_state = TranscriptState()
        self.scheduler_thread = None
        self.last_saved_index = 0
        self.last_archive_index = 0
        self.current_chat_file = None

        try:
             timestamp = datetime.now().strftime('%Y%m%d-T%H%M%S')
             if not hasattr(config, 'session_id') or not config.session_id:
                 config.session_id = timestamp # Ensure session_id exists on config
             event_id = config.event_id or '0000'
             self.current_chat_file = f"chat_D{config.session_id}_aID-{config.agent_name}_eID-{event_id}.txt"
             logger.info(f"WebChat: Initialized chat filename: {self.current_chat_file}")

             self.load_resources()

             from anthropic import Anthropic
             self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
             logger.info("WebChat: Anthropic client initialized.")

        except Exception as e:
             logger.error(f"WebChat: Error during initialization: {e}", exc_info=True)

    def get_document_context(self, query: str):
        """Get relevant document context for query using the initialized retriever."""
        if not self.retriever:
             logger.error("WebChat: Retriever not initialized.")
             return None
        try:
            logger.debug(f"WebChat: Getting document context for query: {query[:100]}...")
            is_transcript = any(word in query.lower() for word in ['transcript', 'conversation', 'meeting', 'session', 'said'])
            logger.debug(f"WebChat: Is transcript query hint: {is_transcript}")

            contexts = self.retriever.get_relevant_context(
                query=query, top_k=3, is_transcript=is_transcript
            )

            if not contexts:
                logger.info(f"WebChat: No relevant context found by retriever for query: {query[:100]}...")
                return None

            logger.info(f"WebChat: Retrieved {len(contexts)} relevant context documents.")
            for i, context in enumerate(contexts[:2]):
                logger.debug(f"WebChat: Context {i+1} metadata: {context.metadata}")
                logger.debug(f"WebChat: Context {i+1} content: {context.page_content[:150]}...")
            return contexts

        except Exception as e:
            logger.error(f"WebChat: Error retrieving document context: {e}", exc_info=True)
            return None

    def load_resources(self):
        """Load prompts, init retriever, start scheduler."""
        try:
            logger.info("WebChat: Loading resources...")
            base_system_prompt = get_latest_system_prompt(self.config.agent_name) if magic_imports_ok else "Default Prompt"
            if not base_system_prompt or base_system_prompt == "Default Prompt":
                logger.error("WebChat: Failed to load base system prompt! Using fallback.")
                base_system_prompt = "You are a helpful assistant."

            source_instructions = "\n\n## Source Attribution Requirements\n..." # Keep existing instructions
            self.system_prompt = base_system_prompt + source_instructions

            frameworks = get_latest_frameworks(self.config.agent_name) if magic_imports_ok else None
            if frameworks: self.system_prompt += "\n\n## Frameworks\n" + frameworks; logger.info("WebChat: Loaded frameworks.")
            context = get_latest_context(self.config.agent_name, self.config.event_id) if magic_imports_ok else None
            if context: self.system_prompt += "\n\n## Context\n" + context; logger.info("WebChat: Loaded context.")
            docs = get_agent_docs(self.config.agent_name) if magic_imports_ok else None
            if docs: self.system_prompt += "\n\n## Agent Documentation\n" + docs; logger.info("WebChat: Loaded agent documentation.")

            self.retriever = RetrievalHandler(
                index_name=self.config.index, agent_name=self.config.agent_name,
                session_id=self.config.session_id, event_id=self.config.event_id
            )
            logger.info(f"WebChat: RetrievalHandler initialized for index '{self.config.index}', agent '{self.config.agent_name}', event '{self.config.event_id}'.")

            if self.config.memory is not None and magic_imports_ok:
                self.system_prompt = self.reload_memory() # Uses reload_memory_magic via self
                logger.info("WebChat: Memory reloaded.")
            elif self.config.memory is not None and not magic_imports_ok:
                 logger.warning("WebChat: Cannot reload memory, imports from magic_chat failed.")

            if self.config.listen_transcript: self.load_initial_transcript()

            logger.info("WebChat: Resources loaded successfully.")
            logger.debug(f"WebChat: Final system prompt length: {len(self.system_prompt)} chars.")

        except Exception as e:
            logger.error(f"WebChat: Error loading resources: {e}", exc_info=True)
            if not self.system_prompt: self.system_prompt = "Error loading configuration."

    def reload_memory(self):
        """Append memory summary to the system prompt using imported function."""
        if not magic_imports_ok:
             logger.error("WebChat: Cannot reload memory, required functions not imported.")
             return self.system_prompt
        # Call the imported reload_memory function
        return reload_memory_magic(self.config.agent_name, self.config.memory, self.system_prompt)


    def setup_routes(self):
        @self.app.route('/')
        def index():
            template_name = 'index_yggdrasil.html' if self.config.agent_name == 'yggdrasil' else 'index.html'
            logger.debug(f"Rendering template: {template_name} for agent: {self.config.agent_name}")
            return render_template(template_name, agent_name=self.config.agent_name)

        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            try:
                data = request.json
                if not data or 'message' not in data:
                    logger.warning("Chat request received without message data.")
                    return jsonify({'error': 'No message provided'}), 400

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
                          source_file = doc.metadata.get('file_name', 'Unknown source')
                          score = doc.metadata.get('score', 0.0)
                          context_items.append(f"[Context {i+1} from {source_file} (Score: {score:.2f})]:\n{doc.page_content}")
                     context_text_for_prompt = "\n\n---\nRelevant Context Found:\n" + "\n\n".join(context_items)
                     logger.debug(f"WebChat: Adding retrieved context to prompt ({len(context_text_for_prompt)} chars).")
                     current_system_prompt += context_text_for_prompt
                else:
                     logger.debug("WebChat: No relevant context retrieved to add to prompt.")

                new_transcript_chunk = self.check_transcript_updates()
                if new_transcript_chunk:
                     logger.debug(f"WebChat: Adding new transcript chunk to messages ({len(new_transcript_chunk)} chars).")
                     llm_messages.append({"role": "user", "content": f"[LIVE TRANSCRIPT UPDATE]\n{new_transcript_chunk}"})

                if not self.client:
                    logger.error("WebChat: Anthropic client not initialized.")
                    return jsonify({'error': 'Chat client not available'}), 500

                # --- Use model name from config ---
                model_to_use = self.config.llm_model_name
                logger.debug(f"WebChat: Using LLM model: {model_to_use}")
                # --- End Use model name ---

                def generate():
                    full_response = ""
                    try:
                        logger.debug(f"WebChat: Calling LLM with {len(llm_messages)} messages. System prompt length: {len(current_system_prompt)}")
                        with self.client.messages.stream(
                            model=model_to_use, # Use variable here
                            max_tokens=1024,
                            system=current_system_prompt,
                            messages=llm_messages
                        ) as stream:
                            for text in stream.text_stream:
                                full_response += text
                                yield f"data: {json.dumps({'delta': text})}\n\n"
                        logger.info(f"WebChat: LLM response received ({len(full_response)} chars).")

                        self.chat_history.append({'role': 'assistant', 'content': full_response})

                        new_messages_to_archive = self.chat_history[self.last_archive_index:]
                        if new_messages_to_archive and magic_imports_ok:
                            try: archive_content = format_chat_history(new_messages_to_archive)
                            except Exception: archive_content = ""; # Basic fallback
                            if archive_content:
                                success, _ = save_chat_to_s3_magic(
                                    agent_name=self.config.agent_name, chat_content=archive_content.strip(),
                                    event_id=self.config.event_id or '0000', is_saved=False,
                                    filename=self.current_chat_file
                                )
                                if success: self.last_archive_index = len(self.chat_history); logger.debug(f"WebChat: Auto-archived {len(new_messages_to_archive)} messages.")
                                else: logger.error("WebChat: Failed to auto-archive chat history.")
                        elif new_messages_to_archive and not magic_imports_ok:
                             logger.warning("WebChat: Cannot auto-archive, imports failed.")

                    except Exception as stream_e:
                        logger.error(f"WebChat: Error during LLM stream or archiving: {stream_e}", exc_info=True)
                        yield f"data: {json.dumps({'error': 'An error occurred generating the response.'})}\n\n"
                    finally:
                        yield f"data: {json.dumps({'done': True})}\n\n"

                return Response(generate(), mimetype='text/event-stream')

            except Exception as e:
                logger.error(f"WebChat: Error in /api/chat endpoint: {e}", exc_info=True)
                return jsonify({'error': 'An internal server error occurred'}), 500

        @self.app.route('/api/status', methods=['GET'])
        def status():
             listen_summary = getattr(self.config, 'listen_summary', False)
             listen_transcript = getattr(self.config, 'listen_transcript', False)
             listen_insights = getattr(self.config, 'listen_insights', False)
             memory_enabled = getattr(self.config, 'memory', None) is not None
             return jsonify({
                 'agent_name': self.config.agent_name, 'listen_summary': listen_summary,
                 'listen_transcript': listen_transcript, 'listen_insights': listen_insights,
                 'memory_enabled': memory_enabled
             })

        @self.app.route('/api/command', methods=['POST'])
        def command():
            # Simplified - assumes magic_imports_ok for save/reload
            data = request.json
            if not data or 'command' not in data: return jsonify({'error': 'No command provided'}), 400
            cmd = data['command'].lower()
            message = f"Executing command: !{cmd}"
            logger.info(f"WebChat: Received command: !{cmd}")
            update_status_data = {}
            status_code = 200

            try:
                if cmd == 'help':
                    message = ("Available commands:\n!help\n!clear\n!save\n!memory\n!listen-transcript")
                elif cmd == 'clear':
                    self.chat_history = []; self.last_saved_index = 0; self.last_archive_index = 0
                    message = 'Chat history cleared.'
                elif cmd == 'save':
                    new_messages_to_save = self.chat_history[self.last_saved_index:]
                    if not new_messages_to_save: message = 'No new messages to save manually.'
                    elif not magic_imports_ok: message = 'Error: Save function not available.'; status_code = 500
                    else:
                        try: save_content = format_chat_history(new_messages_to_save)
                        except Exception: save_content = ""; # Fallback
                        success, filename = save_chat_to_s3_magic(
                            agent_name=self.config.agent_name, chat_content=save_content.strip(),
                            event_id=self.config.event_id or '0000', is_saved=True,
                            filename=self.current_chat_file
                        )
                        if success: self.last_saved_index = len(self.chat_history); message = f'Chat history manually saved as {filename}'
                        else: message = 'Error: Failed to save chat history manually.'; status_code = 500
                elif cmd == 'memory':
                     if not magic_imports_ok: message = 'Error: Memory function not available.'; status_code = 500
                     else:
                         if self.config.memory is None:
                              self.config.memory = [self.config.agent_name]
                              self.system_prompt = self.reload_memory()
                              message = 'Memory mode ACTIVATED.'
                         else:
                              self.config.memory = None
                              self.load_resources() # Reset prompt
                              message = 'Memory mode DEACTIVATED.'
                elif cmd == 'listen-transcript':
                     self.config.listen_transcript = not self.config.listen_transcript
                     if self.config.listen_transcript:
                          loaded = self.load_initial_transcript()
                          message = f"Transcript listening ENABLED." + (" Initial transcript loaded." if loaded else "")
                     else: message = "Transcript listening DISABLED."
                else: message = f"Unknown command: !{cmd}"; status_code = 400

                update_status_data = {
                     'listen_transcript': self.config.listen_transcript,
                     'memory_enabled': self.config.memory is not None
                 }
                response_data = {'message': message}
                if status_code == 200: response_data['status'] = update_status_data
                else: response_data['error'] = message # Use error field for non-200 responses

                return jsonify(response_data), status_code

            except Exception as e:
                 logger.error(f"WebChat: Error executing command !{cmd}: {e}", exc_info=True)
                 return jsonify({'error': f'Error executing command: {str(e)}'}), 500

    def load_initial_transcript(self):
        """Load initial transcript content."""
        if not magic_imports_ok: return False
        try:
             all_content = read_all_transcripts_in_folder(self.config.agent_name, self.config.event_id)
             if all_content:
                 logger.info(f"WebChat: Loaded initial transcript content ({len(all_content)} chars).")
                 if "[INITIAL TRANSCRIPT]" not in self.system_prompt:
                      self.system_prompt += f"\n\n[INITIAL TRANSCRIPT]\n{all_content[:3000]}..."
                 self.transcript_state = TranscriptState() # Reset state
                 return True
             else: logger.info("WebChat: No initial transcript content found."); return False
        except Exception as e: logger.error(f"WebChat: Error loading initial transcript: {e}", exc_info=True); return False

    def check_transcript_updates(self) -> Optional[str]:
        """Check for new transcript content."""
        if not self.config.listen_transcript or not magic_imports_ok: return None
        try:
            if not hasattr(self, 'transcript_state') or self.transcript_state is None:
                self.transcript_state = TranscriptState(); logger.warning("WebChat: TranscriptState re-initialized.")
            new_content = read_new_transcript_content(self.transcript_state, self.config.agent_name, self.config.event_id, read_all=False)
            if new_content: logger.debug(f"WebChat: New transcript content detected ({len(new_content)} chars)."); return new_content
            else: return None
        except Exception as e: logger.error(f"WebChat: Error checking transcript updates: {e}", exc_info=True); return None

    def run(self, host: str = '127.0.0.1', port: int = 5001, debug: bool = False):
        """Run the Flask web server."""
        def transcript_update_loop():
             logger.info("WebChat: Starting internal transcript update loop.")
             while True:
                  time.sleep(5)
                  if self.config.listen_transcript:
                       new_chunk = self.check_transcript_updates()
                       # Currently, new chunks are picked up just before LLM call in /api/chat
                       # if new_chunk: logger.debug("WebChat Update Loop: Found new data.")

        if self.config.interface_mode != 'cli':
            update_thread = threading.Thread(target=transcript_update_loop, daemon=True)
            update_thread.start()

        logger.info(f"WebChat: Starting Flask server on {host}:{port}, Debug: {debug}")
        try:
             self.app.run(host=host, port=port, debug=debug, use_reloader=False)
        except Exception as e:
             logger.critical(f"WebChat: Flask server failed to run: {e}", exc_info=True)