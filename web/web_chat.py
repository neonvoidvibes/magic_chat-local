from flask import Flask, request, jsonify, render_template, current_app, Response
from typing import Optional
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
# Configure logging level if needed (it might be configured globally already)
# logging.basicConfig(level=logging.DEBUG) # Uncomment if global config isn't sufficient

# Assuming magic_chat contains necessary S3 functions
# Adjust imports if structure is different
try:
    from magic_chat import (
        get_latest_system_prompt,
        get_latest_frameworks,
        get_latest_context,
        get_agent_docs,
        load_existing_chats_from_s3,
        save_chat_to_s3 as save_chat_to_s3_magic, # Alias to avoid name clash if needed
        format_chat_history, # Added for saving
        TranscriptState
    )
except ImportError as e:
    logger.error(f"Failed to import from magic_chat: {e}. Some functionalities might be limited.")
    # Define fallbacks or raise error if essential
    def get_latest_system_prompt(agent_name=None): return "Default System Prompt: Error loading from S3."
    def get_latest_frameworks(agent_name=None): return None
    def get_latest_context(agent_name, event_id=None): return None
    def get_agent_docs(agent_name): return None
    def load_existing_chats_from_s3(agent_name, memory_agents=None): return []
    def save_chat_to_s3_magic(agent_name, chat_content, event_id, is_saved=False, filename=None): return False, None
    def format_chat_history(messages): return ""
    class TranscriptState: pass


# Note: s3_utils.py functions seem duplicated in magic_chat. Using magic_chat versions.

class WebChat:
    def __init__(self, config: AppConfig):
        self.config = config
        self.app = Flask(__name__, template_folder='templates', static_folder='static') # Explicitly set folders
        self.app.config['SECRET_KEY'] = os.urandom(24) # Needed for session context if used later
        self.setup_routes()
        self.chat_history = [] # Store as list of dicts {'role': 'user'/'assistant', 'content': '...'}
        self.client = None
        self.system_prompt = "Default system prompt." # Default value
        self.retriever = None # Initialize later after loading resources
        self.transcript_state = TranscriptState() # Initialize transcript state tracking
        self.scheduler_thread = None
        self.last_saved_index = 0
        self.last_archive_index = 0
        self.current_chat_file = None # Set during load_resources or init

        try:
             # Initialize session ID and chat filename with timestamp
             timestamp = datetime.now().strftime('%Y%m%d-T%H%M%S')
             if not hasattr(self.config, 'session_id') or not self.config.session_id:
                 self.config.session_id = timestamp # Set session ID if not already set
             event_id = self.config.event_id or '0000'
             self.current_chat_file = f"chat_D{timestamp}_aID-{config.agent_name}_eID-{event_id}.txt"
             logger.info(f"WebChat: Initialized chat filename: {self.current_chat_file}")

             self.load_resources() # Load prompts, init retriever, start scheduler

             # Initialize Anthropic client
             from anthropic import Anthropic
             self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
             logger.info("WebChat: Anthropic client initialized.")

        except Exception as e:
             logger.error(f"WebChat: Error during initialization: {e}", exc_info=True)
             # Potentially set a state indicating an error

    def get_document_context(self, query: str):
        """Get relevant document context for query using the initialized retriever."""
        if not self.retriever:
             logger.error("WebChat: Retriever not initialized. Cannot get document context.")
             return None
        try:
            logger.debug(f"WebChat: Getting document context for query: {query[:100]}...")

            # Determine if this is a transcript query (optional hint for retriever)
            is_transcript = any(
                word in query.lower()
                for word in ['transcript', 'conversation', 'meeting', 'session', 'said']
            )
            logger.debug(f"WebChat: Is transcript query hint: {is_transcript}")

            # Get relevant contexts using RetrievalHandler - NO filter_metadata passed here
            contexts = self.retriever.get_relevant_context(
                query=query,
                # filter_metadata is removed - handler uses its own state (agent_name, event_id)
                top_k=3,
                is_transcript=is_transcript
            )

            if not contexts:
                logger.info(f"WebChat: No relevant context found by retriever for query: {query[:100]}...")
                return None

            # Log retrieval success
            logger.info(f"WebChat: Retrieved {len(contexts)} relevant context documents.")
            for i, context in enumerate(contexts[:2]): # Log first couple
                logger.debug(f"WebChat: Context {i+1} metadata: {context.metadata}")
                logger.debug(f"WebChat: Context {i+1} content: {context.page_content[:150]}...")
            return contexts # Return the list of Document objects

        except Exception as e:
            logger.error(f"WebChat: Error retrieving document context: {e}", exc_info=True)
            return None

    def load_resources(self):
        """Load prompts, init retriever, start scheduler."""
        try:
            logger.info("WebChat: Loading resources...")
            # Load base system prompt (keep core instructions here)
            base_system_prompt = get_latest_system_prompt(self.config.agent_name)
            if not base_system_prompt:
                logger.error("WebChat: Failed to load base system prompt!")
                base_system_prompt = "You are a helpful assistant." # Fallback

            # Add source attribution instructions (adjust as needed)
            source_instructions = "\n\n## Source Attribution Requirements\n" + \
                                  "1. ALWAYS specify the exact source file when quoting or referencing information.\n" + \
                                  "2. Format source references as: 'From [source file]: [quote]'.\n" + \
                                  "3. If you cannot determine the exact source file, explicitly state so.\n" + \
                                  "4. Differentiate between [VECTOR DB CONTENT] and [LIVE TRANSCRIPT] sources."
            self.system_prompt = base_system_prompt + source_instructions

            # Add frameworks
            frameworks = get_latest_frameworks(self.config.agent_name)
            if frameworks:
                self.system_prompt += "\n\n## Frameworks\n" + frameworks
                logger.info("WebChat: Loaded frameworks.")

            # Add context
            context = get_latest_context(self.config.agent_name, self.config.event_id)
            if context:
                self.system_prompt += "\n\n## Context\n" + context
                logger.info("WebChat: Loaded context.")

            # Add agent documentation
            docs = get_agent_docs(self.config.agent_name)
            if docs:
                self.system_prompt += "\n\n## Agent Documentation\n" + docs
                logger.info("WebChat: Loaded agent documentation.")

            # Initialize RetrievalHandler *after* config is potentially updated
            self.retriever = RetrievalHandler(
                index_name=self.config.index, # Use index from config
                agent_name=self.config.agent_name,
                session_id=self.config.session_id,
                event_id=self.config.event_id
            )
            logger.info(f"WebChat: RetrievalHandler initialized for index '{self.config.index}', agent '{self.config.agent_name}', event '{self.config.event_id}'.")

            # Load memory if enabled
            if self.config.memory is not None:
                self.system_prompt = self.reload_memory() # reload_memory adds to existing self.system_prompt
                logger.info("WebChat: Memory reloaded.")

            # Load initial transcript if listening enabled (simple load, not continuous check here)
            if self.config.listen_transcript:
                 self.load_initial_transcript()

            # Start rolling transcript scheduler if needed (might be redundant if external scheduler runs)
            # Ensure it doesn't run if already started externally
            # if not self.scheduler_thread:
            #     from scripts.transcript_scheduler import start_scheduler
            #     self.scheduler_thread = start_scheduler(
            #         agent_name=self.config.agent_name,
            #         session_id=self.config.session_id,
            #         event_id=self.config.event_id
            #     )
            #     logging.info("WebChat: Transcript scheduler thread started.")

            logger.info("WebChat: Resources loaded successfully.")
            logger.debug(f"WebChat: Final system prompt length: {len(self.system_prompt)} chars.")

        except Exception as e:
            logger.error(f"WebChat: Error loading resources: {e}", exc_info=True)
            # Ensure system_prompt has a fallback
            if not self.system_prompt:
                self.system_prompt = "Error loading configuration. Please act as a basic assistant."


    def reload_memory(self):
        """Append memory summary to the system prompt."""
        try:
            logger.debug("WebChat: Reloading memory...")
            if not self.system_prompt:
                logger.error("WebChat: Cannot reload memory, base system prompt not loaded.")
                return "Base system prompt failed to load." # Return a usable string

            # Ensure memory list is set correctly
            agents_to_load = self.config.memory
            if not agents_to_load: # If --memory was used without arguments
                agents_to_load = [self.config.agent_name]
            if not agents_to_load: # If memory somehow still empty
                 logger.warning("WebChat: Memory reload requested but no agents specified.")
                 return self.system_prompt # Return original prompt

            logger.debug(f"WebChat: Loading chat history for agents: {agents_to_load}")
            previous_chats = load_existing_chats_from_s3(self.config.agent_name, agents_to_load)
            logger.debug(f"WebChat: Loaded {len(previous_chats)} chat files for memory.")

            if not previous_chats:
                logger.debug("WebChat: No previous chat history found to load.")
                return self.system_prompt # Return original prompt

            # Combine and summarize chat content (implement summarization if needed)
            all_content_items = []
            for chat in previous_chats:
                 # Assuming chat['messages'] is a list of {'role': 'user'/'assistant', 'content': '...'}
                 for msg in chat.get('messages', []):
                      role = msg.get('role', 'unknown')
                      content = msg.get('content', '')
                      if content:
                           all_content_items.append(f"{role.capitalize()}: {content}")

            combined_content = "\n\n---\n\n".join(all_content_items)
            # Simple truncation for now, replace with actual summarization if needed
            max_mem_len = 5000 # Limit memory context size
            summarized_content = combined_content[:max_mem_len] + ("..." if len(combined_content) > max_mem_len else "")

            if summarized_content:
                memory_section = "\n\n## Previous Chat History (Memory)\n" + summarized_content
                logger.debug(f"WebChat: Appending memory summary ({len(summarized_content)} chars) to system prompt.")
                # Check if memory section already exists to avoid duplicates if called multiple times
                if "## Previous Chat History (Memory)" not in self.system_prompt:
                    return self.system_prompt + memory_section
                else:
                    logger.warning("WebChat: Memory section already found in system prompt, not appending again.")
                    return self.system_prompt # Avoid appending duplicates
            else:
                logger.debug("WebChat: No content extracted from previous chats for memory.")
                return self.system_prompt

        except Exception as e:
            logger.error(f"WebChat: Error reloading memory: {e}", exc_info=True)
            return self.system_prompt # Return original prompt on error


    def setup_routes(self):
        @self.app.route('/')
        def index():
            # Use specific template for yggdrasil if agent name matches
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

                # Add user message to history (simplified structure)
                self.chat_history.append({'role': 'user', 'content': user_message_content})

                # --- Prepare context for LLM ---
                current_system_prompt = self.system_prompt
                llm_messages = list(self.chat_history) # Start with current history

                # Get relevant document context from Retriever
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
                     # Prepend context to messages for better visibility by model? Or append to system prompt? Let's try appending to system.
                     current_system_prompt += context_text_for_prompt
                else:
                     logger.debug("WebChat: No relevant context retrieved to add to prompt.")


                # --- Add transcript update if listening ---
                # Check for NEW transcript content just before calling LLM
                new_transcript_chunk = self.check_transcript_updates()
                if new_transcript_chunk:
                     logger.debug(f"WebChat: Adding new transcript chunk to messages ({len(new_transcript_chunk)} chars).")
                     # Add as a user message to ensure model sees it in sequence
                     llm_messages.append({"role": "user", "content": f"[LIVE TRANSCRIPT UPDATE]\n{new_transcript_chunk}"})


                # --- Call LLM ---
                if not self.client:
                    logger.error("WebChat: Anthropic client not initialized.")
                    return jsonify({'error': 'Chat client not available'}), 500

                def generate():
                    full_response = ""
                    try:
                        logger.debug(f"WebChat: Calling LLM with {len(llm_messages)} messages. System prompt length: {len(current_system_prompt)}")
                        with self.client.messages.stream(
                            # model="claude-3-5-sonnet-20240620", # Choose appropriate model
                            model="claude-3-haiku-20240307", # Faster, cheaper option
                            max_tokens=1024,
                            system=current_system_prompt, # Includes base, frameworks, context, memory, retrieved docs
                            messages=llm_messages # User message + potential transcript update
                        ) as stream:
                            for text in stream.text_stream:
                                full_response += text
                                yield f"data: {json.dumps({'delta': text})}\n\n"
                        logger.info(f"WebChat: LLM response received ({len(full_response)} chars).")

                        # Add assistant response to history
                        self.chat_history.append({'role': 'assistant', 'content': full_response})

                        # Auto-archive the latest turn (user + assistant)
                        new_messages_to_archive = self.chat_history[self.last_archive_index:]
                        if new_messages_to_archive:
                            # Use format_chat_history if available and compatible
                            try:
                                archive_content = format_chat_history(new_messages_to_archive)
                            except Exception: # Fallback if format_chat_history fails or is incompatible
                                archive_content = ""
                                for msg in new_messages_to_archive:
                                    archive_content += f"**{msg.get('role', 'unknown').capitalize()}:**\n{msg.get('content', '')}\n\n"

                            if archive_content:
                                success, _ = save_chat_to_s3_magic(
                                    agent_name=self.config.agent_name,
                                    chat_content=archive_content.strip(),
                                    event_id=self.config.event_id or '0000',
                                    is_saved=False, # Save to archive
                                    filename=self.current_chat_file
                                )
                                if success:
                                    self.last_archive_index = len(self.chat_history)
                                    logger.debug(f"WebChat: Auto-archived {len(new_messages_to_archive)} messages to {self.current_chat_file}")
                                else:
                                    logger.error("WebChat: Failed to auto-archive chat history.")
                    except Exception as stream_e:
                        logger.error(f"WebChat: Error during LLM stream or archiving: {stream_e}", exc_info=True)
                        yield f"data: {json.dumps({'error': 'An error occurred generating the response.'})}\n\n"
                    finally:
                        # Signal end of stream
                        yield f"data: {json.dumps({'done': True})}\n\n"

                return Response(generate(), mimetype='text/event-stream')

            except Exception as e:
                logger.error(f"WebChat: Error in /api/chat endpoint: {e}", exc_info=True)
                return jsonify({'error': 'An internal server error occurred'}), 500


        @self.app.route('/api/status', methods=['GET'])
        def status():
             # Ensure config attributes exist
             listen_summary = getattr(self.config, 'listen_summary', False)
             listen_transcript = getattr(self.config, 'listen_transcript', False)
             listen_insights = getattr(self.config, 'listen_insights', False)
             memory_enabled = getattr(self.config, 'memory', None) is not None

             return jsonify({
                 'agent_name': self.config.agent_name,
                 'listen_summary': listen_summary,
                 'listen_transcript': listen_transcript,
                 'listen_insights': listen_insights,
                 'memory_enabled': memory_enabled
             })

        @self.app.route('/api/command', methods=['POST'])
        def command():
            data = request.json
            if not data or 'command' not in data:
                return jsonify({'error': 'No command provided'}), 400

            cmd = data['command'].lower()
            message = f"Executing command: !{cmd}"
            logger.info(f"WebChat: Received command: !{cmd}")

            try:
                if cmd == 'help':
                    help_text = (
                        "Available commands:\n"
                        "!help          - Display this help message\n"
                        "!clear         - Clear the chat history\n"
                        "!save          - Save current chat history to S3 saved folder\n"
                        "!memory        - Toggle memory mode (load chat history)\n"
                        # "!listen        - Enable summary listening\n" # Add back if needed
                        "!listen-transcript - Toggle transcript listening"
                        # Add other listen commands if implemented
                    )
                    message = help_text
                elif cmd == 'clear':
                    self.chat_history = []
                    self.last_saved_index = 0
                    self.last_archive_index = 0
                    message = 'Chat history cleared.'
                elif cmd == 'save':
                    # Get messages since last *manual* save
                    new_messages_to_save = self.chat_history[self.last_saved_index:]
                    if not new_messages_to_save:
                        message = 'No new messages to save manually.'
                    else:
                        try:
                            save_content = format_chat_history(new_messages_to_save)
                        except Exception: # Fallback
                             save_content = ""
                             for msg in new_messages_to_save:
                                  save_content += f"**{msg.get('role', 'unknown').capitalize()}:**\n{msg.get('content', '')}\n\n"

                        success, filename = save_chat_to_s3_magic(
                            agent_name=self.config.agent_name,
                            chat_content=save_content.strip(),
                            event_id=self.config.event_id or '0000',
                            is_saved=True, # Save to 'saved' folder
                            filename=self.current_chat_file
                        )
                        if success:
                            self.last_saved_index = len(self.chat_history) # Update manual save index
                            message = f'Chat history manually saved successfully as {filename}'
                        else:
                            message = 'Error: Failed to save chat history manually.'
                            return jsonify({'error': message}), 500
                elif cmd == 'memory':
                    if self.config.memory is None:
                        self.config.memory = [self.config.agent_name] # Default to self if empty
                        self.system_prompt = self.reload_memory() # Reload adds to prompt
                        message = 'Memory mode activated (reloaded previous chats).'
                    else:
                        self.config.memory = None
                        # Need to reset system prompt *without* memory
                        self.load_resources() # Reload all resources to reset prompt cleanly
                        message = 'Memory mode deactivated.'
                elif cmd == 'listen-transcript':
                     self.config.listen_transcript = not self.config.listen_transcript # Toggle
                     if self.config.listen_transcript:
                          # Try loading initial transcript when enabling
                          loaded = self.load_initial_transcript()
                          message = f"Transcript listening ENABLED." + (" Initial transcript loaded." if loaded else " No initial transcript found.")
                     else:
                          message = "Transcript listening DISABLED."
                else:
                    message = f"Unknown command: !{cmd}"
                    return jsonify({'error': message}), 400

                # Update status after command execution
                update_status_data = {
                     'listen_transcript': self.config.listen_transcript,
                     'memory_enabled': self.config.memory is not None
                     # add other status flags if needed
                 }
                return jsonify({'message': message, 'status': update_status_data})

            except Exception as e:
                 logger.error(f"WebChat: Error executing command !{cmd}: {e}", exc_info=True)
                 return jsonify({'error': f'Error executing command: {str(e)}'}), 500


        # Removed /api/save route as !save command handles manual saving now

    def load_initial_transcript(self):
        """Load initial transcript content when transcript listening is enabled."""
        try:
             # Use read_all_transcripts_in_folder to get everything initially
             all_content = read_all_transcripts_in_folder(self.config.agent_name, self.config.event_id)
             if all_content:
                 logger.info(f"WebChat: Loaded initial transcript content ({len(all_content)} chars).")
                 # Store it? Or just use it for immediate context? Let's add to system prompt for now.
                 # Avoid adding if already present to prevent duplication on toggling
                 if "[INITIAL TRANSCRIPT]" not in self.system_prompt:
                      self.system_prompt += f"\n\n[INITIAL TRANSCRIPT]\n{all_content[:3000]}..." # Limit initial load size
                 # Reset transcript state for subsequent updates
                 self.transcript_state = TranscriptState()
                 return True
             else:
                 logger.info("WebChat: No initial transcript content found.")
                 return False
        except Exception as e:
            logger.error(f"WebChat: Error loading initial transcript: {e}", exc_info=True)
            return False


    def check_transcript_updates(self) -> Optional[str]:
        """Check for and return new transcript content since last check."""
        if not self.config.listen_transcript:
             return None # Don't check if not enabled

        try:
            # Ensure transcript_state exists
            if not hasattr(self, 'transcript_state') or self.transcript_state is None:
                self.transcript_state = TranscriptState()
                logger.warning("WebChat: TranscriptState was missing, re-initialized.")

            # Use the utility function to read new content
            # Passing read_all=False uses the single 'latest file' update logic
            new_content = read_new_transcript_content(
                self.transcript_state,
                self.config.agent_name,
                self.config.event_id,
                read_all=False # Use single latest file checking
            )

            if new_content:
                logger.debug(f"WebChat: New transcript content detected ({len(new_content)} chars).")
                # Return the new content only, labeling happens in the chat endpoint
                return new_content
            else:
                # logger.debug("WebChat: No new transcript content found.") # Too noisy for regular checks
                return None

        except Exception as e:
            logger.error(f"WebChat: Error checking transcript updates: {e}", exc_info=True)
            return None


    def run(self, host: str = '127.0.0.1', port: int = 5001, debug: bool = False):
        """Run the Flask web server."""
        # Transcript update checking needs to run periodically in a background thread
        # Note: This internal checker might conflict if an external scheduler is also running.
        # Consider disabling one if both are active.
        def transcript_update_loop():
             logger.info("WebChat: Starting internal transcript update loop.")
             while True:
                  # Check interval
                  time.sleep(5) # Check every 5 seconds
                  if self.config.listen_transcript:
                       new_chunk = self.check_transcript_updates()
                       if new_chunk:
                            # How to notify the main chat thread or LLM?
                            # For now, check_transcript_updates is called just before LLM call.
                            # This loop might be redundant unless used for pushing updates via websockets etc.
                            logger.debug("WebChat Update Loop: Found new transcript data (will be picked up by next chat call).")


        # Start the update loop only if transcript listening might be enabled
        # And maybe only if not web_only? Assume CLI might handle its own updates.
        if self.config.interface_mode != 'cli': # If running web or web_only
            update_thread = threading.Thread(target=transcript_update_loop, daemon=True)
            update_thread.start()

        # Run Flask app
        logger.info(f"WebChat: Starting Flask server on {host}:{port}, Debug: {debug}")
        # use_reloader=False is important when running in a thread or with external debuggers
        self.app.run(host=host, port=port, debug=debug, use_reloader=False)