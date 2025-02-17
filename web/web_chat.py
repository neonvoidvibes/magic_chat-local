from flask import Flask, request, jsonify, render_template, current_app, Response
from typing import Optional
from utils.transcript_utils import TranscriptState, get_latest_transcript_file, read_new_transcript_content, read_all_transcripts_in_folder
from utils.retrieval_handler import RetrievalHandler
import threading
from config import AppConfig
import os
import boto3
import logging
from datetime import datetime
import json
import time

def read_file_content(s3_key, file_name):
    try:
        s3 = boto3.client('s3')
        bucket = 'aiademomagicaudio'
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
        return obj['Body'].read().decode('utf-8')
    except Exception as e:
        logging.error(f"Error reading {file_name} from S3: {e}")
        return ""

def find_file_by_base(base_key, file_name):
    try:
        s3 = boto3.client('s3')
        bucket = 'aiademomagicaudio'
        response = s3.list_objects_v2(Bucket=bucket, Prefix=base_key)
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].startswith(base_key) and obj['Key'] != base_key:
                    return obj['Key']
        logging.error(f"No matching file found in S3 for '{file_name}' with prefix '{base_key}'")
        return None
    except Exception as e:
        logging.error(f"Error finding {file_name} in S3: {e}")
        return None

def get_latest_system_prompt(agent_name=None):
    """Get and combine system prompts from S3"""
    try:
        # Get base system prompt
        base_key = find_file_by_base('_config/systemprompt_base', 'base system prompt')
        if not base_key:
            return None
        base_prompt = read_file_content(base_key, "base system prompt")
        
        # Get agent-specific system prompt if agent name is provided
        agent_prompt = ""
        if agent_name:
            agent_key = find_file_by_base(
                f'organizations/river/agents/{agent_name}/_config/systemprompt_aID-{agent_name}',
                'agent system prompt'
            )
            if agent_key:
                agent_prompt = read_file_content(agent_key, "agent system prompt")
        
        # Combine prompts
        system_prompt = base_prompt
        if agent_prompt:
            system_prompt += "\n\n" + agent_prompt
            
        return system_prompt
    except Exception as e:
        logging.error(f"Error getting system prompts: {e}")
        return None

def get_latest_frameworks(agent_name=None):
    """Get and combine frameworks from S3"""
    try:
        # Get base frameworks
        base_key = find_file_by_base('_config/frameworks_base', 'base frameworks')
        if not base_key:
            return None
        base_frameworks = read_file_content(base_key, "base frameworks")
        
        # Get agent-specific frameworks if agent name is provided
        agent_frameworks = ""
        if agent_name:
            agent_key = find_file_by_base(
                f'organizations/river/agents/{agent_name}/_config/frameworks_aID-{agent_name}',
                'agent frameworks'
            )
            if agent_key:
                agent_frameworks = read_file_content(agent_key, "agent frameworks")
        
        # Combine frameworks
        frameworks = base_frameworks
        if agent_frameworks:
            frameworks += "\n\n" + agent_frameworks
            
        return frameworks
    except Exception as e:
        logging.error(f"Error getting frameworks: {e}")
        return None

def get_latest_context(agent_name, event_id=None):
    """Get and combine contexts from S3"""
    try:
        # Get organization-specific context
        org_key = find_file_by_base(
            f'organizations/river/_config/context_oID-{agent_name}',
            'organization context'
        )
        if not org_key:
            return None
        org_context = read_file_content(org_key, "organization context")
        
        # Get event-specific context if event ID is provided
        event_context = ""
        if event_id:
            event_key = find_file_by_base(
                f'organizations/river/agents/{agent_name}/events/{event_id}/_config/context_aID-{agent_name}_eID-{event_id}',
                'event context'
            )
            if event_key:
                event_context = read_file_content(event_key, "event context")
        
        # Combine contexts
        context = org_context
        if event_context:
            context += "\n\n" + event_context
            
        return context
    except Exception as e:
        logging.error(f"Error getting contexts: {e}")
        return None

def get_agent_docs(agent_name):
    try:
        from magic_chat import get_agent_docs as get_docs
        return get_docs(agent_name)
    except Exception as e:
        logging.error(f"Error getting agent documentation: {e}")
        return None

class WebChat:
    def __init__(self, config: AppConfig):
        self.config = config
        self.app = Flask(__name__)
        self.setup_routes()
        self.chat_history = []
        self.client = None
        self.context = None
        self.frameworks = None
        self.transcript = None
        self.system_prompt = None
        self.retriever = RetrievalHandler(
            index_name="magicchat",
            agent_name=config.agent_name,  # Pass agent name for namespace
            session_id=config.session_id,  # Current session
            event_id=config.event_id      # Current event
        )
        
        # Initialize session ID and chat filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d-T%H%M%S')
        self.config.session_id = timestamp  # Set session ID
        self.current_chat_file = f"chat_D{timestamp}_aID-{config.agent_name}_eID-{config.event_id}.txt"
        
        # Start rolling transcript scheduler
        from scripts.transcript_scheduler import start_scheduler
        self.scheduler_thread = start_scheduler(
            agent_name=config.agent_name,
            session_id=config.session_id,
            event_id=config.event_id
        )
        self.last_saved_index = 0     # Track messages saved via !save command
        self.last_archive_index = 0   # Track messages auto-archived
        logging.debug(f"Initialized chat filename: {self.current_chat_file}")
        
        self.load_resources()
        
    def get_document_context(self, query: str):
        """Get relevant document context for query."""
        try:
            logging.info(f"WebChat: Getting document context for query: {query}")
            
            # Build metadata filters
            filter_metadata = {
                'agent_name': self.config.agent_name,  # Always filter by agent name
            }
            logging.info(f"WebChat: Using agent name filter: {self.config.agent_name}")
            
            # Add event_id filter if available
            if self.config.event_id:
                filter_metadata['event_id'] = self.config.event_id
                logging.info(f"WebChat: Added event ID filter: {self.config.event_id}")
                
            # Determine if this is a transcript query
            is_transcript = any(
                word in query.lower()
                for word in ['transcript', 'conversation', 'meeting', 'session', 'said']
            )
            logging.info(f"WebChat: Is transcript query: {is_transcript}")
            
            # Get relevant contexts using RetrievalHandler
            logging.info(f"WebChat: Calling retriever with metadata filters: {filter_metadata}")
            contexts = self.retriever.get_relevant_context(
                query=query,
                filter_metadata=filter_metadata,
                top_k=3,  # Limit to top 3 most relevant matches
                is_transcript=is_transcript  # Only search transcript namespace if transcript-related
            )
            
            if not contexts:
                logging.info(f"WebChat: No relevant context found for query: {query}")
                return None
                
            # Log retrieval success
            logging.info(f"WebChat: Retrieved {len(contexts)} relevant context chunks")
            for i, context in enumerate(contexts):
                logging.info(f"WebChat: Context {i+1} content: {context.page_content[:100]}...")
            return contexts
            
        except Exception as e:
            logging.error(f"Error retrieving document context: {e}")
            return None

    def load_resources(self):
        """Load context, frameworks, and transcript from S3"""
        # Load and combine system prompts
        system_prompt = get_latest_system_prompt(self.config.agent_name)
        if not system_prompt:
            logging.error("Failed to load system prompt")
            return
            
        # Add source differentiation instructions
        system_prompt += "\n\n## Source Attribution Requirements\n"
        system_prompt += "1. ALWAYS specify the exact source file when quoting or referencing information\n"
        system_prompt += "2. Format source references as: 'From [source file]: [quote]'\n"
        system_prompt += "3. If you cannot determine the exact source file, explicitly state: 'I cannot determine the specific source file for this information'\n"
        system_prompt += "4. When multiple sources contain similar information, list all relevant sources\n"
        system_prompt += "5. Differentiate between:\n"
        system_prompt += "   - [VECTOR DB CONTENT] for historical/stored information\n"
        system_prompt += "   - [LIVE TRANSCRIPT] for real-time updates\n\n"
        system_prompt += "## Data Source Guidelines\n"
        system_prompt += "You have access to two types of information:\n"
        system_prompt += "1. Live Transcript Data: Real-time conversation updates marked with [LIVE TRANSCRIPT]\n"
        system_prompt += "2. Vector Database Knowledge: Historical information marked with [VECTOR DB]\n"
        system_prompt += "\nWhen providing information:\n"
        system_prompt += "- Always specify which source file you are quoting from\n"
        system_prompt += "- When citing content, include both the source type ([VECTOR DB] or [LIVE TRANSCRIPT]) and the specific filename\n"
        system_prompt += "- If you cannot determine the exact source file, explicitly state that\n"
        system_prompt += "- Prioritize live transcript data for current context\n"
        system_prompt += "- Use vector database knowledge for historical context and background\n"
        system_prompt += "- If mixing sources, clearly indicate which parts come from where\n"
        
        # Add frameworks
        frameworks = get_latest_frameworks(self.config.agent_name)
        if frameworks:
            system_prompt += "\n\n## Frameworks\n" + frameworks
        
        # Add context
        context = get_latest_context(self.config.agent_name)  # Note: event_id not implemented yet
        if context:
            system_prompt += "\n\n## Context\n" + context
        
        # Add agent documentation
        docs = get_agent_docs(self.config.agent_name)
        if docs:
            system_prompt += "\n\n## Agent Documentation\n" + docs
        
        # Store the system prompt
        self.system_prompt = system_prompt
        
        # Load memory if enabled
        if self.config.memory is not None:
            self.system_prompt = self.reload_memory()
            if not self.system_prompt:  # If reload_memory fails, revert to original system prompt
                self.system_prompt = system_prompt

        # Load transcript if listening is enabled
        if self.config.listen_transcript:
            self.load_transcript()

    def reload_memory(self):
        """Reload memory from chat history files"""
        # Import the necessary functions
        from magic_chat import load_existing_chats_from_s3, summarize_text
        
        # Make sure we have a valid system prompt
        if not self.system_prompt:
            logging.error("Cannot reload memory: system prompt is not initialized")
            return None
        
        # Load and process chat history
        if not self.config.memory:
            self.config.memory = [self.config.agent_name]
        previous_chats = load_existing_chats_from_s3(self.config.agent_name, self.config.memory)
        
        # Combine all chat content
        all_content = []
        for chat in previous_chats:
            for msg in chat['messages']:
                all_content.append(msg['content'])
        
        combined_content = "\n\n".join(all_content)  # Add extra newline between files
        summarized_content = summarize_text(combined_content, max_length=None)
        
        # Build new system prompt
        if summarized_content:
            new_system_prompt = (
                self.system_prompt + 
                "\n\n## Previous Chat History\nThe following is a summary of previous chat interactions:\n\n" + 
                summarized_content
            )
        else:
            new_system_prompt = self.system_prompt
        
        return new_system_prompt

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html', agent_name=self.config.agent_name)
            
        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            data = request.json
            if not data or 'message' not in data:
                return jsonify({'error': 'No message provided'}), 400
            # Process the message and get response
            try:
                # Initialize Anthropic client if needed
                if not self.client:
                    from anthropic import Anthropic
                    import os
                    self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

                # Add user message to history with timestamp
                current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                user_content = f"On {current_timestamp}, user said: {data['message']}"
                self.chat_history.append({
                    'user': user_content,
                    'assistant': None,
                    'timestamp': current_timestamp
                })
                logging.debug(f"Added user message to history. Total messages: {len(self.chat_history)}")
                # Get relevant document context
                relevant_context = self.get_document_context(data['message'])
                current_system_prompt = self.system_prompt
                if relevant_context:
                    context_text = "\n\n".join(c['content'] for c in relevant_context)
                    current_system_prompt += f"\n\nRelevant context:\n{context_text}"

                # Build conversation history from previous messages
                messages = []
                for chat in self.chat_history:
                    if chat['user']:  # Only add if user message exists
                        messages.append({"role": "user", "content": chat['user']})
                    if chat['assistant']:  # Only add if assistant message exists
                        messages.append({"role": "assistant", "content": chat['assistant']})
                # Add source handling instructions
                source_instructions = """
                Important source handling instructions:
                1. When referencing information, always specify the source:
                   - Use "[LIVE TRANSCRIPT]" for real-time updates
                   - Use "[VECTOR DB]" for historical knowledge
                2. Prioritize live transcript data for current context
                3. Cross-reference vector database knowledge for historical context
                4. When mixing sources, clearly indicate which parts come from where
                """
                
                enhanced_prompt = current_system_prompt + "\n\n" + source_instructions

                def generate():
                    full_response = ""
                    with self.client.messages.stream(
                        model="claude-3-5-sonnet-20241022",
                        # model="claude-3-5-sonnet-20240620",
                        # model="claude-3-5-haiku-20241022",
                        max_tokens=1024,
                        system=enhanced_prompt,
                        messages=messages
                    ) as stream:
                        for text in stream.text_stream:
                            full_response += text
                            yield f"data: {json.dumps({'delta': text})}\n\n"
                    # Update the last message with assistant's response
                    if self.chat_history:
                        self.chat_history[-1]['assistant'] = full_response
                    # Save to archive
                    new_messages = self.chat_history[self.last_archive_index:]
                    if new_messages:
                        chat_content = ""
                        for chat in new_messages:
                            if chat.get('user'):
                                chat_content += f"**User:**\n{chat['user']}\n\n"
                            if chat.get('assistant'):
                                chat_content += f"**Agent:**\n{chat['assistant']}\n\n"
                        from magic_chat import save_chat_to_s3
                        success, _ = save_chat_to_s3(
                            agent_name=self.config.agent_name,
                            chat_content=chat_content,
                            event_id=self.config.event_id,
                            is_saved=False,
                            filename=self.current_chat_file
                        )
                        if success:
                            self.last_archive_index = len(self.chat_history)
                        else:
                            logging.error("Failed to save chat history")
                    yield f"data: {json.dumps({'done': True})}\n\n"
                return Response(generate(), mimetype='text/event-stream')
            except Exception as e:
                logging.error(f"Error in chat endpoint: {e}")
                return jsonify({'error': str(e)}), 500
            
        @self.app.route('/api/status', methods=['GET'])
        def status():
            return jsonify({
                'agent_name': self.config.agent_name,
                'listen_summary': self.config.listen_summary,
                'listen_transcript': self.config.listen_transcript,
                'listen_insights': self.config.listen_insights,
                'memory_enabled': self.config.memory is not None
            })
            
        @self.app.route('/api/command', methods=['POST'])
        def command():
            data = request.json
            if not data or 'command' not in data:
                return jsonify({'error': 'No command provided'}), 400
                
            cmd = data['command'].lower()
            if cmd == 'help':
                help_text = (
                    "Available commands:\n"
                    "!help          - Display this help message\n"
                    "!clear         - Clear the chat history\n"
                    "!save          - Save current chat history to S3 saved folder\n"
                    "!memory        - Toggle memory mode (load chat history)\n"
                    "!listen        - Enable summary listening\n"
                    "!listen-all    - Enable all listening modes\n"
                    "!listen-deep   - Enable summary and insights listening\n"
                    "!listen-insights - Enable insights listening\n"
                    "!listen-transcript - Enable transcript listening"
                )
                return jsonify({'message': help_text})
            elif cmd == 'clear':
                self.chat_history = []
                self.last_saved_index = 0
                return jsonify({'message': 'Chat history cleared'})
            elif cmd == 'save':
                try:
                    # Get new messages since last save
                    new_messages = self.chat_history[self.last_saved_index:]
                    logging.debug(f"Messages to save: {len(new_messages)} (total: {len(self.chat_history)}, last saved: {self.last_saved_index})")
                    
                    if not new_messages:
                        return jsonify({'message': 'No new messages to save'})
                    
                    chat_content = ""
                    for chat in new_messages:
                        timestamp = chat.get('timestamp', '')
                        chat_content += f"**User ({timestamp}):**\n{chat['user']}\n\n"
                        if chat['assistant']:
                            chat_content += f"**Agent:**\n{chat['assistant']}\n\n"
                    
                    # Import the save function from magic_chat
                    from magic_chat import save_chat_to_s3
                    
                    success, filename = save_chat_to_s3(
                        agent_name=self.config.agent_name,
                        chat_content=chat_content,
                        event_id=self.config.event_id,
                        is_saved=True,
                        filename=self.current_chat_file
                    )
                    
                    if success:
                        self.last_saved_index = len(self.chat_history)  # Update save index only
                        return jsonify({'message': f'Chat history saved successfully as {filename}'})
                    else:
                        return jsonify({'error': 'Failed to save chat history'})
                except Exception as e:
                    logging.error(f"Error saving chat history: {e}")
                    return jsonify({'error': f'Error saving chat history: {str(e)}'})
            elif cmd == 'memory':
                if self.config.memory is None:
                    self.config.memory = [self.config.agent_name]
                    self.system_prompt = self.reload_memory()
                    if not self.system_prompt:  # If reload_memory fails, revert to original system prompt
                        self.system_prompt = system_prompt
                    return jsonify({'message': 'Memory mode activated'})
                else:
                    self.config.memory = None
                    return jsonify({'message': 'Memory mode deactivated'})
            elif cmd == 'listen':
                self.config.listen_summary = True
                if self.load_transcript():
                    return jsonify({'message': 'Listening to summaries activated and transcript loaded'})
                else:
                    return jsonify({'message': 'Listening to summaries activated (no transcript found)'})
            elif cmd == 'listen-transcript':
                self.config.listen_transcript = True
                if self.load_transcript():
                    return jsonify({'message': 'Transcript loaded and listening mode activated'})
                else:
                    return jsonify({'message': 'No transcript files found'})
            elif cmd == 'listen-insights':
                self.config.listen_insights = True
                return jsonify({'message': 'Listening to insights activated'})
            elif cmd == 'listen-all':
                self.config.listen_summary = True
                self.config.listen_transcript = True
                self.config.listen_insights = True
                self.config.listen_all = True
                if self.load_transcript():
                    return jsonify({'message': 'All listening modes activated and transcript loaded'})
                else:
                    return jsonify({'message': 'All listening modes activated (no transcript found)'})
            elif cmd == 'listen-deep':
                self.config.listen_summary = True
                self.config.listen_insights = True
                self.config.listen_deep = True
                return jsonify({'message': 'Deep listening mode activated'})
            else:
                return jsonify({'error': 'Unknown command'}), 400
            
        @self.app.route('/api/save', methods=['POST'])
        def save_chat():
            """Copy current chat file from archive to saved folder"""
            try:
                if not self.current_chat_file:
                    return jsonify({'error': 'No chat file exists to save'}), 404
                
                # Import the save function from magic_chat
                from magic_chat import save_chat_to_s3
                
                # Copy the current chat file from archive to saved
                success, filename = save_chat_to_s3(
                    agent_name=self.config.agent_name,
                    chat_content="",  # Empty content since we're just copying
                    event_id=self.config.event_id,
                    is_saved=True,
                    filename=self.current_chat_file
                )
                if success:
                    return jsonify({'message': 'Chat history saved successfully'}), 200
                else:
                    return jsonify({'error': 'Failed to save chat history'}), 500
            except Exception as e:
                logging.error(f"Error saving chat history: {e}")
                return jsonify({'error': f'Error saving chat history: {str(e)}'}), 500

    def load_transcript(self):
        """Load transcript(s) from the agent's event folder"""
        try:
            if self.config.read_all:
                # Load and append all transcripts at once
                all_content = read_all_transcripts_in_folder(self.config.agent_name, self.config.event_id)
                if all_content:
                    logging.debug(f"Loaded all transcripts, total length: {len(all_content)}")
                    self.transcript = all_content
                    self.system_prompt += f"\\n\\nTranscript update (all): {all_content}"
                    logging.debug(f"Updated system prompt, new length: {len(self.system_prompt)}")
                else:
                    logging.debug("No transcripts found in folder (or error).")
                # keep rolling updates
            else:
                
                # Get latest original transcript and derive rolling key
                s3 = boto3.client('s3')
                base_key = get_latest_transcript_file(self.config.agent_name, self.config.event_id)
                base_path = os.path.dirname(base_key)
                filename = os.path.basename(base_key)
                # Prepend rolling- to the filename
                rolling_key = f"{base_path}/rolling-{filename}"
                
                try:
                    # Try to get rolling transcript
                    transcript_obj = s3.get_object(Bucket=self.config.aws_s3_bucket, Key=rolling_key)
                    transcript = transcript_obj['Body'].read().decode('utf-8')
                    if transcript:
                        logging.debug(f"Loaded rolling transcript from {rolling_key}, length: {len(transcript)}")
                        self.transcript = transcript
                        self.system_prompt += f"\n\nTranscript update: {transcript}"
                        return
                except Exception as e:
                    logging.warning(f"Rolling transcript not found, falling back to original: {e}")
                
                # Fallback to original transcript if rolling not found
                if base_key:
                    logging.debug(f"Using original transcript: {base_key}")
                    transcript_obj = s3.get_object(Bucket=self.config.aws_s3_bucket, Key=base_key)
                    transcript = transcript_obj['Body'].read().decode('utf-8')
                    if transcript:
                        logging.debug(f"Loaded original transcript, length: {len(transcript)}")
                        self.transcript = transcript
                        self.system_prompt += f"\n\nTranscript update: {transcript}"
                        logging.debug(f"Updated system prompt, new length: {len(self.system_prompt)}")
                        return True
            
            return False
            
        except Exception as e:
            logging.error(f"Error loading transcript from S3: {e}")
            return False

    def check_transcript_updates(self):
        """Check for new transcript updates"""
        logging.debug("Checking for transcript updates...")
        
        try:
            # First check if we actually have the state and config we need
            if not hasattr(self, 'transcript_state') or not self.transcript_state:
                self.transcript_state = TranscriptState()
                
            new_content = read_new_transcript_content(
                self.transcript_state,
                self.config.agent_name,
                self.config.event_id
            )
            
            if new_content:
                # Add clear source labeling to transcript updates
                labeled_content = f"[LIVE TRANSCRIPT] {new_content}"
                self.chat_history.append({
                    'user': f"[Transcript update]: {labeled_content}",
                    'assistant': None
                })
                return True
                    
            return False
            
        except Exception as e:
            logging.error(f"Error checking transcript updates: {e}")
            return False

    def run(self, host: str = '127.0.0.1', port: int = 5001, debug: bool = False):
        def check_updates():
            while True:
                if self.config.listen_transcript:
                    self.check_transcript_updates()
                time.sleep(5)  # Same 5-second interval as CLI

        if self.config.listen_transcript:
            from magic_chat import TranscriptState
            self.transcript_state = TranscriptState()
            threading.Thread(target=check_updates, daemon=True).start()

        if self.config.interface_mode == 'web_only':
            self.app.run(host=host, port=port, debug=debug)
        else:
            thread = threading.Thread(
                target=lambda: self.app.run(host=host, port=port, debug=debug, use_reloader=False),
                daemon=True
            )
            thread.start()
            return thread