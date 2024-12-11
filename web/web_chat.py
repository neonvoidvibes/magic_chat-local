from flask import Flask, request, jsonify, render_template, current_app, Response
from typing import Optional
import threading
from config import AppConfig
import os
import boto3
import logging
from datetime import datetime
import json

def read_file_content(s3_key, file_name):
    try:
        s3 = boto3.client('s3')
        bucket = 'aiademomagicaudio'
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
        return obj['Body'].read().decode('utf-8')
    except Exception as e:
        logging.error(f"Error reading {file_name} from S3: {e}")
        return ""

def get_latest_system_prompt(agent_name=None):
    """Get and combine system prompts from S3"""
    try:
        # Get base system prompt
        base_prompt = read_file_content('_config/systemprompt_base.md', "base system prompt")
        
        # Get agent-specific system prompt if agent name is provided
        agent_prompt = ""
        if agent_name:
            agent_prompt_key = f'organizations/river/agents/{agent_name}/_config/systemprompt_aID-{agent_name}.md'
            agent_prompt = read_file_content(agent_prompt_key, "agent system prompt")
        
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
        base_frameworks = read_file_content('_config/frameworks_base.md', "base frameworks")
        
        # Get agent-specific frameworks if agent name is provided
        agent_frameworks = ""
        if agent_name:
            agent_frameworks_key = f'organizations/river/agents/{agent_name}/_config/frameworks_aID-{agent_name}.md'
            agent_frameworks = read_file_content(agent_frameworks_key, "agent frameworks")
        
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
        org_context = read_file_content(f'organizations/river/_config/context_oID-{agent_name}.md', "organization context")
        
        # Get event-specific context if event ID is provided
        event_context = ""
        if event_id:
            event_context_key = f'organizations/river/agents/{agent_name}/events/{event_id}/_config/context_aID-{agent_name}_eID-{event_id}.md'
            event_context = read_file_content(event_context_key, "event context")
        
        # Combine contexts
        context = org_context
        if event_context:
            context += "\n\n" + event_context
            
        return context
    except Exception as e:
        logging.error(f"Error getting contexts: {e}")
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
        self.load_resources()
        
    def load_resources(self):
        """Load context, frameworks, and transcript from S3"""
        # Load and combine system prompts
        system_prompt = get_latest_system_prompt(self.config.agent_name)
        if not system_prompt:
            logging.error("Failed to load system prompt")
            return
            
        # Add frameworks
        frameworks = get_latest_frameworks(self.config.agent_name)
        if frameworks:
            system_prompt += "\n\n## Frameworks\n" + frameworks
            
        # Add context
        context = get_latest_context(self.config.agent_name)  # Note: event_id not implemented yet
        if context:
            system_prompt += "\n\n## Context\n" + context
        
        # Load memory if enabled
        if self.config.memory is not None:
            self.system_prompt = self.reload_memory()
        else:
            self.system_prompt = system_prompt

        # Load transcript if listening is enabled
        if self.config.listen_transcript:
            self.load_transcript()

    def reload_memory(self):
        """Reload memory from chat history files"""
        # Import the necessary functions
        from magic_chat import load_existing_chats_from_s3, summarize_text
        
        # Load and process chat history
        previous_chats = load_existing_chats_from_s3(self.config.agent_name, self.config.memory)
        
        # Combine all chat content
        all_content = []
        for chat in previous_chats:
            for msg in chat['messages']:
                all_content.append(msg['content'])
        
        combined_content = "\n\n".join(all_content)  # Add extra newline between files
        summarized_content = summarize_text(combined_content, max_length=None)
        
        # Build new system prompt
        system_prompt = self.system_prompt
        
        if summarized_content:
            new_system_prompt = (
                system_prompt + 
                "\n\n## Previous Chat History\nThe following is a summary of previous chat interactions:\n\n" + 
                summarized_content
            )
        else:
            new_system_prompt = system_prompt
        
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
                
            # Initialize Anthropic client if needed
            if not self.client:
                from anthropic import Anthropic
                import os
                self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            
            # Process the chat message
            try:
                current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                user_content = f"On {current_timestamp}, user said: {data['message']}"
                
                # Build conversation history from previous messages
                messages = []
                for chat in self.chat_history:
                    messages.append({"role": "user", "content": chat['user']})
                    messages.append({"role": "assistant", "content": chat['assistant']})
                messages.append({"role": "user", "content": user_content})
                
                def generate():
                    full_response = ""
                    with self.client.messages.stream(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=1024,
                        system=self.system_prompt,
                        messages=messages
                    ) as stream:
                        for text in stream.text_stream:
                            full_response += text
                            yield f"data: {json.dumps({'delta': text})}\n\n"
                    
                    # Store in chat history after completion
                    self.chat_history.append({
                        'user': user_content,
                        'assistant': full_response
                    })
                    
                    # Save chat history to archive folder
                    try:
                        chat_content = ""
                        for chat in self.chat_history:
                            chat_content += f"**User:**\n{chat['user']}\n\n"
                            chat_content += f"**Agent:**\n{chat['assistant']}\n\n"
                        
                        # Import the save function from magic_chat
                        from magic_chat import save_chat_to_s3
                        save_chat_to_s3(self.config.agent_name, chat_content, is_saved=False)
                    except Exception as e:
                        logging.error(f"Error saving chat history to S3: {e}")
                    
                    yield f"data: {json.dumps({'done': True})}\n\n"
                
                return Response(generate(), mimetype='text/event-stream')
            except Exception as e:
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
                return jsonify({'message': 'Chat history cleared'})
            elif cmd == 'save':
                try:
                    chat_content = ""
                    for chat in self.chat_history:
                        chat_content += f"**User:**\n{chat['user']}\n\n"
                        chat_content += f"**Agent:**\n{chat['assistant']}\n\n"
                    
                    # Import the save function from magic_chat
                    from magic_chat import save_chat_to_s3
                    
                    success, filename = save_chat_to_s3(self.config.agent_name, chat_content, is_saved=True)
                    if success:
                        return jsonify({'message': 'Chat history saved successfully', 'file': filename})
                    else:
                        return jsonify({'error': 'Failed to save chat history'})
                except Exception as e:
                    return jsonify({'error': f'Error saving chat history: {str(e)}'})
            elif cmd == 'memory':
                if self.config.memory is None:
                    self.config.memory = [self.config.agent_name]
                    self.system_prompt = self.reload_memory()
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
            """Save current chat history to a file"""
            try:
                chat_content = ""
                for chat in self.chat_history:
                    chat_content += f"**User:**\n{chat['user']}\n\n"
                    chat_content += f"**Agent:**\n{chat['assistant']}\n\n"
                
                # Import the save function from magic_chat
                from magic_chat import save_chat_to_s3
                
                success, filename = save_chat_to_s3(self.config.agent_name, chat_content, is_saved=True)
                if success:
                    return jsonify({'message': 'Chat history saved successfully', 'file': filename})
                else:
                    return jsonify({'error': 'Failed to save chat history'})
            except Exception as e:
                return jsonify({'error': f'Error saving chat history: {str(e)}'})

    def run(self, host: str = '127.0.0.1', port: int = 5001, debug: bool = False):
        if self.config.interface_mode == 'web_only':
            self.app.run(host=host, port=port, debug=debug)
        else:
            # Run in a separate thread if we're also running CLI
            # Disable reloader when running in a thread to avoid signal handling issues
            thread = threading.Thread(
                target=lambda: self.app.run(host=host, port=port, debug=debug, use_reloader=False),
                daemon=True
            )
            thread.start()
            return thread
