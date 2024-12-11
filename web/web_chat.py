from flask import Flask, request, jsonify, render_template, current_app, Response
from typing import Optional
import threading
from config import AppConfig
import os
import boto3
import logging
from datetime import datetime
import json

def read_file_content_local(file_path, file_name):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error reading {file_name} from local file: {e}")
        return ""

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
    try:
        s3 = boto3.client('s3')
        bucket = 'aiademomagicaudio'
        
        # Get agent-specific system prompt
        if agent_name:
            prefix = f'agents/{agent_name}/system-prompt/'
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            if 'Contents' in response:
                prompt_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.md')]
                if prompt_files:
                    latest_file = max(prompt_files, key=lambda x: x['LastModified'])
                    return latest_file['Key']
        
        # Get standard system prompt
        prefix = 'system-prompt/'
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' in response:
            prompt_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.md')]
            if prompt_files:
                latest_file = max(prompt_files, key=lambda x: x['LastModified'])
                return latest_file['Key']
        
        return None
    except Exception as e:
        logging.error(f"Error getting latest system prompt from S3: {e}")
        return None

def get_latest_context(agent_name):
    """Get the latest context file from S3"""
    try:
        s3 = boto3.client('s3')
        bucket = 'aiademomagicaudio'
        prefix = f'agents/{agent_name}/context/'
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' in response:
            context_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.txt')]
            if context_files:
                latest_file = max(context_files, key=lambda x: x['LastModified'])
                return latest_file['Key']
        return None
    except Exception as e:
        logging.error(f"Error getting latest context file: {e}")
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
        s3 = boto3.client('s3')
        bucket = 'aiademomagicaudio'
        
        # Load standard and unique system prompts from S3
        standard_prompt_key = get_latest_system_prompt()
        unique_prompt_key = get_latest_system_prompt(self.config.agent_name)
        
        if not standard_prompt_key or not unique_prompt_key:
            logging.error("Failed to find system prompt files in S3")
            return
            
        standard_system_prompt = read_file_content(standard_prompt_key, "standard system prompt")
        unique_system_prompt = read_file_content(unique_prompt_key, "unique system prompt")
        self.system_prompt = standard_system_prompt + "\n" + unique_system_prompt
        
        # Load memory if enabled
        if self.config.memory is not None:
            self.system_prompt = self.reload_memory()
        else:
            self.system_prompt = initial_system_prompt

        # Load frameworks
        try:
            frameworks_obj = s3.get_object(Bucket=bucket, Key='frameworks/frameworks.txt')
            self.frameworks = frameworks_obj['Body'].read().decode('utf-8')
            if self.frameworks:
                self.system_prompt += f"\nFrameworks:\n{self.frameworks}"
        except Exception as e:
            logging.error(f"Error loading frameworks from S3: {e}")
        
        # Load context
        try:
            context_key = get_latest_context(self.config.agent_name)
            if context_key:
                context_obj = s3.get_object(Bucket=bucket, Key=context_key)
                self.context = context_obj['Body'].read().decode('utf-8')
                if self.context:
                    self.system_prompt += f"\nContext:\n{self.context}"
        except Exception as e:
            logging.error(f"Error loading context from S3: {e}")
        
        # Load transcript if listening is enabled
        if self.config.listen_transcript:
            self.load_transcript()
        
    def load_transcript(self):
        """Load latest transcript from agent's transcript directory"""
        try:
            s3 = boto3.client('s3')
            prefix = f'agents/{self.config.agent_name}/transcripts/'
            response = s3.list_objects_v2(Bucket=self.config.aws_s3_bucket, Prefix=prefix)
            
            if 'Contents' in response:
                transcript_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.txt')]
                if transcript_files:
                    latest_file = max(transcript_files, key=lambda x: x['LastModified'])
                    transcript_obj = s3.get_object(Bucket=self.config.aws_s3_bucket, Key=latest_file['Key'])
                    transcript = transcript_obj['Body'].read().decode('utf-8')
                    if transcript:
                        self.transcript = transcript
                        self.system_prompt += f"\n\nTranscript update: {transcript}"
                        return True
            
            # Fallback to root transcript directory
            response = s3.list_objects_v2(Bucket=self.config.aws_s3_bucket, Prefix='transcript_')
            if 'Contents' in response:
                transcript_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.txt')]
                if transcript_files:
                    latest_file = max(transcript_files, key=lambda x: x['LastModified'])
                    transcript_obj = s3.get_object(Bucket=self.config.aws_s3_bucket, Key=latest_file['Key'])
                    transcript = transcript_obj['Body'].read().decode('utf-8')
                    if transcript:
                        self.transcript = transcript
                        self.system_prompt += f"\n\nTranscript update: {transcript}"
                        return True
            
            return False
        except Exception as e:
            logging.error(f"Error loading transcript from S3: {e}")
            return False

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
        standard_system_prompt = read_file_content(get_latest_system_prompt(), "standard system prompt")
        unique_system_prompt = read_file_content(get_latest_system_prompt(self.config.agent_name), "unique system prompt")
        initial_system_prompt = standard_system_prompt + "\n" + unique_system_prompt
        
        # Add the summarized content to the system prompt with clear context
        if summarized_content:
            new_system_prompt = (
                initial_system_prompt + 
                "\n\n## Previous Chat History\nThe following is a summary of previous chat interactions:\n\n" + 
                summarized_content
            )
        else:
            new_system_prompt = initial_system_prompt
        
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
                    
                    # Save chat history to S3 archive folder
                    try:
                        chat_content = ""
                        for chat in self.chat_history:
                            chat_content += f"**User:**\n{chat['user']}\n\n"
                            chat_content += f"**Agent:**\n{chat['assistant']}\n\n"
                        
                        s3 = boto3.client('s3')
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"chat_{self.config.agent_name}_{timestamp}.txt"
                        s3_key = f"agents/{self.config.agent_name}/chat_history/archive/{filename}"
                        
                        s3.put_object(
                            Bucket=self.config.aws_s3_bucket,
                            Key=s3_key,
                            Body=chat_content.encode('utf-8')
                        )
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
                    
                    s3 = boto3.client('s3')
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"chat_{self.config.agent_name}_{timestamp}.txt"
                    s3_key = f"agents/{self.config.agent_name}/chat_history/saved/{filename}"
                    
                    s3.put_object(
                        Bucket=self.config.aws_s3_bucket,
                        Key=s3_key,
                        Body=chat_content.encode('utf-8')
                    )
                    
                    return jsonify({'message': 'Chat history saved successfully'})
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
                
                s3 = boto3.client('s3')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"chat_{self.config.agent_name}_{timestamp}.txt"
                s3_key = f"agents/{self.config.agent_name}/chat_history/saved/{filename}"
                
                s3.put_object(
                    Bucket=self.config.aws_s3_bucket,
                    Key=s3_key,
                    Body=chat_content.encode('utf-8')
                )
                
                return jsonify({'message': 'Chat history saved successfully'})
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
