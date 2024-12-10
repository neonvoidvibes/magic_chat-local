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
        
        # Load standard and unique system prompts
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        standard_prompt_file = os.path.join(script_dir, 'system_prompt_standard.txt')
        unique_prompt_file = os.path.join(script_dir, f'system_prompt_{self.config.agent_name}.txt')
        
        standard_system_prompt = read_file_content_local(standard_prompt_file, "standard system prompt")
        unique_system_prompt = read_file_content_local(unique_prompt_file, "unique system prompt")
        self.system_prompt = standard_system_prompt + "\n" + unique_system_prompt
        
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
            context_obj = s3.get_object(Bucket=bucket, Key='context/context_River.txt')
            self.context = context_obj['Body'].read().decode('utf-8')
            if self.context:
                self.system_prompt += f"\nContext:\n{self.context}"
        except Exception as e:
            logging.error(f"Error loading context from S3: {e}")
        
        # Load transcript if listening is enabled
        if self.config.listen_transcript:
            try:
                response = s3.list_objects_v2(Bucket=bucket, Prefix='transcript_')
                if 'Contents' in response:
                    transcript_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.txt')]
                    if transcript_files:
                        latest_file = max(transcript_files, key=lambda x: x['LastModified'])
                        transcript_obj = s3.get_object(Bucket=bucket, Key=latest_file['Key'])
                        self.transcript = transcript_obj['Body'].read().decode('utf-8')
                        if self.transcript:
                            self.system_prompt += f"\nTranscript:\n{self.transcript}"
            except Exception as e:
                logging.error(f"Error loading transcript from S3: {e}")
        
    def reload_memory(self):
        """Reload memory from chat history files"""
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        chat_history_folder = os.path.join(script_dir, 'chat_history')
        
        # Import the necessary functions
        from magic_chat import load_existing_chats, generate_summary_of_chats, summarize_text
        
        # Load and process chat history
        previous_chats = load_existing_chats(chat_history_folder, self.config.agent_name, self.config.memory)
        chat_summary = generate_summary_of_chats(previous_chats)
        summarized_chat = summarize_text(chat_summary, max_length=None)
        
        # Extract insights
        insights = []
        for chat in previous_chats:
            for msg in chat['messages']:
                if msg['role'] == 'assistant' and msg['content'].startswith("[Insights]"):
                    insights.append(msg['content'].replace("[Insights] ", ""))
        
        insights_summary = "\n".join(insights) if insights else ""
        
        # Build new system prompt
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        standard_prompt_file = os.path.join(script_dir, 'system_prompt_standard.txt')
        unique_prompt_file = os.path.join(script_dir, f'system_prompt_{self.config.agent_name}.txt')
        
        standard_system_prompt = read_file_content_local(standard_prompt_file, "standard system prompt")
        unique_system_prompt = read_file_content_local(unique_prompt_file, "unique system prompt")
        initial_system_prompt = standard_system_prompt + "\n" + unique_system_prompt
        
        if insights_summary:
            new_system_prompt = (
                initial_system_prompt +
                "\nSummary of past conversations:\n" + summarized_chat +
                "\n\nLatest Insights:\n" + insights_summary
            )
        else:
            new_system_prompt = initial_system_prompt + "\nSummary of past conversations:\n" + summarized_chat
        
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
                return jsonify({'message': 'Listening to summaries activated'})
            elif cmd == 'listen-transcript':
                self.config.listen_transcript = True
                return jsonify({'message': 'Listening to transcripts activated'})
            elif cmd == 'listen-insights':
                self.config.listen_insights = True
                return jsonify({'message': 'Listening to insights activated'})
            elif cmd == 'listen-all':
                self.config.listen_summary = True
                self.config.listen_transcript = True
                self.config.listen_insights = True
                self.config.listen_all = True
                return jsonify({'message': 'All listening modes activated'})
            elif cmd == 'listen-deep':
                self.config.listen_summary = True
                self.config.listen_insights = True
                self.config.listen_deep = True
                return jsonify({'message': 'Deep listening mode activated'})
            else:
                return jsonify({'error': 'Unknown command'}), 400
            
        @self.app.route('/api/save', methods=['POST'])
        def save_chat():
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
                
                return jsonify({'message': 'Chat history saved successfully', 'file': filename})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
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
