from flask import Flask, request, jsonify, render_template, current_app
from typing import Optional
import threading
from config import AppConfig

class WebChat:
    def __init__(self, config: AppConfig):
        self.config = config
        self.app = Flask(__name__)
        self.setup_routes()
        self.chat_history = []
        self.client = None  # Will be initialized when needed
        
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
                message = self.client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": data['message']}]
                )
                response = message.content[0].text
                
                # Store in chat history
                self.chat_history.append({
                    'user': data['message'],
                    'assistant': response
                })
                
                return jsonify({'response': response})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
            
        @self.app.route('/api/status', methods=['GET'])
        def status():
            return jsonify({
                'agent_name': self.config.agent_name,
                'listen_summary': self.config.listen_summary,
                'listen_transcript': self.config.listen_transcript,
                'listen_insights': self.config.listen_insights
            })
            
        @self.app.route('/api/command', methods=['POST'])
        def command():
            data = request.json
            if not data or 'command' not in data:
                return jsonify({'error': 'No command provided'}), 400
                
            cmd = data['command'].lower()
            if cmd == 'listen':
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
