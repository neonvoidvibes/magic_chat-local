from dataclasses import dataclass
from typing import List, Dict, Optional
import os
import argparse
from dotenv import load_dotenv

@dataclass
class AppConfig:
    """Configuration class that holds all application settings"""
    # Core settings
    agent_name: str
    interface_mode: str  # 'cli', 'web', or 'web_only'
    web_port: int = 5001  # Default web port
    
    # Optional settings with defaults
    memory: List[str] = None
    debug: bool = False
    
    # Listener settings
    listen_summary: bool = False
    listen_transcript: bool = False
    listen_insights: bool = False
    listen_deep: bool = False
    listen_all: bool = False
    listen_transcript_enabled: bool = False  # Track if transcript listening is currently enabled

    # Environment settings
    aws_region: Optional[str] = None
    aws_s3_bucket: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    @classmethod
    def from_env_and_args(cls) -> 'AppConfig':
        """Create configuration from environment variables and command line arguments"""
        # Load environment variables
        load_dotenv()
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Run a Claude agent instance.")
        parser.add_argument('--agent', required=True, help='Unique name for the agent.')
        parser.add_argument('--memory', nargs='*', help='Names of agents to load chat history from.', default=None)
        parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
        parser.add_argument('--web', action='store_true', help='Run with web interface.')
        parser.add_argument('--web-only', action='store_true', help='Run with web interface only (no CLI fallback).')
        parser.add_argument('--web-port', type=int, default=5001, help='Port for web interface (default: 5001)')
        
        # Listener arguments
        parser.add_argument('--listen', action='store_true', help='Enable summary listening at startup.')
        parser.add_argument('--listen-transcript', action='store_true', help='Enable transcript listening at startup.')
        parser.add_argument('--listen-insights', action='store_true', help='Enable insights listening at startup.')
        parser.add_argument('--listen-deep', action='store_true', help='Enable summary and insights listening at startup.')
        parser.add_argument('--listen-all', action='store_true', help='Enable all listening at startup.')
        
        args = parser.parse_args()
        
        # Determine interface mode
        if args.web_only:
            interface_mode = 'web_only'
        elif args.web:
            interface_mode = 'web'
        else:
            interface_mode = 'cli'

        # Process listener flags
        listen_summary = args.listen or args.listen_deep or args.listen_all
        listen_transcript = args.listen_transcript or args.listen_all
        listen_insights = args.listen_insights or args.listen_deep or args.listen_all
        
        # Create config instance
        config = cls(
            agent_name=args.agent,
            interface_mode=interface_mode,
            web_port=args.web_port,
            memory=args.memory,
            debug=args.debug,
            listen_summary=listen_summary,
            listen_transcript=listen_transcript,  # Set from command line arg
            listen_insights=listen_insights,
            listen_deep=args.listen_deep,
            listen_all=args.listen_all,
            listen_transcript_enabled=False,  # Always start disabled, enable only when needed
            # Environment variables
            aws_region=os.getenv('AWS_REGION'),
            aws_s3_bucket=os.getenv('AWS_S3_BUCKET'),
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Validate configuration
        config.validate()
        return config
    
    def validate(self) -> None:
        """Validate the configuration"""
        missing_vars = []
        
        # Check required environment variables
        if not self.aws_region:
            missing_vars.append('AWS_REGION')
        if not self.aws_s3_bucket:
            missing_vars.append('AWS_S3_BUCKET')
        if not self.anthropic_api_key:
            missing_vars.append('ANTHROPIC_API_KEY')
        if not self.openai_api_key:
            missing_vars.append('OPENAI_API_KEY')
            
        if missing_vars:
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
        
        # Validate interface mode
        valid_modes = {'cli', 'web', 'web_only'}
        if self.interface_mode not in valid_modes:
            raise ValueError(f"Invalid interface mode: {self.interface_mode}")
