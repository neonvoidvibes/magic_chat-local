from dataclasses import dataclass, field
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
    memory: List[str] = field(default_factory=list) # Use field for mutable default
    debug: bool = False

    # Listener settings
    listen_summary: bool = False
    listen_transcript: bool = False
    listen_insights: bool = False
    listen_deep: bool = False
    listen_all: bool = False
    listen_transcript_enabled: bool = False  # Track if transcript listening is currently enabled
    read_all: bool = False                  # If True, read all transcripts in event folder at once

    # Environment settings
    aws_region: Optional[str] = None
    aws_s3_bucket: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    event_id: str = '0000'  # Default event ID
    session_id: str = None  # Will be set to timestamp on initialization

    # Pinecone index name
    index: str = "magicchat"

    # LLM Model Name (Centralized Definition)
    llm_model_name: str = "claude-3-7-sonnet-20250219" # Default model

    # LLM Max Output Tokens
    llm_max_output_tokens: int = 4096 # Default max tokens for LLM response

    @classmethod
    def from_env_and_args(cls) -> 'AppConfig':
        """Create configuration from environment variables and command line arguments"""
        load_dotenv() # Ensure .env is loaded early

        parser = argparse.ArgumentParser(description="Run a Claude agent instance.")
        parser.add_argument('--agent', required=True, help='Unique name for the agent.')
        parser.add_argument('--memory', nargs='*', help='Names of agents to load chat history from.', default=None)
        parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
        parser.add_argument('--web', action='store_true', help='Run with web interface.')
        parser.add_argument('--web-only', action='store_true', help='Run with web interface only (no CLI fallback).')
        parser.add_argument('--web-port', type=int, default=5001, help='Port for web interface (default: 5001)')
        parser.add_argument('--listen', action='store_true', help='Enable summary listening at startup.')
        parser.add_argument('--listen-transcript', action='store_true', help='Enable transcript listening at startup.')
        parser.add_argument('--listen-insights', action='store_true', help='Enable insights listening at startup.')
        parser.add_argument('--listen-deep', action='store_true', help='Enable summary and insights listening at startup.')
        parser.add_argument('--listen-all', action='store_true', help='Enable all listening at startup.')
        parser.add_argument('--all', action='store_true', help='Read all transcripts in the selected folder at launch, ignore further updates.')
        parser.add_argument('--event', type=str, default='0000', help='Event ID (default: 0000)')
        parser.add_argument('--index', type=str, default='magicchat', help='Pinecone index name to fetch context from')
        # Add optional CLI override for max tokens
        parser.add_argument('--max-tokens', type=int, help='Override the default max output tokens for the LLM')

        args = parser.parse_args()

        # Determine interface mode
        if args.web_only: interface_mode = 'web_only'
        elif args.web: interface_mode = 'web'
        else: interface_mode = 'cli'

        # Process listener flags
        listen_summary = args.listen or args.listen_deep or args.listen_all
        listen_transcript = args.listen_transcript or args.listen_all
        listen_insights = args.listen_insights or args.listen_deep or args.listen_all
        memory_agents = args.memory if args.memory is not None else []

        # Determine max tokens: CLI arg > Env Var > Class Default
        default_max_tokens = cls.llm_max_output_tokens # Get class default
        max_tokens_from_env = os.getenv('LLM_MAX_OUTPUT_TOKENS')
        try:
            llm_max_tokens = int(args.max_tokens if args.max_tokens is not None else max_tokens_from_env or default_max_tokens)
        except (ValueError, TypeError):
            print(f"Warning: Invalid value for max_tokens ('{args.max_tokens or max_tokens_from_env}'). Using default: {default_max_tokens}", file=sys.stderr)
            llm_max_tokens = default_max_tokens


        config = cls(
            agent_name=args.agent,
            interface_mode=interface_mode,
            web_port=args.web_port,
            event_id=args.event,
            memory=memory_agents,
            debug=args.debug,
            listen_summary=listen_summary,
            listen_transcript=listen_transcript,
            listen_insights=listen_insights,
            listen_deep=args.listen_deep,
            listen_all=args.listen_all,
            listen_transcript_enabled=False,
            read_all=args.all,
            aws_region=os.getenv('AWS_REGION'),
            aws_s3_bucket=os.getenv('AWS_S3_BUCKET'),
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            index=args.index,
            llm_model_name="claude-3-7-sonnet-20250219", # Keep this hardcoded for now, or add CLI arg
            llm_max_output_tokens=llm_max_tokens # Use the determined value
        )

        config.validate()
        return config

    def validate(self) -> None:
        """Validate the configuration"""
        missing_vars = []
        if not self.aws_region: missing_vars.append('AWS_REGION')
        if not self.aws_s3_bucket: missing_vars.append('AWS_S3_BUCKET')
        if not self.anthropic_api_key: missing_vars.append('ANTHROPIC_API_KEY')
        if not self.openai_api_key: missing_vars.append('OPENAI_API_KEY')

        if missing_vars: raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
        if self.interface_mode not in {'cli', 'web', 'web_only'}: raise ValueError(f"Invalid interface mode: {self.interface_mode}")
        if not isinstance(self.llm_model_name, str) or not self.llm_model_name: raise ValueError("LLM model name must be a non-empty string")
        if not isinstance(self.llm_max_output_tokens, int) or self.llm_max_output_tokens <= 0: raise ValueError("LLM max output tokens must be a positive integer")