"""Rolling transcript management and vector embedding processing."""
import os
import boto3
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List
from dateutil import parser

from .document_handler import DocumentHandler
from .embedding_handler import EmbeddingHandler
from .transcript_utils import get_latest_transcript_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RollingTranscriptManager:
    """Manages a rolling transcript file with vector embedding processing."""
    
    def __init__(
        self,
        agent_name: str,
        session_id: str,
        event_id: str,
        s3_bucket: Optional[str] = None,
        s3_client: Optional[boto3.client] = None,
        live_window_minutes: int = 4,
        total_window_minutes: int = 6
    ):
        """Initialize the rolling transcript manager.
        
        Args:
            agent_name: Name of the agent
            session_id: Current session ID
            event_id: Current event ID
            s3_bucket: S3 bucket name (defaults to AWS_S3_BUCKET env var)
            s3_client: Optional pre-configured S3 client
            live_window_minutes: Minutes to keep in live text only (default 4)
            total_window_minutes: Total minutes to maintain in rolling file (default 6)
        """
        self.agent_name = agent_name
        self.session_id = session_id
        self.event_id = event_id
        self.live_window = timedelta(minutes=live_window_minutes)
        self.total_window = timedelta(minutes=total_window_minutes)
        
        # Initialize S3 client if not provided
        if s3_client is None:
            self.s3_client = boto3.client(
                's3',
                region_name=os.getenv('AWS_REGION'),
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
        else:
            self.s3_client = s3_client
            
        self.s3_bucket = s3_bucket or os.getenv('AWS_S3_BUCKET')
        
        # Initialize handlers
        self.doc_handler = DocumentHandler(chunk_size=500, chunk_overlap=100)
        # Use {agent}-{event} as namespace
        namespace = f"{agent_name}-{event_id}" if event_id else f"{agent_name}-0000"
        self.embed_handler = EmbeddingHandler(
            index_name="magicchat",
            namespace=namespace
        )
        
        # Get the folder path where transcripts are stored
        if agent_name and event_id:
            self.transcript_folder = f'organizations/river/agents/{agent_name}/events/{event_id}/transcripts/'
        else:
            self.transcript_folder = '_files/transcripts/'
            
        # Initialize empty transcript keys list - scheduler will populate this
        self.transcript_keys = []
        
    def update_rolling_transcript(self) -> None:
        """Update rolling transcript files for all transcripts in the folder."""
        try:
            # Process each transcript file
            for transcript_key in self.transcript_keys:
                try:
                    # Read the transcript
                    response = self.s3_client.get_object(
                        Bucket=self.s3_bucket,
                        Key=transcript_key
                    )
                    transcript_data = response['Body'].read().decode('utf-8')
                    
                    # Create rolling transcript key
                    base_path = os.path.dirname(transcript_key)
                    filename = os.path.basename(transcript_key)
                    # Prepend rolling- to the filename
                    rolling_key = f"{base_path}/rolling-{filename}"
                    
                    # Split transcript into lines
                    lines = transcript_data.splitlines()
                    now = datetime.now(timezone.utc)
                    
                    # Check for existing session header
                    session_time = None
                    for line in lines:
                        if line.startswith('# Transcript - Session '):
                            session_time = line.split('Session ')[1]
                            break
                    
                    # Create rolling header
                    header = f"# Rolling Transcript - Session {session_time if session_time else datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    rolling_lines = [header]  # Start with header
                    
                    # Process each line
                    for line in lines:
                        # Skip original header
                        if line.startswith('# Transcript - Session '):
                            continue
                            
                        try:
                            # Extract timestamp from line format "[HH:MM:SS - ...]"
                            start_idx = line.index('[') + 1
                            end_idx = line.index(']')
                            timestamp_str = line[start_idx:end_idx].split(' - ')[0]
                            timestamp = parser.parse(timestamp_str).replace(tzinfo=timezone.utc)
                            
                            # Keep lines within live window (not total window)
                            # This ensures we only keep recent content to preserve tokens
                            if now - timestamp <= self.live_window:
                                rolling_lines.append(line)
                        except Exception as e:
                            logger.warning(f"Error parsing line timestamp: {e}")
                            continue
                    
                    # Write filtered content to rolling transcript file
                    rolling_content = "\n".join(rolling_lines)
                    self.s3_client.put_object(
                        Bucket=self.s3_bucket,
                        Key=rolling_key,
                        Body=rolling_content.encode('utf-8')
                    )
                    
                    logger.info(f"Updated rolling transcript {rolling_key} with {len(rolling_lines)} lines")
                    
                except Exception as e:
                    logger.error(f"Error processing transcript {transcript_key}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error updating rolling transcripts: {e}")
    
    def process_embeddings(self) -> None:
        """Process transcript data older than the live window into vector embeddings.
        Processes ALL transcripts in the folder to maintain complete meeting history.
        """
        try:
            # Process each original transcript file
            for transcript_key in self.transcript_keys:
                try:
                    # Read the transcript
                    response = self.s3_client.get_object(
                        Bucket=self.s3_bucket,
                        Key=transcript_key
                    )
                    transcript_data = response['Body'].read().decode('utf-8')
                    
                    # Split into lines and filter for embedding
                    lines = transcript_data.splitlines()
                    now = datetime.now(timezone.utc)
                    embedding_lines = []
                    
                    for line in lines:
                        try:
                            # Extract timestamp from line
                            start_idx = line.index('[') + 1
                            end_idx = line.index(']')
                            timestamp_str = line[start_idx:end_idx].split(' - ')[0]
                            timestamp = parser.parse(timestamp_str).replace(tzinfo=timezone.utc)
                            
                            # Only process lines older than live window (4 minutes)
                            if now - timestamp > self.live_window:
                                embedding_lines.append(line)
                        except Exception as e:
                            logger.warning(f"Error parsing line timestamp for embedding: {e}")
                            continue
                    
                    if not embedding_lines:
                        logger.info(f"No new content to process for embeddings in {transcript_key}")
                        continue
                    
                    # Extract stream/breakout info from filename
                    filename = os.path.basename(transcript_key)
                    # Parse filename to get uID, oID, etc.
                    file_info = {}
                    for part in filename.replace('.txt', '').split('_'):
                        if '-' in part:
                            key, value = part.split('-', 1)
                            file_info[key] = value
                    
                    # Join lines and process into chunks
                    content = "\n".join(embedding_lines)
                    chunks = self.doc_handler.process_document(
                        content,
                        metadata={
                            'session_id': self.session_id,
                            'event_id': self.event_id,
                            'source': 'transcript_vector',
                            'agent_name': self.agent_name,
                            'stream_id': file_info.get('sID', 'unknown'),  # Stream/breakout identifier
                            'user_id': file_info.get('uID', 'unknown'),   # User identifier
                            'original_file': filename                       # Original transcript file
                        }
                    )
                    
                    # Upsert chunks to Pinecone
                    for chunk in chunks:
                        self.embed_handler.embed_and_upsert(
                            chunk['content'],
                            chunk['metadata']
                        )
                    
                    logger.info(f"Processed {len(chunks)} chunks for vector embedding from {transcript_key}")
                    
                except Exception as e:
                    logger.error(f"Error processing embeddings for {transcript_key}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in embedding process: {e}")
    
    def get_live_context(self, read_all: bool = False) -> str:
        """Get the current rolling transcript content.
        
        Args:
            read_all: If True, combine content from all rolling transcripts.
                     If False, use only the latest rolling transcript.
        """
        try:
            if read_all:
                # Get all rolling transcript files
                response = self.s3_client.list_objects_v2(
                    Bucket=self.s3_bucket,
                    Prefix=self.transcript_folder,
                    Delimiter='/'
                )
                
                if 'Contents' not in response:
                    return ""
                
                # Find all rolling transcripts
                rolling_files = [
                    obj['Key'] for obj in response['Contents']
                    if obj['Key'].startswith(self.transcript_folder)
                    and obj['Key'].replace(self.transcript_folder, '').startswith('rolling-')
                    and obj['Key'].endswith('.txt')
                ]
                
                # Combine content from all rolling transcripts
                all_content = []
                for rolling_key in rolling_files:
                    try:
                        response = self.s3_client.get_object(
                            Bucket=self.s3_bucket,
                            Key=rolling_key
                        )
                        content = response['Body'].read().decode('utf-8')
                        if content:
                            all_content.append(content)
                    except Exception as e:
                        logger.warning(f"Error reading rolling transcript {rolling_key}: {e}")
                        continue
                
                return "\n\n".join(all_content)
            else:
                # Get latest rolling transcript
                latest_original = get_latest_transcript_file(
                    self.agent_name,
                    self.event_id,
                    self.s3_client,
                    self.s3_bucket
                )
                if not latest_original:
                    return ""
                
                base_path = os.path.dirname(latest_original)
                filename = os.path.basename(latest_original)
                rolling_key = f"{base_path}/transcript-rolling_{filename}"
                
                try:
                    response = self.s3_client.get_object(
                        Bucket=self.s3_bucket,
                        Key=rolling_key
                    )
                    return response['Body'].read().decode('utf-8')
                except Exception as e:
                    logger.error(f"Error reading rolling transcript: {e}")
                    return ""
                    
        except Exception as e:
            logger.error(f"Error getting live context: {e}")
            return ""
    
    def retrieve_vector_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant vector context for a query.
        
        Args:
            query: The query to find relevant context for
            top_k: Number of top results to return
            
        Returns:
            List of relevant context chunks with metadata
        """
        try:
            # Use embedding handler to search with session/event filters
            results = self.embed_handler.query_index(
                query,
                filter={
                    'session_id': self.session_id,
                    'event_id': self.event_id
                },
                top_k=top_k
            )
            return results
        except Exception as e:
            logger.error(f"Error retrieving vector context: {e}")
            return []
