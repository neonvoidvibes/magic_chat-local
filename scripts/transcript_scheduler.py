"""Scheduler for rolling transcript management."""
import os
import time
import logging
import schedule
import threading
from datetime import datetime
from utils.rolling_transcript import RollingTranscriptManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_session_id() -> str:
    """Create a session ID using current timestamp."""
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def create_event_id() -> str:
    """Create an event ID using current date."""
    return datetime.now().strftime('%Y%m%d')

def start_scheduler(agent_name: str, session_id: str, event_id: str) -> threading.Thread:
    """Start the rolling transcript scheduler in a background thread.
    
    Args:
        agent_name: Name of the agent
        session_id: Current session ID
        event_id: Current event ID
        
    Returns:
        The scheduler thread
    """
    manager = RollingTranscriptManager(
        agent_name=agent_name,
        session_id=session_id,
        event_id=event_id
    )
    
    def update_all():
        """Update both rolling transcript and process embeddings."""
        try:
            logger.info("Starting scheduled update")
            # Get current transcript files
            response = manager.s3_client.list_objects_v2(
                Bucket=manager.s3_bucket,
                Prefix=manager.transcript_folder,
                Delimiter='/'
            )
            
            # Find original transcripts (excluding rolling ones)
            if 'Contents' in response:
                transcript_files = [
                    obj['Key'] for obj in response['Contents']
                    if obj['Key'].startswith(manager.transcript_folder)
                    and obj['Key'] != manager.transcript_folder
                    and not obj['Key'].replace(manager.transcript_folder, '').strip('/').count('/')
                    and obj['Key'].endswith('.txt')
                    and not obj['Key'].replace(manager.transcript_folder, '').startswith('rolling-')
                ]
                
                if transcript_files:
                    # Update transcript keys and process
                    manager.transcript_keys = transcript_files
                    manager.update_rolling_transcript()
                    manager.process_embeddings()
                    logger.info("Completed scheduled update")
                else:
                    logger.info("No original transcripts found yet, waiting...")
            else:
                logger.info("No files in transcript folder yet, waiting...")
                
        except Exception as e:
            logger.error(f"Error in scheduled update: {e}")
    
    def run_scheduler():
        """Run the scheduler in a loop."""
        # Schedule updates every minute
        schedule.every(1).minutes.do(update_all)
        
        # Run initial update
        update_all()
        
        logger.info(f"Started transcript scheduler for agent {agent_name}")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Event ID: {event_id}")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
    
    # Start scheduler in background thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    return scheduler_thread

def main():
    """Main function when running scheduler standalone."""
    session_id = create_session_id()
    event_id = create_event_id()
    agent_name = os.getenv('AGENT_NAME', 'river')
    
    # Start scheduler and wait for interrupt
    thread = start_scheduler(agent_name, session_id, event_id)
    try:
        thread.join()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")

if __name__ == "__main__":
    main()
