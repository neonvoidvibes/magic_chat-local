"""Scheduler for rolling transcript management."""
import os
import time
import logging
import schedule
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

def main():
    """Main scheduler function."""
    # Initialize manager with session/event IDs
    session_id = create_session_id()
    event_id = create_event_id()
    agent_name = os.getenv('AGENT_NAME', 'river')
    
    manager = RollingTranscriptManager(
        agent_name=agent_name,
        session_id=session_id,
        event_id=event_id
    )
    
    def update_all():
        """Update both rolling transcript and process embeddings."""
        try:
            logger.info("Starting scheduled update")
            manager.update_rolling_transcript()
            manager.process_embeddings()
            logger.info("Completed scheduled update")
        except Exception as e:
            logger.error(f"Error in scheduled update: {e}")
    
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
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")

if __name__ == "__main__":
    main()
