# Updated Implementation Plan in Markdown

Below is the complete plan in Markdown with detailed implementation steps and code examples.

---

## Overview

This plan describes how to implement a live transcript processing system that:
- Maintains a **6-minute rolling transcript file** in S3.
- Keeps the **latest 4 minutes exclusively** in the live text file.
- Processes all transcript data older than 4 minutes (plus a 2-minute overlap, roughly one 500-token chunk) into vector embeddings in Pinecone.
- Ensures all transcript content (both in text and embeddings) is tagged with a **session ID, event ID, and timestamps** to maintain session integrity.

---

## Components

1. **Rolling Transcript File Manager**
   - Continuously updates a rolling transcript file with the latest 6 minutes of transcript data.
   - Ensures that only transcript older than 4 minutes (i.e. the first 2 minutes of the 6-minute window) is available for vector embedding.

2. **Vector Embedding Processor**
   - Reads transcript data older than 4 minutes from the rolling file.
   - Splits this data into 500-token chunks with ~100-token overlap.
   - Enriches each chunk with metadata (session ID, event ID, timestamps) and upserts it into the Pinecone vector database.

3. **Agent Retrieval Logic**
   - Combines live text from the rolling transcript (covering the entire 6-minute window, with the latest 4 minutes being live-only) with vector embeddings (filtered by session metadata) for query processing.

4. **Scheduler and State Management**
   - Periodically updates the rolling transcript file (e.g., every 1 minute) and processes new transcript data for vector embedding (every 1–2 minutes).

---

## Detailed Steps & Code Examples

### 1. Rolling Transcript File Manager

This component reads the live transcript file from S3, filters out lines older than 6 minutes, and writes them to a rolling transcript file.

```python
import boto3
import datetime
from dateutil import parser

S3_BUCKET = "your-s3-bucket-name"
TRANSCRIPT_KEY = "organizations/river/agents/{agent_name}/transcripts/live.txt"
ROLLING_TRANSCRIPT_KEY = "organizations/river/agents/{agent_name}/transcripts/rolling.txt"
TIME_WINDOW = datetime.timedelta(minutes=6)

def update_rolling_transcript(agent_name):
    s3 = boto3.client('s3')
    # Read the full live transcript file from S3
    transcript_obj = s3.get_object(Bucket=S3_BUCKET, Key=TRANSCRIPT_KEY.format(agent_name=agent_name))
    transcript_data = transcript_obj['Body'].read().decode('utf-8')
    
    # Split transcript into lines (each with a timestamp like: "[13:09:04 - ...]")
    lines = transcript_data.splitlines()
    now = datetime.datetime.now(datetime.timezone.utc)
    rolling_lines = []
    
    for line in lines:
        try:
            start_idx = line.index('[') + 1
            end_idx = line.index(']')
            # Assuming the timestamp format is "HH:MM:SS"
            timestamp_str = line[start_idx:end_idx].split(' - ')[0]
            timestamp = parser.parse(timestamp_str)
        except Exception:
            continue
        
        # Retain lines within the last 6 minutes
        if now - timestamp <= TIME_WINDOW:
            rolling_lines.append(line)
    
    rolling_content = "\n".join(rolling_lines)
    s3.put_object(Bucket=S3_BUCKET, Key=ROLLING_TRANSCRIPT_KEY.format(agent_name=agent_name), Body=rolling_content.encode('utf-8'))
```

### 2. Vector Embedding Processor

This module processes transcript data older than 4 minutes (from the rolling file) into vector embeddings.

```python
from utils.document_handler import DocumentHandler
from utils.embedding_handler import EmbeddingHandler
import boto3
import datetime
from dateutil import parser

S3_BUCKET = "your-s3-bucket-name"
ROLLING_TRANSCRIPT_KEY = "organizations/river/agents/{agent_name}/transcripts/rolling.txt"

def process_transcript_for_embedding(agent_name, session_id, event_id):
    s3 = boto3.client('s3')
    # Read the rolling transcript file
    rolling_obj = s3.get_object(Bucket=S3_BUCKET, Key=ROLLING_TRANSCRIPT_KEY.format(agent_name=agent_name))
    rolling_data = rolling_obj['Body'].read().decode('utf-8')
    lines = rolling_data.splitlines()
    now = datetime.datetime.now(datetime.timezone.utc)
    embedding_lines = []
    
    # Process lines older than 4 minutes (i.e., not in the latest 4 minutes)
    for line in lines:
        try:
            start_idx = line.index('[') + 1
            end_idx = line.index(']')
            timestamp_str = line[start_idx:end_idx].split(' - ')[0]
            timestamp = parser.parse(timestamp_str)
        except Exception:
            continue
        
        if now - timestamp > datetime.timedelta(minutes=4):
            embedding_lines.append(line)
    
    content = "\n".join(embedding_lines)
    
    # Use DocumentHandler to split the content into ~500-token chunks with ~100-token overlap
    doc_handler = DocumentHandler(chunk_size=500, chunk_overlap=100)
    chunks = doc_handler.process_document(content, metadata={
        'session_id': session_id,
        'event_id': event_id,
        'source': 'transcript_vector'
    })
    
    # Initialize the EmbeddingHandler for upserting to Pinecone
    embed_handler = EmbeddingHandler(index_name="magicchat", namespace=agent_name)
    for chunk in chunks:
        embed_handler.embed_and_upsert(chunk['content'], chunk['metadata'])
```

### 3. Agent Retrieval Logic

When a query is made, the agent will retrieve live text from the rolling file and vector embeddings from Pinecone.

```python
def get_live_context(agent_name):
    import boto3
    s3 = boto3.client('s3')
    ROLLING_TRANSCRIPT_KEY = "organizations/river/agents/{agent_name}/transcripts/rolling.txt"
    transcript_obj = s3.get_object(Bucket="your-s3-bucket-name", Key=ROLLING_TRANSCRIPT_KEY.format(agent_name=agent_name))
    live_text = transcript_obj['Body'].read().decode('utf-8')
    return live_text

def retrieve_vector_context(query, agent_name, session_id, event_id):
    from utils.retrieval_handler import RetrievalHandler
    retriever = RetrievalHandler(index_name="magicchat", agent_name=agent_name, top_k=3)
    # Filter by session_id and event_id to ensure only current session data is retrieved
    contexts = retriever.get_relevant_context(query, filter_metadata={'session_id': session_id, 'event_id': event_id})
    return contexts
```

### 4. Scheduler and State Management

Use a scheduler to update the rolling transcript file and process embeddings periodically.

```python
import schedule
import time

AGENT_NAME = "river"
SESSION_ID = "session_20250214_130903"
EVENT_ID = "20250214"

def update_all():
    update_rolling_transcript(AGENT_NAME)
    process_transcript_for_embedding(AGENT_NAME, SESSION_ID, EVENT_ID)

# Schedule updates every 1 minute
schedule.every(1).minutes.do(update_all)

while True:
    schedule.run_pending()
    time.sleep(1)
```

---

## Summary

- **Rolling Transcript File (6 Minutes):**  
  The system maintains a 6-minute rolling transcript file in S3. The latest 4 minutes exist only in the live text file; all transcript data older than 4 minutes (including an overlapping 2-minute segment) is used for vector embedding.

- **Vector Embedding Pipeline:**  
  Transcript data older than 4 minutes is split into 500-token chunks (with ~100-token overlap), enriched with session metadata (session ID, event ID, timestamps), and upserted into Pinecone.

- **Agent Retrieval:**  
  When a user query arrives, the agent retrieves live context from the rolling transcript file and fetches vector embeddings (filtered by session ID and event ID) from Pinecone. The two sources are combined to provide contextually rich, token-efficient responses.

This implementation ensures that live processing, token efficiency, and rigorous session integrity are maintained throughout the system.
</Plan>
```