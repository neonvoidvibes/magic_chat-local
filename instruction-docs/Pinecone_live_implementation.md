# Pinecone Vector Store Implementation Plan

## Overview
Implement Pinecone vector storage for document embeddings to reduce token usage in the chat system. Documents uploaded to S3 will be automatically embedded and archived, with the chat agent using vector similarity search to retrieve relevant context.

## Directory Structure
```
organizations/river/agents/{agent_name}/
├── docs/                   # Upload directory (source)
│   ├── archived/          # Archived documents after embedding
│   └── *.txt              # New documents to be processed
```

## Implementation Steps

### Day 1 - Morning: Setup & Configuration

1. **Dependencies**
```bash
pip install pinecone-client==3.0.0 langchain==0.1.0
```

2. **Configuration Updates**
- Add to `magic_chat/utils/config.py`:
```python
class AppConfig:
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_batch_size: int = 100
    embedding_model: str = "text-embedding-3-small"
```

3. **Create Vector Store Manager**
- Create `magic_chat/utils/vector_store.py`:
```python
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import boto3
from datetime import datetime
import logging

class VectorStoreManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            retry_on_error=True,
            max_retries=3
        )
        pinecone.init(
            api_key=config.pinecone_api_key,
            environment=config.pinecone_environment
        )
        self.index_name = f"river-{config.agent_name}"
        self.logger = logging.getLogger('magic_chat.vector_store')
```

### Day 1 - Afternoon: Core Implementation

1. **Document Processing**
- Add to `VectorStoreManager`:
```python
def process_s3_docs(self):
    """Process new documents from S3"""
    try:
        # Get docs from S3
        docs = read_agent_docs(self.config.agent_name)
        if not docs:
            return

        # Initialize/get index
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=1536,
                metric='cosine'
            )

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_text(docs)

        # Process in batches
        for i in range(0, len(texts), self.config.pinecone_batch_size):
            batch = texts[i:i + self.config.pinecone_batch_size]
            vectorstore = Pinecone.from_texts(
                texts=batch,
                embedding=self.embeddings,
                index_name=self.index_name,
                metadata={
                    "agent": self.config.agent_name,
                    "event_id": self.config.event_id,
                    "processed_at": datetime.now().isoformat()
                }
            )
            self.logger.info(f"Processed batch {i//self.config.pinecone_batch_size + 1}")

        # Archive processed documents
        self._archive_processed_docs()
        
    except Exception as e:
        self.logger.error(f"Error processing documents: {e}")
        raise

def _archive_processed_docs(self):
    """Move processed docs to archive"""
    s3 = boto3.client('s3')
    prefix = f'organizations/river/agents/{self.config.agent_name}/docs/'
    
    # List files in docs folder
    response = s3.list_objects_v2(
        Bucket=AWS_S3_BUCKET,
        Prefix=prefix
    )
    
    if 'Contents' not in response:
        return
        
    for obj in response['Contents']:
        if obj['Key'] == prefix or 'archived/' in obj['Key']:
            continue
            
        # New path in archived subfolder
        new_key = obj['Key'].replace(
            '/docs/', 
            '/docs/archived/'
        )
        
        try:
            # Copy to archive
            s3.copy_object(
                Bucket=AWS_S3_BUCKET,
                CopySource={'Bucket': AWS_S3_BUCKET, 'Key': obj['Key']},
                Key=new_key
            )
            
            # Delete original
            s3.delete_object(
                Bucket=AWS_S3_BUCKET,
                Key=obj['Key']
            )
            
            self.logger.info(f"Archived {obj['Key']} to {new_key}")
            
        except Exception as e:
            self.logger.error(f"Error archiving {obj['Key']}: {e}")
```

2. **Integration with WebChat**
- Update `WebChat.__init__`:
```python
def __init__(self, config: AppConfig):
    # Existing init code...
    self.vector_store = VectorStoreManager(config)
```

- Update `load_resources`:
```python
def load_resources(self):
    """Load resources and process docs"""
    # Load basic system prompt
    self.system_prompt = get_latest_system_prompt(self.config.agent_name)
    
    # Process any new docs
    try:
        self.vector_store.process_s3_docs()
    except Exception as e:
        self.logger.error(f"Failed to process documents: {e}")
    
    # Rest of resource loading...
```

### Day 2 - Morning: Chat Integration

1. **Query Implementation**
- Add to `VectorStoreManager`:
```python
def search_similar(self, query: str, k: int = 3):
    """Search for similar documents"""
    try:
        vectorstore = Pinecone.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        return vectorstore.similarity_search(
            query,
            k=k,
            namespace=f"docs-{self.config.event_id}"
        )
    except Exception as e:
        self.logger.error(f"Error searching documents: {e}")
        return []
```

- Update `WebChat.generate`:
```python
def generate(self):
    """Handle chat generation"""
    user_message = request.json.get('message')
    
    # Get relevant docs
    similar_docs = self.vector_store.search_similar(user_message)
    
    if similar_docs:
        # Add as temporary system message
        self.chat_history.append({
            "role": "system",
            "content": "Relevant context:\n" + 
                      "\n---\n".join([doc.page_content for doc in similar_docs])
        })
    
    # Generate response
    response = self.client.messages.create(
        messages=self.chat_history,
        model="claude-2.1"
    )
    
    # Remove temporary context
    if similar_docs:
        self.chat_history.pop()
    
    return response
```

### Day 2 - Afternoon: Testing & Monitoring

1. **Error Handling & Retries**
- Add retry logic for embeddings
- Add document processing status tracking
- Implement fallback mechanisms

2. **Monitoring**
- Add metrics for:
  - Document processing time
  - Embedding costs
  - Query latency
  - Storage usage

3. **Testing**
- Test document upload flow
- Verify archival process
- Test vector search accuracy
- Load testing for concurrent users

## Cost Considerations
1. Pinecone Storage: ~$0.02 per 1000 vectors per month
2. OpenAI Embeddings: $0.00002 per 1K tokens
3. Batch processing to optimize costs

## Maintenance Tasks
1. Regular index optimization
2. Monitoring embedding costs
3. Cleanup of old/unused vectors
4. Backup of vector indexes

## Future Enhancements
1. Document versioning
2. Metadata filtering
3. Multi-modal embeddings
4. Hybrid search (keyword + semantic)
