**Task 3: Step-by-Step Guide for a Fresh Pinecone Account (.md)**
Date: 2025.02.13

# Step-by-Step Setup Guide for Your New Pinecone Account

Use this guide to initialize your Pinecone account, create an index, and integrate it into your existing code.

---

## 1. Account Preparation

1. **Sign Up / Log In**  
   - Visit [Pinecone.io](https://www.pinecone.io) and log in (or create an account).
2. **Create an API Key**  
   - From your Pinecone console, go to **API Keys** and create a new key.  
   - Copy the key for use in `.env` (or wherever you store secrets).

3. **Set Environment Variables**  
   In your `.env` file or environment config:
   ```bash
   PINECONE_API_KEY="YOUR_NEW_PINECONE_API_KEY"
   PINECONE_ENVIRONMENT="us-east1-gcp"  # Or the environment shown in your console
   ```

---

## 2. Create an Index in the Console (Serverless Recommended)

1. **Navigate to Indexes**  
   - In the Pinecone console, go to **Indexes** → **Create Index**.
2. **Choose “Serverless”**  
   - Typically recommended for small/medium workloads and pay-as-you-go usage.
3. **Set Dimensions**  
   - If using OpenAI’s `text-embedding-ada-002`, set **dimension=1536**.
4. **Select Metric**  
   - Usually **cosine** for textual embeddings.
5. **Finish**  
   - Wait a few seconds while the index is created.

*(Alternatively, you can call `pinecone.create_index(...)` from Python, but for a new account, the console-based approach is simpler.)*

---

## 3. Connect Your Code to This New Index

1. **Update `.env`**  
   - Add `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT`.
   - Optionally, specify a default `PINECONE_INDEX_NAME="chat-docs-index"`.
2. **Initialize Pinecone**  
   In your `pinecone_utils.py` (or similar):
   ```python
   import pinecone
   import os

   def init_pinecone():
       api_key = os.getenv("PINECONE_API_KEY")
       env = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
       pinecone.init(api_key=api_key, environment=env)
   ```
3. **Validate**  
   - Run a small script (or the existing `create_or_verify_index` logic) to confirm you can list indexes:
     ```python
     import pinecone
     init_pinecone()
     print(pinecone.list_indexes())
     ```
   - You should see `[ "chat-docs-index" ]` (or whatever you named your index).

---

## 4. Test Document Embedding

1. **Pick a Sample File**  
   - e.g., `docs/test_document.md`.
2. **Call CLI** (if you have `cli_embed.py`):
   ```bash
   python utils/cli_embed.py docs/test_document.md --agent river --index chat-docs-index
   ```
   - This should split and embed `test_document.md` into the namespace `river` in Pinecone.
3. **Check Pinecone Console**  
   - Look at your index → “Explore” tab to see if new vectors appear.

---

## 5. Basic Retrieval Test

1. **Run a Query**  
   - Use your code’s retrieval method (`RetrievalHandler.get_relevant_context(...)`) or a quick snippet:
     ```python
     from langchain_openai import OpenAIEmbeddings
     import pinecone

     init_pinecone()
     index = pinecone.Index("chat-docs-index")
     query_vec = OpenAIEmbeddings().embed_query("What is this document about?")
     results = index.query(vector=query_vec, top_k=3, namespace="river", include_metadata=True)
     print(results)
     ```
   - Validate that you get relevant metadata, chunk texts, etc.

---

## 6. Integration into Your Chat Flows

1. **Add to `web_chat.py` or `magic_chat.py`**  
   - Where you see a user message, embed it, call the retrieval, then form your final LLM prompt with context.  
2. **Finalize**  
   - Test by sending a question that should be answerable from your embedded doc.  
   - Confirm that the LLM references the correct chunk text from Pinecone.

---

## 7. Maintenance and Next Steps

- **Scaling**: Pinecone serverless will auto-scale up to large datasets.  
- **Namespace Management**: You can maintain separate namespaces for each agent, event, or data category.  
- **Metadata Filters**: Attach metadata like `event_id`, `doc_type`, or timestamps so you can refine your searches.  
- **Periodic Cleanup**: If documents are updated or removed in S3, reflect that in Pinecone by deleting vectors or re-embedding as needed.

---

## Summary

By following these steps—setting your environment variables, creating a Pinecone index in serverless mode, embedding your documents, and querying them—you’ll have a fully operational Pinecone vector store that pairs with your existing S3 bucket and chat application. This ensures you can store, retrieve, and generate answers based on all your documents efficiently.