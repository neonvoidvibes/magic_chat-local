**Task 2: RAG Implementation Plan (.md)**

# Implementation Plan for RAG Integration

This document outlines how to integrate your Pinecone-based document store into your existing Retrieval-Augmented Generation (RAG) workflow.

---

## 1. RAG Overview

RAG involves:
1. **Ingestion/Embedding**: Reading documents (S3 or local) → chunking → embedding → upserting into Pinecone.
2. **Retrieval**: For each user query, we retrieve top-K semantically similar chunks from Pinecone (and optionally fallback to S3).
3. **Augmented Prompt**: We append retrieved chunks to the system or user prompt.
4. **Generation**: The LLM (Anthropic, OpenAI, etc.) uses the augmented prompt to produce context-aware answers.

---

## 2. Proposed Workflow

Below is the recommended plan to unify your code with the RAG pipeline:

### A. **Document Ingestion & Embedding**
1. **Ingestion**  
   - Whenever new documents are dropped into S3 or manually uploaded, your ingestion flow will:
     - Read file contents from S3.
     - Possibly parse or chunk the file.  
2. **Chunking**  
   - Use `DocumentHandler.process_document(...)` to split the text into ~1000-character chunks with ~200 overlap.  
3. **Embedding**  
   - Use `EmbeddingHandler.embed_and_upsert(...)` to generate embeddings and upsert into the Pinecone index, with `namespace=agent_name`.

### B. **Context Retrieval**
1. **Query Embedding**  
   - When the user asks a question, embed the query using the same `OpenAIEmbeddings`.
2. **Similarity Search**  
   - `RetrievalHandler` calls `self.index.query(...)` or uses a `PineconeRetriever` to fetch top-K relevant chunks from the same namespace.  
3. **Optional Fallback**  
   - If top-K is empty, optionally fetch the raw doc from S3 or consider a keyword-based retrieval.  

### C. **Prompt Assembly (Augmented Prompt)**
1. **System Prompt + Context**  
   - Combine your existing system prompt (`get_latest_system_prompt(...)`) with the chunked text from Pinecone:  
     ```python
     relevant_context = "\n\n".join(c['content'] for c in contexts)
     final_prompt = f"{system_prompt}\n\nRelevant context:\n{relevant_context}"
     ```
2. **LLM Generation**  
   - Send this final prompt to your LLM (Anthropic/OpenAI), capturing the response.  

### D. **Chat History & Storage**
1. **Conversation History**  
   - Maintain a local list of turns in memory or in S3.  
2. **Archival**  
   - Save the conversation to S3 at intervals so you can reload memory or use prior transcripts.

---

## 3. Technical Integration Steps

1. **Fix the `namespace=namespace` Bug**  
   Update `namespace=self.namespace` so Pinecone retrieval queries the correct namespace.

2. **Test Embedding**  
   - Use `cli_embed.py` or a minimal script to embed a sample file into Pinecone.  
   - Check Pinecone logs and confirm your namespace is populated.

3. **Validate Retrieval**  
   - Use `RetrievalHandler.get_relevant_context()` to retrieve chunks for a known query.  
   - Confirm the chunk text and metadata are correct.

4. **Integrate with the Chat Agent**  
   - In `web_chat.py` or `magic_chat.py`, after the user message arrives, embed it, retrieve context, augment the system prompt, and call your LLM.

5. **Optional Re-rank**  
   - If you want better accuracy, you can fetch a slightly larger top-K from Pinecone, then re-rank with a separate LLM or “hybrid search” approach.

---

## 4. Implementation Sequence

1. **Deploy Fix** – Update `retrieval_handler.py` so it references the correct namespace.
2. **Create Index** (if needed) – Make sure `create_or_verify_index` is set to serverless or your desired config.
3. **Embed** – Use your `cli_embed.py` or a script to embed test documents into Pinecone.
4. **Test Query** – Verify you can retrieve those documents by a relevant question.
5. **Plug into Chat** – Modify the chat flow to fetch `top_k` results from Pinecone, attach them to the prompt, and finalize LLM generation.
6. **Refine** – Adjust chunk sizes, top_k, or fallback logic as usage evolves.

---

## 5. Conclusion

By following these steps, your codebase will support RAG workflows using Pinecone as the primary vector store, with S3 continuing as the source of truth for raw documents. This approach ensures quick retrieval, scalable embedding, and the ability to handle large volumes of text with minimal overhead.