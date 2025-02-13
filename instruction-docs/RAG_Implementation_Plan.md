**Task 2: RAG Implementation Plan (.md)**
Date: 2025.02.13

# Implementation Plan for RAG Integration (Pinecone & S3 as Parallel Systems)

This document outlines how to integrate your Pinecone-based document store into your existing Retrieval-Augmented Generation (RAG) workflow while keeping S3 and Pinecone as parallel systems. In this setup, the admin uploads documents directly to Pinecone for embedding and retrieval, and S3 continues to serve as a separate file storage system if needed.

---

## 1. RAG Overview

RAG involves:
1. **Ingestion/Embedding**: Documents are uploaded by an admin directly into Pinecone. The process involves:
   - Chunking the text
   - Embedding the chunks
   - Upserting them into the Pinecone index
2. **Retrieval**: For each user query, the system retrieves the top-K semantically similar chunks from Pinecone.
3. **Augmented Prompt**: Retrieved chunks are appended to the system or user prompt.
4. **Generation**: The LLM (Anthropic, OpenAI, etc.) uses the augmented prompt to produce context-aware answers.

> **Note:** S3 remains available as a file storage solution but is not used for document retrieval in this workflow.

---

## 2. Proposed Workflow

Below is the recommended plan to integrate Pinecone as the primary vector store for your RAG pipeline:

### A. **Document Ingestion & Embedding**
1. **Ingestion**  
   - Admins upload documents directly for processing.
   - These documents do not need to be synchronized between S3 and Pinecone.
2. **Chunking**  
   - Use `DocumentHandler.process_document(...)` to split the text into ~1000-character chunks with ~200 characters of overlap.
3. **Embedding**  
   - Use `EmbeddingHandler.embed_and_upsert(...)` to generate embeddings and upsert them into the Pinecone index under the appropriate namespace (e.g., agent name).

### B. **Context Retrieval**
1. **Query Embedding**  
   - When the user asks a question, embed the query using the same embedding method (e.g., OpenAIEmbeddings).
2. **Similarity Search**  
   - Use `RetrievalHandler.get_relevant_context(...)` to query Pinecone and fetch the top-K relevant chunks from the specified namespace.
3. **No Fallback Mechanism**  
   - Since documents are maintained independently in Pinecone, there is no fallback to S3 for context retrieval.

### C. **Prompt Assembly (Augmented Prompt)**
1. **System Prompt + Context**  
   - Combine your system prompt (loaded via `get_latest_system_prompt(...)`) with the retrieved context:
     ```python
     relevant_context = "\n\n".join(c['content'] for c in contexts)
     final_prompt = f"{system_prompt}\n\nRelevant context:\n{relevant_context}"
     ```
2. **LLM Generation**  
   - Send the final prompt to your LLM (Anthropic/OpenAI) and generate an answer.

### D. **Chat History & Storage**
1. **Conversation History**  
   - Maintain a local list of conversation turns (user and assistant messages).
2. **Archival**  
   - Optionally, save the conversation history to S3 for record keeping or memory reloading, but this is independent of the Pinecone retrieval process.

---

## 3. Technical Integration Steps

1. **Fix Namespace Issues**  
   - Ensure that in `RetrievalHandler`, the Pinecone queries use the correct namespace (e.g., `namespace=self.namespace`).
2. **Index Creation**  
   - Use `create_or_verify_index` to create a new Pinecone index (preferably in serverless mode) with the appropriate dimension (1536 for ada-002) and metric (cosine).
3. **Embed Documents**  
   - Use the CLI tool (`cli_embed.py`) or an automated process to embed and upsert documents into Pinecone.
4. **Test Retrieval**  
   - Verify that querying Pinecone using `RetrievalHandler.get_relevant_context()` returns the expected document chunks.
5. **Integrate into Chat**  
   - In your chat flow (e.g., in `web_chat.py` or `magic_chat.py`), update the logic to fetch context solely from Pinecone, assemble the final prompt, and pass it to the LLM.
6. **Review and Refine**  
   - Adjust chunk sizes, top-K values, and prompt formatting based on testing and performance feedback.

---

## 4. Conclusion

By following these steps, your system will use Pinecone as the primary vector store for retrieving context relevant to user queries, while S3 continues to operate independently as a storage solution. This parallel system design simplifies the document retrieval process, with admin uploads directly to Pinecone ensuring fast and reliable access to embedded document content during chat sessions.