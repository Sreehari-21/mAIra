# API Contracts Specification (v1)

> **Status**: Proposed GenAI Service Interface Contracts  
> **Target Audience**: GenAI Team & Main Backend Engineering Team  
> **Security Note**: This service runs behind the primary backend gateway. Authentication/Authorization (OAuth2/JWT/RBAC) is delegated to the main backend and must not be implemented inside this GenAI microservice.

---

## 1. Standard Error Envelope

All error responses across all endpoints follow a uniform JSON structure:

```json
{
  "success": false,
  "error": {
    "code": "QUERY_VALIDATION_ERROR",
    "message": "Query text cannot be empty or whitespace only.",
    "details": {
      "field": "query"
    }
  }
}
```

Standard Status Codes:
* `400 Bad Request`: Input validation failed.
* `404 Not Found`: Document ID not found in knowledge base.
* `415 Unsupported Media Type`: Document format not supported.
* `503 Service Unavailable`: Local Ollama daemon unreachable.
* `504 Gateway Timeout`: Ollama inference exceeded timeout threshold.
* `500 Internal Server Error`: Unhandled server exception.

---

## 2. Health & Diagnostic Endpoints

### 2.1 `GET /api/v1/health`
Checks whether the FastAPI application process is healthy.

* **Response Headers**: `Content-Type: application/json`
* **Status**: `200 OK`
* **Response Body**:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "environment": "development",
  "timestamp": "2026-11-06T00:00:00.000000+00:00"
}
```

---

### 2.2 `GET /api/v1/health/ollama`
Verifies connection to the local Ollama runtime and verifies whether `qwen2.5:7b` and `nomic-embed-text` are pulled.

* **Response Headers**: `Content-Type: application/json`
* **Status**: `200 OK`
* **Response Body (Healthy)**:
```json
{
  "status": "connected",
  "base_url": "http://localhost:11434",
  "generation_model": "qwen2.5:7b",
  "embedding_model": "nomic-embed-text",
  "is_available": true,
  "models_available": ["qwen2.5:7b", "nomic-embed-text:latest"],
  "message": "Ollama is online and all required models are present."
}
```

* **Response Body (Offline / Degraded)**:
```json
{
  "status": "offline",
  "base_url": "http://localhost:11434",
  "generation_model": "qwen2.5:7b",
  "embedding_model": "nomic-embed-text",
  "is_available": false,
  "models_available": [],
  "message": "Local Ollama daemon unreachable: Cannot connect to local Ollama runtime. Ensure 'ollama serve' is running."
}
```

---

## 3. Conversational / Chat Endpoint

### 3.1 `POST /api/v1/chat`
Processes user question through the local RAG pipeline and returns grounded synthesis with source references.

* **Request Headers**: `Content-Type: application/json`
* **Request Body**:
```json
{
  "query": "What is the primary LLM generation runtime used in Phase 1?",
  "conversation_id": "conv-a1b2c3d4e5f6",
  "history": [
    {
      "role": "user",
      "content": "Hello, can you help me with platform architecture?"
    },
    {
      "role": "assistant",
      "content": "Of course! What would you like to know?"
    }
  ],
  "top_k": 4,
  "temperature": 0.1
}
```

* **Parameters**:
  - `query` (*string*, required): 1 to 4000 characters.
  - `conversation_id` (*string*, optional): Session ID to maintain multi-turn memory. Generated if omitted.
  - `history` (*array of ChatMessage*, optional): Client-provided prior turns if client manages state.
  - `top_k` (*integer*, optional, 1-20): Chunks to retrieve. Defaults to server setting (4).
  - `temperature` (*float*, optional, 0.0-1.0): Generation sampling temperature. Defaults to 0.1 for high factual consistency.

* **Response Body (200 OK)**:
```json
{
  "answer": "According to the architecture specification, Phase 1 uses Ollama as the local LLM runtime with qwen2.5:7b as the default generation model [Document: architecture.md | Chunk ID: doc-architecture-a1b2#c0001].",
  "conversation_id": "conv-a1b2c3d4e5f6",
  "sources": [
    {
      "document_id": "doc-architecture-a1b2",
      "filename": "architecture.md",
      "chunk_id": "doc-architecture-a1b2#c0001",
      "page_number": 1,
      "similarity_score": 0.884,
      "snippet": "Phase 1 constraint: Everything must run locally. Use Ollama as the LLM runtime with qwen2.5:7b..."
    }
  ],
  "is_grounded": true,
  "has_sufficient_context": true,
  "latency_seconds": 0.742,
  "model_used": "qwen2.5:7b"
}
```

---

## 4. Knowledge Management Endpoints

### 4.1 `POST /api/v1/knowledge/ingest`
Ingests raw text directly into the ChromaDB vector store.

* **Request Body**:
```json
{
  "title": "Platform SLA Policies",
  "content": "The Conversational AI platform provides 99.9% local uptime for internal inference...",
  "metadata": {
    "author": "Architecture Team",
    "department": "Engineering"
  }
}
```

* **Response Body (201 Created)**:
```json
{
  "document_id": "doc-platform-sla-polic-3f8c7e1a",
  "filename": "Platform SLA Policies",
  "chunk_count": 3,
  "character_count": 1420,
  "status": "indexed",
  "message": "Text successfully processed, chunked, and indexed into local vector database."
}
```

---

### 4.2 `POST /api/v1/knowledge/upload`
Uploads and indexes a document file (`.txt`, `.md`, `.json`, `.csv`, `.pdf`, `.docx`).

* **Request Content-Type**: `multipart/form-data`
* **Form Field**: `file` (binary document)
* **Response Body (201 Created)**:
```json
{
  "document_id": "doc-platform-spec-4b2a19cd",
  "filename": "platform_spec.pdf",
  "chunk_count": 12,
  "character_count": 5830,
  "status": "indexed",
  "message": "File 'platform_spec.pdf' successfully ingested into knowledge base."
}
```

---

### 4.3 `GET /api/v1/knowledge`
Returns an inventory of all indexed documents in the local knowledge base.

* **Response Body (200 OK)**:
```json
{
  "documents": [
    {
      "document_id": "doc-platform-spec-4b2a19cd",
      "filename": "platform_spec.pdf",
      "chunk_count": 12,
      "source_type": ".pdf",
      "ingested_at": "2026-11-06T00:15:00.000000+00:00",
      "metadata": {
        "total_pages": 4,
        "content_hash": "a8f5f167f44f4964e6c998dee827110c..."
      }
    }
  ],
  "total_count": 1
}
```

---

### 4.4 `DELETE /api/v1/knowledge/{document_id}`
Deletes a document and all of its vectorized chunks from the vector database.

* **Path Parameter**: `document_id` (string, required)
* **Response Body (200 OK)**:
```json
{
  "document_id": "doc-platform-spec-4b2a19cd",
  "deleted": true,
  "chunks_removed": 12,
  "message": "Document 'doc-platform-spec-4b2a19cd' successfully removed from knowledge base."
}
```

* **Error (404 Not Found)**:
```json
{
  "detail": "Document 'doc-nonexistent' does not exist in knowledge base."
}
```

