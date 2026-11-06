# Backend Team Integration Guide — GenAI Platform

## 1. Architectural Boundaries & Division of Responsibility

This microservice provides core GenAI, Retrieval-Augmented Generation (RAG), and knowledge base functionality. 

### Division of Responsibility
* **Main Backend Team Responsibility**:
  - User authentication (JWT, OAuth2, session cookies).
  - Role-based authorization & tenant isolation.
  - Rate limiting, billing, and API gateway routing.
  - Storing user chat logs in SQL/NoSQL databases.
* **GenAI Team Responsibility**:
  - LLM inference and generation quality.
  - Document chunking, vector embeddings, and persistent semantic search.
  - Context window construction and anti-hallucination guardrails.
  - Citation compilation and grounding verification.

---

## 2. API Communication Patterns

All GenAI endpoints are mounted under `/api/v1`. The main backend should reverse-proxy or make service-to-service calls to this service.

### 2.1 Chat & Conversational Query (`POST /api/v1/chat`)

When a user submits a query:
1. The main backend validates the user's JWT token and permission.
2. The main backend forwards the request to `http://genai-service:8000/api/v1/chat`.
3. Pass `conversation_id` so the GenAI platform can maintain multi-turn memory across turns.
4. If your system maintains conversation history in a centralized database, you may also pass the previous turns explicitly in the `history` array.

#### Example Upstream Python Request:
```python
import httpx

async def ask_genai_service(user_query: str, session_id: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://127.0.0.1:8000/api/v1/chat",
            json={
                "query": user_query,
                "conversation_id": session_id,
                "temperature": 0.1,
            },
            timeout=90.0,
        )
        response.raise_for_status()
        return response.json()
```

---

## 3. Handling Fallback & Low Confidence

When the platform cannot find relevant knowledge chunks for a query, it returns:
* `has_sufficient_context: false`
* `is_grounded: false`
* `sources: []`
* A polite, safe fallback message in the `answer` field.

The upstream application can inspect `has_sufficient_context` to:
- Direct the user to customer support.
- Prompt the user to upload domain documents.
- Display an informative disclaimer in the UI.

---

## 4. Ingestion Workflows

When domain documents are uploaded via your administrative UI:
* You can call `POST /api/v1/knowledge/upload` as multipart form data, OR
* Extract text upstream and call `POST /api/v1/knowledge/ingest` with JSON payload `{ "title": "...", "content": "..." }`.

The returned `document_id` should be stored in your main database alongside your document records so you can later call `DELETE /api/v1/knowledge/{document_id}` when a document is deleted.

---

## 5. Error Codes & Retries

| Error Code | HTTP Status | Recommended Backend Action |
| :--- | :--- | :--- |
| `QUERY_VALIDATION_ERROR` | 400 | Return validation error to end user. Do not retry. |
| `DOCUMENT_NOT_FOUND` | 404 | Synchronize document ID registry. |
| `UNSUPPORTED_DOCUMENT_TYPE`| 415 | Advise user to upload PDF, DOCX, TXT, or MD files. |
| `OLLAMA_CONNECTION_FAILED` | 503 | Local LLM daemon down; trigger DevOps alert. |
| `OLLAMA_TIMEOUT` | 504 | Transient load; retry once with exponential backoff. |

