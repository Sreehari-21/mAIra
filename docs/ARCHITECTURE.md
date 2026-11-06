# System Architecture — GenAI Conversational Platform

## 1. Executive Summary & Context

This document outlines the architectural blueprint for the enterprise **Conversational AI Platform**. The GenAI team's scope covers AI behavior, knowledge retrieval (RAG), response quality, local model integration, and evaluation.

### Phase 1 Operational Constraints
* **100% Local Execution**: All inference and vector operations run strictly on the local machine or internal container stack.
* **Ollama Runtime**: Local daemon (`http://localhost:11434`) hosting `qwen2.5:7b` (generation) and `nomic-embed-text` (embeddings).
* **Zero External Cloud AI**: Prohibited from calling Gemini API, OpenAI API, Anthropic API, or paid cloud inference providers.
* **Deterministic RAG & Anti-Hallucination**: The system answers strictly from retrieved context chunks. If context is missing or confidence is low, it invokes deterministic fallback routines.
* **No Agentic/Autonomous Loops**: Phase 1 is a clean, reliable, inspectable RAG pipeline.

---

## 2. High-Level System Architecture

```mermaid
flowchart TD
    subgraph Client / Upstream Backend
        UserApp[Client Web/Mobile App]
        MainBackend[Main Backend Gateway]
    end

    subgraph FastAPI Platform Core
        Router[API Route Controllers]
        Val[Validators & Sanitization]
        RAGServ[RAG Service]
        RetrServ[Retrieval Service]
        ConvServ[Conversation Service]
        IngestPipe[Ingestion Pipeline]
        DocProc[Document Cleaners & Splitters]
    end

    subgraph Local Storage & Model Runtimes
        OllamaGen[Ollama: qwen2.5:7b]
        OllamaEmb[Ollama: nomic-embed-text]
        ChromaStore[(Local Persistent ChromaDB)]
        DataDir[(Local Document Store)]
    end

    UserApp -->|REST API| MainBackend
    MainBackend -->|Authorized Forwarding| Router
    Router -->|/api/v1/chat| RAGServ
    Router -->|/api/v1/knowledge| IngestPipe

    RAGServ --> Val
    RAGServ --> ConvServ
    RAGServ --> RetrServ
    RetrServ --> OllamaEmb
    RetrServ --> ChromaStore
    RAGServ --> OllamaGen

    IngestPipe --> DocProc
    DocProc --> OllamaEmb
    OllamaEmb --> ChromaStore
    DocProc --> DataDir
```

---

## 3. Data Pipelines & Execution Flows

### 3.1 End-to-End Query & RAG Synthesis Flow

```mermaid
sequenceDiagram
    autonumber
    participant Client as Client / Backend
    participant API as FastAPI (/api/v1/chat)
    participant RAG as RAGService
    participant Conv as ConversationService
    participant Retr as RetrievalService
    participant Emb as OllamaEmbedder (nomic)
    participant Chroma as ChromaStore
    participant LLM as OllamaGenerator (qwen2.5)

    Client->>API: POST /api/v1/chat (query, conversation_id)
    API->>RAG: answer_query(request)
    RAG->>RAG: validate_query(query)
    RAG->>Conv: get_or_create_session(conv_id)
    Conv-->>RAG: history

    RAG->>Retr: retrieve(query, top_k)
    Retr->>Emb: embed_text(query)
    Emb-->>Retr: query_vector [768d]
    Retr->>Chroma: similarity_search(query_vector, top_k)
    Chroma-->>Retr: ranked_chunks (with cosine scores)
    Retr-->>RAG: RetrievalResult(chunks, has_sufficient_context)

    alt Context Insufficient or Below Threshold
        RAG->>RAG: apply fallback template
        RAG->>Conv: add_turn(query, fallback)
        RAG-->>API: ChatResponse (is_grounded=False, has_context=False)
        API-->>Client: 200 OK (Clean fallback message)
    else Context Sufficient
        RAG->>RAG: construct prompt (rag.txt + context + history)
        RAG->>LLM: generate(prompt, temp=0.1)
        LLM-->>RAG: generated_text
        RAG->>RAG: verify grounding & compile citations
        RAG->>Conv: add_turn(query, answer)
        RAG-->>API: ChatResponse (answer, sources, is_grounded=True)
        API-->>Client: 200 OK (Grounded Answer with Sources)
    end
```

### 3.2 Document Ingestion Flow

```mermaid
sequenceDiagram
    autonumber
    participant Admin as Ingestion Client / CLI
    participant Pipe as IngestionPipeline
    participant Loader as FileLoader (.pdf, .docx, .txt, .md)
    participant Cleaner as DocumentProcessor & Chunker
    participant Emb as OllamaEmbedder (nomic)
    participant Chroma as ChromaStore

    Admin->>Pipe: ingest_file(file_path)
    Pipe->>Loader: load(file_path)
    Loader-->>Pipe: RawDocument(pages)
    Pipe->>Cleaner: clean_text() & compute SHA-256 hash
    Pipe->>Pipe: check existing content_hash (deduplication)
    Pipe->>Cleaner: chunk_document(chunk_size=500, overlap=50)
    Cleaner-->>Pipe: List[DocumentChunk]
    Pipe->>Emb: embed_batch(chunk_texts)
    Emb-->>Pipe: chunk_embeddings
    Pipe->>Chroma: upsert(chunks, embeddings, metadatas)
    Chroma-->>Pipe: stored_chunk_ids
    Pipe-->>Admin: IngestionResponse(doc_id, chunk_count, status="indexed")
```

---

## 4. Key Architectural Decisions

1. **Local Persistent ChromaDB**:
   - Zero-network overhead, eliminates cloud vector DB egress costs and latency.
   - Persists locally at `./data/chroma`.
   - Uses cosine distance metric (`hnsw:space = "cosine"`).
2. **Provider Isolation Pattern**:
   - Direct Ollama calls are encapsulated in `app/providers/ollama/`.
   - Higher layers (`RAGService`, `IngestionPipeline`) consume abstract services (`LLMService`, `EmbeddingService`), ensuring future swapability without touching business logic.
3. **Traceability & Grounding**:
   - Every chunk receives a deterministic unique ID: `{doc_id}#c{index:04d}`.
   - Metadata stores `filename`, `page_number`, `chunk_index`, and `content_hash`.
   - Prompt engineering explicitly instructs the LLM to refer only to provided context and emit source references.

