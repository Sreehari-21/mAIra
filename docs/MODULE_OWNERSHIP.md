# Module Ownership & Team Collaboration Architecture

To enable autonomous parallel development without merge conflicts, the codebase is partitioned into **6 independently developable modules**. Each module has explicit file ownership boundaries, clear public interfaces, input/output contracts, and dedicated test suites.

---

## Module Matrix Overview

| Module # | Module Domain | Primary Owner Role | Core Directory / Files |
| :--- | :--- | :--- | :--- |
| **Module 1** | Document Processing & Ingestion | Data/Ingestion Engineer | `app/ingestion/`, `scripts/ingest_documents.py` |
| **Module 2** | Embedding & Vector Storage | Vector/Storage Engineer | `app/vectorstore/`, `app/providers/ollama/embeddings.py` |
| **Module 3** | Retrieval & RAG Pipeline | RAG/Search Engineer | `app/services/retrieval_service.py`, `app/services/rag_service.py` |
| **Module 4** | LLM Integration & Prompt Engineering | GenAI/Prompt Engineer | `app/providers/ollama/generation.py`, `app/prompts/`, `app/services/llm_service.py` |
| **Module 5** | Conversation Context & API Routing | Backend/API Engineer | `app/api/`, `app/services/conversation_service.py`, `app/main.py` |
| **Module 6** | Evaluation & Quality Assurance | QA/ML Evaluation Engineer | `app/evaluation/`, `app/services/evaluation_service.py`, `tests/` |

---

## Detailed Module Specifications

### Module 1: Document Processing and Ingestion
* **Responsibility**: Extracting raw text from various document formats (.pdf, .docx, .txt, .md, .csv), cleaning noise, computing deduplication hashes, and splitting text into cohesive chunks.
* **Input**: File path or raw text stream, chunking parameters (`chunk_size`, `chunk_overlap`).
* **Output**: List of `DocumentChunk` objects enriched with page numbers, document IDs, and traceability metadata.
* **Files Owned**:
  - `app/ingestion/loaders/` (`base.py`, `text_loader.py`, `pdf_loader.py`, `docx_loader.py`)
  - `app/ingestion/processor.py`
  - `app/ingestion/chunker.py`
  - `app/ingestion/metadata.py`
  - `app/ingestion/pipeline.py`
  - `scripts/ingest_documents.py`
  - `tests/unit/test_loaders.py`, `tests/unit/test_chunker.py`, `tests/unit/test_processor.py`
* **Dependencies**: `pypdf`, `python-docx`, `app.utils.text_cleaner`, `app.vectorstore.base`.
* **Public Interfaces**:
  - `BaseLoader.load(file_path: Path) -> RawDocument`
  - `DocumentProcessor.process(raw_doc: RawDocument) -> RawDocument`
  - `DocumentChunker.chunk_document(doc_id: str, raw_doc: RawDocument) -> List[DocumentChunk]`
  - `IngestionPipeline.ingest_file(file_path: Path) -> Dict[str, Any]`
  - `IngestionPipeline.ingest_raw_text(title: str, content: str) -> Dict[str, Any]`
* **Tests Required**: Parser integrity for valid/corrupt PDFs and DOCX, sliding window overlap accuracy, deduplication hashing stability.
* **Suggested First Task**: Add HTML or Markdown table extractor to `TextLoader`.

---

### Module 2: Embedding and Vector Storage
* **Responsibility**: Generating vector representations via local Ollama (`nomic-embed-text`) and managing persistent ChromaDB collections, indices, search, and chunk deletion.
* **Input**: Text chunks or search queries (strings), precomputed float vectors.
* **Output**: Vector IDs, similarity-ranked `VectorSearchResult` lists (with distance and similarity scores).
* **Files Owned**:
  - `app/vectorstore/base.py`
  - `app/vectorstore/chroma_store.py`
  - `app/providers/ollama/embeddings.py`
  - `app/services/embedding_service.py`
  - `tests/unit/test_chroma_store.py`
* **Dependencies**: `chromadb`, `httpx`, `app.core.config`, `app.core.exceptions`.
* **Public Interfaces**:
  - `BaseVectorStore.add_documents(docs: List[VectorDocument], embeddings: List[List[float]]) -> List[str]`
  - `BaseVectorStore.similarity_search(query_embedding: List[float], top_k: int) -> List[VectorSearchResult]`
  - `BaseVectorStore.delete_by_document_id(document_id: str) -> int`
  - `EmbeddingService.embed_text(text: str) -> List[float]`
* **Tests Required**: ChromaDB persistence across restarts, collection isolation, similarity score monotonicity, embedding dimension verification.
* **Suggested First Task**: Implement batch embedding chunking for large document volumes to optimize Ollama throughput.

---

### Module 3: Retrieval and RAG Pipeline
* **Responsibility**: Orchestrating semantic search, filtering retrieved chunks against confidence thresholds, constructing grounded prompt contexts, verifying grounding, and compiling citation metadata.
* **Input**: User `ChatRequest` (validated query + conversation history).
* **Output**: Complete `ChatResponse` with citations, grounding confidence score, and latency metrics.
* **Files Owned**:
  - `app/services/retrieval_service.py`
  - `app/services/rag_service.py`
  - `tests/unit/test_retrieval_service.py`, `tests/unit/test_rag_service.py`
* **Dependencies**: `EmbeddingService`, `BaseVectorStore`, `LLMService`, `ConversationService`.
* **Public Interfaces**:
  - `RetrievalService.retrieve(query: str, top_k: int, threshold: float) -> RetrievalResult`
  - `RAGService.answer_query(request: ChatRequest) -> ChatResponse`
* **Tests Required**: Fallback invocation when context is absent or below threshold, citation mapping accuracy, prompt construction fidelity.
* **Suggested First Task**: Implement reciprocal rank re-ranking or dynamic threshold adjustments based on query type.

---

### Module 4: LLM Integration and Prompt Engineering
* **Responsibility**: Managing the Ollama runtime connection, system prompt templates, generation parameters, ambiguity clarification templates, and anti-hallucination directives.
* **Input**: Prompt strings, temperature, max generation tokens.
* **Output**: Generated text completion dictionary (tokens, latency, model identifier).
* **Files Owned**:
  - `app/providers/ollama/client.py`
  - `app/providers/ollama/generation.py`
  - `app/services/llm_service.py`
  - `app/prompts/system.txt`, `app/prompts/rag.txt`, `app/prompts/clarification.txt`, `app/prompts/fallback.txt`
  - `scripts/setup_ollama.sh`
  - `tests/unit/test_ollama_client.py`
* **Dependencies**: `httpx`, `app.core.config`, `app.core.exceptions`.
* **Public Interfaces**:
  - `OllamaClient.check_health() -> Dict[str, Any]`
  - `OllamaClient.list_local_models() -> List[str]`
  - `LLMService.generate_response(prompt: str, temperature: float) -> Dict[str, Any]`
* **Tests Required**: Connection failure retries, timeout handling, non-streaming and streaming generation parsers.
* **Suggested First Task**: Benchmark response latency across different `qwen2.5:7b` quantization levels (q4_k_m vs q8_0).

---

### Module 5: Conversation Context and API Routing
* **Responsibility**: FastAPI endpoint definitions, Pydantic request/response validation, sliding-window session memory, global exception translation, and documentation schemas.
* **Input**: HTTP REST requests from client or upstream backend gateway.
* **Output**: Standard JSON HTTP responses, OpenAPI schema (`/docs`).
* **Files Owned**:
  - `app/api/` (`routes/`, `schemas/`)
  - `app/services/conversation_service.py`
  - `app/utils/validators.py`, `app/utils/text_cleaner.py`
  - `app/main.py`
  - `tests/integration/test_api_*.py`
* **Dependencies**: `FastAPI`, `Pydantic`, `RAGService`, `IngestionPipeline`.
* **Public Interfaces**:
  - `POST /api/v1/chat`
  - `POST /api/v1/knowledge/ingest`
  - `GET /api/v1/knowledge`
  - `DELETE /api/v1/knowledge/{document_id}`
  - `GET /api/v1/health`, `GET /api/v1/health/ollama`
* **Tests Required**: Status code assertions (200, 201, 400, 404, 415, 503), multi-turn session persistence, schema validation edge cases.
* **Suggested First Task**: Add conversation export/import endpoints for user chat history download.

---

### Module 6: Evaluation and Quality Assurance
* **Responsibility**: Measuring retrieval quality (Hit@K, MRR), assessing response grounding and lexical faithfulness, maintaining reference benchmark datasets, and preventing regression.
* **Input**: Benchmark query pairs, generated answers, retrieved context chunks, reference ground truth.
* **Output**: Evaluation scorecards (Hit Rate, MRR, Groundedness ratio, F1 overlap).
* **Files Owned**:
  - `app/evaluation/` (`metrics.py`, `test_cases/benchmark_cases.json`)
  - `app/services/evaluation_service.py`
  - `tests/unit/test_evaluation.py`
* **Dependencies**: `RetrievalService`, standard Python statistical utilities.
* **Public Interfaces**:
  - `RetrievalMetrics.hit_rate(retrieved: List[str], ground_truth: List[str]) -> float`
  - `GenerationMetrics.groundedness_score(answer: str, context: str) -> float`
  - `EvaluationService.evaluate_retrieval(test_queries: List[Dict]) -> Dict[str, float]`
* **Tests Required**: Groundedness calculations on synthetic hallucinated text vs grounded text, MRR ranking accuracy.
* **Suggested First Task**: Expand `app/evaluation/test_cases/benchmark_cases.json` with 25 domain-specific test scenarios.

