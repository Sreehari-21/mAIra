# GenAI Platform — Local Development Guide

This guide walks engineers through setting up a clean local development environment, running tests, following team conventions, and contributing to the Conversational AI Platform.

---

## 1. Prerequisites

Before starting, ensure your local system meets the following prerequisites:
* **Operating System**: macOS or Linux (Ubuntu 20.04+)
* **Python**: Python 3.9, 3.10, or 3.11 installed
* **Ollama**: Installed locally via [ollama.com](https://ollama.com)
* **Git**: Version control
* **Disk Space**: ~8GB free disk space (to store `qwen2.5:7b` (~4.7GB) and `nomic-embed-text` (~274MB))

---

## 2. Environment Setup

### 2.1 Clone Repository and Create Virtual Environment

```bash
cd genai-platform

# Create Python virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 2.2 Configure Local Environment Variables

Copy the template configuration into `.env`:

```bash
cp .env.example .env
```

Review `.env` settings:
* `OLLAMA_BASE_URL="http://localhost:11434"`
* `OLLAMA_GENERATION_MODEL="qwen2.5:7b"`
* `OLLAMA_EMBEDDING_MODEL="nomic-embed-text"`
* `CHROMA_PERSIST_DIRECTORY="./data/chroma"`

---

## 3. Ollama Runtime Setup

Start the Ollama daemon in a separate terminal:

```bash
ollama serve
```

Execute the platform setup script to download and verify the initial models:

```bash
./scripts/setup_ollama.sh
```

Or pull the models manually:

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
ollama list
```

---

## 4. Running the Application

Start the FastAPI application with live code reloading:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Once running:
* **Interactive OpenAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Alternative Redoc Docs**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
* **Health Check**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)
* **Ollama Runtime Diagnostic**: [http://localhost:8000/api/v1/health/ollama](http://localhost:8000/api/v1/health/ollama)

---

## 5. Running Tests

The test suite contains both fast unit tests (with mocked Ollama calls) and full integration tests.

```bash
# Run the entire test suite
pytest

# Run only unit tests
pytest tests/unit

# Run only integration tests
pytest tests/integration

# Run tests with detailed output
pytest -v -s
```

---

## 6. Code Quality & Standards

Every contribution must comply with the following standards:
1. **Type Annotations**: All function signatures and class attributes must include Python type hints.
2. **Strict Local Execution**: Do NOT introduce API calls to OpenAI, Gemini, or third-party cloud services.
3. **No Monolithic Files**: Maintain single responsibility. Keep business logic in `services/`, loaders in `ingestion/loaders/`, and routes thin.
4. **Error Handling**: Use domain exceptions defined in `app/core/exceptions.py`. Do not return generic 500 errors.
5. **Formatting**:
   ```bash
   black app tests
   isort app tests
   flake8 app tests
   ```

---

## 7. Working with Docker (Optional)

You can run the complete stack (FastAPI + Ollama) using Docker Compose:

```bash
# Start all containers in background
docker compose up -d

# Check logs
docker compose logs -f api

# Stop containers
docker compose down
```

