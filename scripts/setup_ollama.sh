#!/usr/bin/env bash
# ==============================================================================
# Setup & Verification Script for Local Ollama LLM Runtime
# Phase 1: Local Ollama only (Strictly No External / Paid Cloud APIs)
# ==============================================================================

set -euo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
GENERATION_MODEL="${1:-qwen2.5:7b}"
EMBEDDING_MODEL="${2:-nomic-embed-text}"

echo "=================================================================="
echo "    GenAI Conversational AI Platform - Ollama Runtime Setup       "
echo "=================================================================="
echo "Endpoint:         ${OLLAMA_HOST}"
echo "Generation Model: ${GENERATION_MODEL}"
echo "Embedding Model:  ${EMBEDDING_MODEL}"
echo "------------------------------------------------------------------"

# Step 1: Check CLI installation
if ! command -v ollama &> /dev/null; then
    echo "[!] Ollama CLI is not found in your PATH."
    echo "    Please install Ollama from https://ollama.ai/download or run:"
    echo "    - macOS: brew install ollama"
    echo "    - Linux: curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

echo "[✓] Ollama CLI detected at: $(which ollama)"

# Step 2: Check whether daemon is running
echo "[*] Checking Ollama daemon connectivity..."
if ! curl -s --max-time 3 "${OLLAMA_HOST}/api/version" &> /dev/null; then
    echo "[!] Ollama daemon does not appear to be running on ${OLLAMA_HOST}."
    echo "    Start the Ollama daemon in a separate terminal using:"
    echo "        ollama serve"
    echo ""
    echo "    Once started, re-run this setup script."
    exit 1
fi
echo "[✓] Ollama daemon is active and responding."

# Step 3: Pull generation model
echo "[*] Ensuring generation model '${GENERATION_MODEL}' is downloaded..."
ollama pull "${GENERATION_MODEL}"
echo "[✓] Generation model '${GENERATION_MODEL}' ready."

# Step 4: Pull embedding model
echo "[*] Ensuring embedding model '${EMBEDDING_MODEL}' is downloaded..."
ollama pull "${EMBEDDING_MODEL}"
echo "[✓] Embedding model '${EMBEDDING_MODEL}' ready."

# Step 5: Verification
echo ""
echo "=================================================================="
echo "    Local Models Available in Ollama:                             "
echo "=================================================================="
ollama list

echo ""
echo "[✓] Setup complete! You can now start the FastAPI backend:"
echo "    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

