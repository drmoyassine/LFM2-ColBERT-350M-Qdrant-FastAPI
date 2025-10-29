# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Stack & Technology
- Python 3.10+ FastAPI application with Uvicorn ASGI server
- ColBERT embedding model (LiquidAI/LFM2-ColBERT-350M) via pylate library
- Qdrant vector database for similarity search
- Docker containerized deployment with docker-compose
- Single-file API (`app.py`) with REST endpoints for indexing and searching

## Critical Commands
- Run via Docker: `docker compose up --build`
- Development server: `uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4`
- Health check endpoint: GET `/health` (returns Qdrant status and collection info)

## Non-Obvious Patterns Discovered
- Collection recreated on startup (line 27-35 in app.py), clearing previous data
- Embeddings use mean pooling (`.mean(axis=0)`) for document vectors, not ColBERT's native multi-vector approach
- API key authentication required for all write/search operations (X-API-Key header)
- Environment variables control Qdrant connection, model name, collection name, and vector size
- Batch operations encode texts once then process individually, avoiding redundant model calls

## Code Style Notes
- Uses Pydantic models for request/response validation
- Environment variables with fallback defaults (API_KEY defaults to "change_this_key")
- Type hints throughout (List, etc.)
- Async endpoints but synchronous model operations