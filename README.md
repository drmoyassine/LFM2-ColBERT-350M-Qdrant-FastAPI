# LFM2-ColBERT-350M + Qdrant API

A production-ready stack for multilingual document retrieval: LFM2-ColBERT-350M (`pylate`) + FastAPI + Qdrant, containerized with Docker Compose.

---

## Features

- Sentence/document embedding via `LFM2-ColBERT-350M`
- Storage and retrieval in Qdrant vector DB
- Simple REST API for indexing and search

## Getting Started

### 1. Clone this repo

git clone <REPO_URL>
cd lfm2-colbert-qdrant-api

### 2. Build and Launch

docker compose up --build

FastAPI available at [http://localhost:8000](http://localhost:8000)  
Qdrant available at [http://localhost:6333](http://localhost:6333)

### 3. Example API Usage

**Index a Document**
curl -X POST localhost:8000/index/
-H "Content-Type: application/json"
-d '{"doc_id":"doc1","text":"Example document here."}'

**Search for Documents**
curl -X POST localhost:8000/search/
-H "Content-Type: application/json"
-d '{"query_texts":["search phrase"],"top_k":3}'


## File Structure

- `app.py`: FastAPI server with endpoints
- `requirements.txt`: Python dependencies
- `Dockerfile`: FastAPI app container
- `docker-compose.yml`: Qdrant & API multi-container setup

---

## Notes

- The script stores embeddings with *mean pooling*; for ColBERT-style late interaction or multi-vector search, further adaptation is possible.
- Modify collection settings in `app.py` as needed.

---