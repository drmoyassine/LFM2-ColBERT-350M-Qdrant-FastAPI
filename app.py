import os
from typing import List
from fastapi import FastAPI, HTTPException, Depends, Security
from pydantic import BaseModel
from fastapi.security import APIKeyHeader
from pylate import models
from qdrant_client import QdrantClient

app = FastAPI(title="LFM2-ColBERT-350M + Qdrant API")

API_KEY = os.environ.get("API_KEY", "change_this_key")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

MODEL_NAME = os.environ.get("MODEL_NAME", "LiquidAI/LFM2-ColBERT-350M")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "colbert_docs")
VECTOR_SIZE = int(os.environ.get("VECTOR_SIZE", 128))

# Load model and qdrant client at startup
model = models.ColBERT(model_name_or_path=MODEL_NAME)
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "default": {
            "size": VECTOR_SIZE,
            "distance": "Cosine"
        }
    }
)

class IndexRequest(BaseModel):
    doc_id: str
    text: str

class QueryRequest(BaseModel):
    query_texts: List[str]
    top_k: int = 3

class BatchIndexRequest(BaseModel):
    docs: List[IndexRequest]

class BatchQueryRequest(BaseModel):
    queries: List[str]
    top_k: int = 3

@app.get("/health")
async def health():
    try:
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        status = "ok"
        details = {
            "qdrant_status": "reachable",
            "collection_points_count": collection_info.points_count,
        }
    except Exception as e:
        status = "error"
        details = {"qdrant_status": "unreachable", "error": str(e)}
    return {"status": status, "details": details}

@app.post("/index/")
async def add_document(req: IndexRequest, _=Depends(verify_api_key)):
    emb = model.encode([req.text], is_query=False)[0]
    pooled_emb = emb.mean(axis=0)
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[{"id": req.doc_id, "vector": pooled_emb.tolist(), "payload": {"text": req.text}}]
    )
    return {"message": "Indexed", "id": req.doc_id}

@app.post("/search/")
async def search_documents(req: QueryRequest, _=Depends(verify_api_key)):
    results = []
    for query in req.query_texts:
        emb = model.encode([query], is_query=True)[0]
        pooled_emb = emb.mean(axis=0)
        hits = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=pooled_emb.tolist(),
            limit=req.top_k
        )
        found = [{"id": h.id, "score": h.score, "payload": h.payload} for h in hits]
        results.append({"query": query, "results": found})
    return results

@app.post("/batch_index/")
async def batch_index(req: BatchIndexRequest, _=Depends(verify_api_key)):
    ids = [d.doc_id for d in req.docs]
    texts = [d.text for d in req.docs]
    embs = model.encode(texts, is_query=False)
    points = [
        {"id": ids[i], "vector": embs[i].mean(axis=0).tolist(), "payload": {"text": texts[i]}}
        for i in range(len(ids))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return {"success": True, "count": len(ids)}

@app.post("/batch_search/")
async def batch_search(req: BatchQueryRequest, _=Depends(verify_api_key)):
    results = []
    embs = model.encode(req.queries, is_query=True)
    for i, query in enumerate(req.queries):
        hits = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embs[i].mean(axis=0).tolist(),
            limit=req.top_k
        )
        found = [{"id": h.id, "score": h.score, "payload": h.payload} for h in hits]
        results.append({"query": query, "results": found})
    return results
