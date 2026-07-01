import json
import logging
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding

log = logging.getLogger(__name__)

INDEX_DIR = Path(__file__).parent.parent / "data" / "faiss_index"
INDEX_PATH = INDEX_DIR / "index.faiss"
META_PATH = INDEX_DIR / "metadata.json"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

class Retriever:
    def __init__(self):
        if not INDEX_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(f"FAISS index files not found in {INDEX_DIR}")

        log.info("Loading FAISS search index...")
        self.index = faiss.read_index(str(INDEX_PATH))

        with open(META_PATH, encoding="utf-8") as f:
            self.catalog = json.load(f)

        self.embedder = TextEmbedding(EMBED_MODEL)

    def retrieve(self, query: str, top_k: int = 15) -> list[dict]:
        """Runs a semantic cosine similarity search over catalog items."""
        if not query or not query.strip():
            return []

        top_k = min(top_k, self.index.ntotal)

        # Generate query embedding
        emb = list(self.embedder.embed([query]))[0].astype(np.float32)
        emb = emb.reshape(1, -1)
        faiss.normalize_L2(emb)

        # Search index
        scores, indices = self.index.search(emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self.catalog[idx])
            item["_score"] = float(score)
            results.append(item)

        return results

    def get_by_url(self, url: str) -> dict | None:
        if not url:
            return None
        target = url.rstrip("/")
        for item in self.catalog:
            if item.get("url", "").rstrip("/") == target:
                return item
        return None

    def get_by_name(self, name: str) -> dict | None:
        if not name:
            return None
        target = name.lower().strip()
        for item in self.catalog:
            if item.get("name", "").lower().strip() == target:
                return item
        return None

@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()

