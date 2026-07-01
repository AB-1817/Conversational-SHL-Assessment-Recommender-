import json
import logging
import sys
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CATALOG_PATH = Path(__file__).parent.parent / "data" / "catalog.json"
INDEX_DIR = Path(__file__).parent.parent / "data" / "faiss_index"
INDEX_PATH = INDEX_DIR / "index.faiss"
META_PATH = INDEX_DIR / "metadata.json"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

def build_document(item: dict) -> str:
    """Formats assessment fields into a single text representation for embeddings."""
    parts = [f"Assessment: {item['name']}."]

    if item.get("test_type"):
        parts.append(f"Type: {_type_label(item['test_type'])}.")

    if item.get("duration") and item["duration"] not in ("-", "—", ""):
        parts.append(f"Duration: {item['duration']}.")

    if item.get("languages"):
        parts.append(f"Languages: {item['languages']}.")

    if item.get("remote_testing"):
        parts.append("Supports remote testing.")

    if item.get("adaptive_irt"):
        parts.append("Adaptive/IRT test.")

    if item.get("description"):
        parts.append(item["description"])

    return " ".join(parts)

def _type_label(codes: str) -> str:
    mapping = {
        "A": "Ability and Aptitude",
        "B": "Biodata and Situational Judgment",
        "C": "Competencies",
        "D": "Development and 360",
        "E": "Assessment Exercises",
        "K": "Knowledge and Skills",
        "M": "Motivation",
        "P": "Personality and Behavior",
        "S": "Simulations",
    }
    labels = [mapping.get(c.strip(), c.strip()) for c in codes.split(",") if c.strip()]
    return ", ".join(labels)

def main():
    if not CATALOG_PATH.exists():
        log.error(f"Catalog JSON file not found at {CATALOG_PATH}")
        sys.exit(1)

    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)

    # Filter invalid entries
    catalog = [c for c in catalog if c.get("url") and c.get("name")]
    log.info(f"Loaded {len(catalog)} valid catalog items")

    # Generate documents
    documents = [build_document(item) for item in catalog]

    log.info(f"Initializing embedding model: {EMBED_MODEL}")
    embedder = TextEmbedding(EMBED_MODEL)

    log.info("Generating embeddings...")
    embeddings = list(embedder.embed(documents))
    matrix = np.array(embeddings, dtype=np.float32)

    # Normalize vectors for cosine similarity (Inner Product)
    faiss.normalize_L2(matrix)

    # Setup FAISS flat inner-product index
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    log.info(f"Built index with {index.ntotal} vectors")

    # Save outputs
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    log.info("Index files successfully created")

if __name__ == "__main__":
    main()

