"""
Lazy-initialized singletons for heavy resources (models, DB).
Replaces @st.cache_resource from app.py.
"""
import hashlib
import json
import os

from .config import (
    EMBED_MODEL_PATH, BASE_MODEL_EMBED,
    SUMMARY_MODEL_PATH, CLASSIFY_MODEL_PATH,
    CHROMA_DB_PATH, NOTICES_CACHE_PATH,
)
from .utils import infer_category, chunk_text

_embed_model      = None
_summary_pipeline = None
_classifier       = None
_label_map: dict  = {}
_chroma           = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        path = EMBED_MODEL_PATH if os.path.exists(EMBED_MODEL_PATH) else BASE_MODEL_EMBED
        _embed_model = SentenceTransformer(path, device="cpu")
    return _embed_model


def get_summary_pipeline():
    global _summary_pipeline
    if _summary_pipeline is None and os.path.exists(SUMMARY_MODEL_PATH):
        from transformers import pipeline
        _summary_pipeline = pipeline(
            "summarization", model=SUMMARY_MODEL_PATH,
            tokenizer=SUMMARY_MODEL_PATH, max_new_tokens=128, device=-1,
        )
    return _summary_pipeline


def get_classifier():
    global _classifier, _label_map
    if _classifier is None and os.path.exists(CLASSIFY_MODEL_PATH):
        from transformers import pipeline
        _classifier = pipeline(
            "text-classification", model=CLASSIFY_MODEL_PATH,
            tokenizer=CLASSIFY_MODEL_PATH, device=-1,
        )
        label_map_path = os.path.join(CLASSIFY_MODEL_PATH, "label_map.json")
        if os.path.exists(label_map_path):
            with open(label_map_path) as f:
                _label_map = json.load(f)
    return _classifier, _label_map


def get_chroma():
    global _chroma
    if _chroma is None:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _chroma = client.get_or_create_collection(
            name="hansung_notices",
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma


def classify_notice(title: str, body: str) -> str:
    clf, label_map = get_classifier()
    if clf is None:
        return infer_category(title, body)
    try:
        result   = clf(f"{title} {body[:200]}", truncation=True)[0]
        label_id = result["label"].replace("LABEL_", "")
        return label_map.get(label_id, "기타")
    except Exception:
        return infer_category(title, body)


def load_notices_cache() -> list[dict]:
    if os.path.exists(NOTICES_CACHE_PATH):
        with open(NOTICES_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def index_notices(notices: list[dict]) -> int:
    """Index notices into ChromaDB. Returns count of newly indexed documents."""
    model      = get_embed_model()
    collection = get_chroma()
    new_count  = 0

    for item in notices:
        doc_id   = hashlib.md5(item["url"].encode()).hexdigest()
        body     = item.get("body", "")
        category = item.get("category") or classify_notice(item["title"], body)
        chunks   = chunk_text(f"제목: {item['title']}\n\n{body}")

        if collection.get(ids=[f"{doc_id}_0"])["ids"]:
            continue
        if collection.get(ids=[doc_id])["ids"]:
            collection.delete(ids=[doc_id])

        meta = {
            "title":    item["title"],
            "url":      item["url"],
            "date":     item["date"],
            "category": category,
        }
        collection.add(
            ids        = [f"{doc_id}_{i}" for i in range(len(chunks))],
            embeddings = model.encode(chunks).tolist(),
            documents  = chunks,
            metadatas  = [meta] * len(chunks),
        )
        new_count += 1

    return new_count
