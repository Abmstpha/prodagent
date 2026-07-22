"""Optional Pinecone dense leg for the hybrid retriever.

When PINECONE_API_KEY is set, child chunks live in a Pinecone index with integrated
embeddings (llama-text-embed-v2) and dense search runs there; without the key the
retriever falls back to the local TF-IDF dense proxy, so tests and CI stay offline.
"""
import os

INDEX_NAME = "prodagent"
NAMESPACE = "corpus"

_index = None


def _get_index():
    global _index
    if _index is not None:
        return _index
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        return None
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=api_key)
        _index = pc.Index(INDEX_NAME)
        return _index
    except Exception:
        return None


def pinecone_search(query: str, k: int = 10) -> list:
    """Dense search over child chunks. Returns [(score, child_text), ...]; [] on any failure."""
    index = _get_index()
    if index is None:
        return []
    try:
        res = index.search(namespace=NAMESPACE,
                           query={"top_k": k, "inputs": {"text": query}})
        hits = res.get("result", {}).get("hits", [])
        return [(h.get("_score", 0.0), h.get("fields", {}).get("chunk_text", ""))
                for h in hits if h.get("fields", {}).get("chunk_text")]
    except Exception:
        return []


def setup_index(children: dict) -> str:
    """One-time: create the index (integrated embedding) and upsert all child chunks."""
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if not pc.has_index(INDEX_NAME):
        pc.create_index_for_model(
            name=INDEX_NAME, cloud="aws", region="us-east-1",
            embed={"model": "llama-text-embed-v2", "field_map": {"text": "chunk_text"}},
        )
    index = pc.Index(INDEX_NAME)
    records = [{"_id": cid, "chunk_text": text} for cid, text in children.items()]
    for i in range(0, len(records), 90):
        index.upsert_records(namespace=NAMESPACE, records=records[i:i + 90])
    return f"{len(records)} child chunks upserted to '{INDEX_NAME}/{NAMESPACE}'"


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from dotenv import load_dotenv
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    from src.corpus import CHILDREN
    print(setup_index(CHILDREN))
