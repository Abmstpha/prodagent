"""Corpus loading and parent-child chunking (Lab 1 pipeline)."""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_corpus() -> dict:
    """doc_id -> full text, from every .md file in data/."""
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(DATA_DIR.glob("*.md"))}


def split_words(text: str, size: int, overlap: int) -> list:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return chunks


def build_chunks(corpus: dict, parent_size: int = 80, child_size: int = 20):
    """Two-level index: children (retrieval granularity) -> parents (context granularity)."""
    children, parents, child_to_parent = {}, {}, {}
    for doc_id, text in corpus.items():
        for p_idx, parent_text in enumerate(split_words(text, parent_size, 10)):
            parent_id = f"{doc_id}_p{p_idx}"
            parents[parent_id] = parent_text
            for c_idx, child_text in enumerate(split_words(parent_text, child_size, 3)):
                child_id = f"{parent_id}_c{c_idx}"
                children[child_id] = child_text
                child_to_parent[child_id] = parent_id
    return children, parents, child_to_parent


CORPUS = load_corpus()
CHILDREN, PARENTS, CHILD_TO_PARENT = build_chunks(CORPUS)

# 5 evaluation questions with ground truths (RAGAS + regression tests)
EVAL_QUESTIONS = [
    {"question": "What was the global installed solar PV capacity at the end of 2023?",
     "ground_truth": "About 1,419 GW of solar PV capacity was installed globally at the end of 2023."},
    {"question": "Which electricity source has the lowest average LCOE?",
     "ground_truth": "Onshore wind, at about $0.033 per kWh in 2023, the lowest of all electricity sources."},
    {"question": "What share of global electricity does hydropower generate?",
     "ground_truth": "Hydropower generated around 14% of global electricity in 2023, about 4,300 TWh."},
    {"question": "What capacity factors do offshore wind farms achieve?",
     "ground_truth": "Offshore wind farms reach capacity factors of 40% to 60%."},
    {"question": "What is the largest power station in the world?",
     "ground_truth": "The Three Gorges Dam in China, at 22.5 GW."},
]
