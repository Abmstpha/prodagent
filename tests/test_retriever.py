"""Retrieval regression: the production pipeline must place the right document in the top-3."""
import pytest

from src.corpus import EVAL_QUESTIONS
from src.retriever import baseline_retrieve, production_retrieve

EXPECTED_MARKER = ["1,419", "0.033", "14%", "40% to 60%", "Three Gorges"]


@pytest.mark.parametrize("case,marker", list(zip(EVAL_QUESTIONS, EXPECTED_MARKER)),
                         ids=[c["question"][:40] for c in EVAL_QUESTIONS])
def test_production_hit_at_3(case, marker):
    contexts = production_retrieve(case["question"], k_final=3)
    assert any(marker in c for c in contexts), \
        f"expected '{marker}' in top-3 for: {case['question']}"


def test_baseline_returns_documents():
    assert len(baseline_retrieve("solar capacity", k=3)) == 3


def test_production_returns_strings():
    for c in production_retrieve("hydropower storage", k_final=3):
        assert isinstance(c, str) and c
