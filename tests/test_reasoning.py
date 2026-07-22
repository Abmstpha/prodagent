"""Reasoning units: format extraction and the Lab-3 lesson — vote on meaning, not wording."""
from src.reasoning import extract_conclusion, extract_confidence, majority_vote

ANSWER = """EVIDENCE:
  - wind.md: onshore LCOE $0.033/kWh
ANALYSIS:
  Step 1: single source
CONCLUSION: Onshore wind is the cheapest source.
CONFIDENCE: MEDIUM — single source in context."""


def test_extract_conclusion():
    assert extract_conclusion(ANSWER) == "Onshore wind is the cheapest source."


def test_extract_confidence():
    assert extract_confidence(ANSWER) == "MEDIUM"


def test_majority_vote_groups_equivalent_wordings():
    conclusions = [
        "Yes, onshore wind is the cheapest electricity source at $0.033 per kWh.",
        "Onshore wind is the cheapest source of electricity (about $0.033/kWh).",
        "Insufficient data in the corpus to answer.",
    ]
    winner, agreement = majority_vote(conclusions)
    assert winner in (0, 1)
    assert agreement == 2


def test_majority_vote_single_answer():
    winner, agreement = majority_vote(["Only answer."])
    assert winner == 0 and agreement == 1
