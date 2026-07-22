"""RAGAS evaluation: baseline (TF-IDF + direct answer) vs improved (hybrid+rerank + CoT/Self-Consistency).

Run:  python -m src.evaluate
Needs: pip install ragas langchain-openai "langchain-community<0.4" datasets
Writes docs/ragas_results.json and prints the REPORT.md table.
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.corpus import EVAL_QUESTIONS  # noqa: E402
from src.llm_helpers import make_client  # noqa: E402
from src.reasoning import SYSTEM_SYNTHESIS, self_consistent_answer  # noqa: E402
from src.retriever import baseline_retrieve, production_retrieve  # noqa: E402

BASELINE_SYSTEM = "Answer the question using the provided context."


def build_rows(client, improved: bool) -> list:
    rows = []
    for case in EVAL_QUESTIONS:
        q = case["question"]
        if improved:
            contexts = production_retrieve(q, k_final=3)
            answer = self_consistent_answer(client, q, "\n---\n".join(contexts), k=3)["answer"]
        else:
            contexts = baseline_retrieve(q, k=3)
            reply = client.complete(
                [{"role": "system", "content": BASELINE_SYSTEM},
                 {"role": "user", "content":
                     f"CONTEXT:\n{chr(10).join(contexts)}\n\nQUESTION: {q}"}],
                temperature=0.1, max_tokens=512)
            answer = reply.content or ""
        rows.append({"user_input": q, "retrieved_contexts": contexts,
                     "response": answer, "reference": case["ground_truth"]})
        print(f"  [{'improved' if improved else 'baseline'}] {q[:50]}… ok")
    return rows


def ragas_scores(rows: list) -> dict:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference,
                               LLMContextRecall, ResponseRelevancy)
    from ragas.run_config import RunConfig

    api_key = os.environ["MISTRAL_API_KEY"]
    judge = LangchainLLMWrapper(ChatOpenAI(
        model="mistral-large-latest", api_key=api_key,
        base_url="https://api.mistral.ai/v1", temperature=0, max_retries=10))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model="mistral-embed", api_key=api_key,
        base_url="https://api.mistral.ai/v1", check_embedding_ctx_length=False))

    result = evaluate(
        dataset=EvaluationDataset.from_list(rows),
        metrics=[LLMContextRecall(), LLMContextPrecisionWithReference(),
                 Faithfulness(), ResponseRelevancy(strictness=1)],  # Mistral caps n at 1
        llm=judge, embeddings=emb,
        run_config=RunConfig(max_workers=1, timeout=180, max_retries=10),
    )
    df = result.to_pandas()
    cols = [c for c in df.columns if c not in
            ("user_input", "retrieved_contexts", "response", "reference")]
    return {c: round(float(df[c].mean()), 3) for c in cols}


def main():
    client = make_client(quiet=True)
    print("Generating baseline answers…")
    baseline_rows = build_rows(client, improved=False)
    print("Generating improved answers…")
    improved_rows = build_rows(client, improved=True)

    print("Scoring baseline with RAGAS…")
    baseline = ragas_scores(baseline_rows)
    print("Scoring improved with RAGAS…")
    improved = ragas_scores(improved_rows)

    out = {"baseline": baseline, "improved": improved,
           "n_questions": len(EVAL_QUESTIONS),
           "judge": "mistral-large-latest", "embeddings": "mistral-embed"}
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "ragas_results.json").write_text(json.dumps(out, indent=2))

    print("\n| Metric | Baseline | Improved |")
    print("|---|---|---|")
    for metric in baseline:
        print(f"| {metric} | {baseline[metric]} | {improved.get(metric, '—')} |")
    print("\nSaved to docs/ragas_results.json")


if __name__ == "__main__":
    main()
