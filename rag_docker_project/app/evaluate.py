"""Evaluate retrieval on data/eval_questions.json.

Run:  python app/evaluate.py            (locally)
      docker compose exec rag python app/evaluate.py   (inside the container)

A retrieved chunk counts as relevant if it comes from the expected source file.
Metrics: Precision@k, Recall@k (document level) and Hit Rate@k.
"""
import json
from pathlib import Path

from rag import RAGPipeline, DEFAULT_DATA_DIR


def main():
    rag = RAGPipeline()
    questions = json.loads((Path(DEFAULT_DATA_DIR) / "eval_questions.json").read_text(encoding="utf-8"))
    k = rag.top_k
    precisions, recalls, hits = [], [], []

    for q in questions:
        retrieved = rag.retrieve(q["question"], k)
        sources = [c.source for c, _ in retrieved]
        relevant = [s == q["expected_source"] for s in sources]
        precisions.append(sum(relevant) / k)
        recalls.append(1.0 if any(relevant) else 0.0)  # one relevant document per question
        hits.append(1.0 if any(relevant) else 0.0)
        print(f"{'OK ' if any(relevant) else 'MISS'} | {q['question']}  ->  {sources}")

    n = len(questions)
    print(f"\nQuestions: {n} | k = {k}")
    print(f"Precision@{k}: {sum(precisions)/n:.3f}")
    print(f"Recall@{k}:    {sum(recalls)/n:.3f}")
    print(f"Hit rate@{k}:  {sum(hits)/n:.3f}")
    print("Answer relevance and faithfulness need manual review or an LLM judge.")


if __name__ == "__main__":
    main()
