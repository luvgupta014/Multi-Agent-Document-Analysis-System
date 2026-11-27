"""
Retrieval evaluation: Recall@k and MRR against the golden QA set.

Usage:
    # From project root with venv active:
    python eval/run_eval.py

    # Adjust k values or skip Ragas with flags:
    python eval/run_eval.py --k 1 3 5 10 --no-ragas

Output:
    eval/results.md   - full report (overwritten on each run)
    stdout            - summary table
"""
import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ---------------------------------------------------------------------------
# Project root on sys.path so rag_system imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_system.knowledge_base.vector_store import VectorStore
from rag_system.knowledge_base.loader import DocumentLoader
from rag_system.knowledge_base.retriever import HybridRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> Dict:
    cfg_path = PROJECT_ROOT / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def load_golden_qa() -> List[Dict]:
    qa_path = Path(__file__).parent / "golden_qa.yaml"
    with open(qa_path) as f:
        data = yaml.safe_load(f)
    return data["questions"]


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """1.0 if any relevant chunk appears in the top-k retrieved ids, else 0.0."""
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k & set(relevant_ids) else 0.0


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """1/rank of the first relevant chunk; 0 if none found."""
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def run_retrieval_eval(
    questions: List[Dict],
    vector_store,  # VectorStore or HybridRetriever — both have .search()
    k_values: List[int],
) -> Dict[str, Any]:
    """Run recall@k and MRR for all questions."""
    max_k = max(k_values)
    recall_sums = {k: 0.0 for k in k_values}
    rr_sum = 0.0
    per_question = []

    for item in questions:
        question = item["question"]
        relevant = item["relevant_chunks"]

        results = vector_store.search(question, top_k=max_k)
        retrieved_ids = [r["id"] for r in results]

        recalls = {k: recall_at_k(retrieved_ids, relevant, k) for k in k_values}
        rr = reciprocal_rank(retrieved_ids, relevant)

        for k in k_values:
            recall_sums[k] += recalls[k]
        rr_sum += rr

        per_question.append({
            "question": question,
            "relevant": relevant,
            "retrieved_top5": retrieved_ids[:5],
            "recalls": recalls,
            "rr": round(rr, 4),
            "top1_similarity": round(results[0]["similarity"], 4) if results else 0.0,
        })

    n = len(questions)
    return {
        "recall_at_k": {k: round(recall_sums[k] / n, 4) for k in k_values},
        "mrr": round(rr_sum / n, 4),
        "n_questions": n,
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# Ragas evaluation (optional — requires GEMINI_API_KEY and ragas package)
# ---------------------------------------------------------------------------

def run_ragas_eval(questions: List[Dict], vector_store: VectorStore) -> Dict:
    """
    Run Ragas faithfulness + answer_relevancy for a sample of questions.

    Requires: pip install ragas
    Uses the first 10 questions to keep API costs low.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from datasets import Dataset
    except ImportError as e:
        logger.warning(f"Ragas/langchain not installed, skipping: {e}")
        return {"skipped": True, "reason": str(e)}

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        return {"skipped": True, "reason": "GEMINI_API_KEY not set"}

    sample = questions[:10]
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

    for item in sample:
        q = item["question"]
        results = vector_store.search(q, top_k=5)
        contexts = [r["content"] for r in results]

        # Generate a candidate answer using retrieved context
        context_text = "\n\n".join(contexts[:3])
        prompt = (
            f"Answer the question using only the context below.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {q}\nAnswer:"
        )
        try:
            answer = llm.invoke(prompt).content
        except Exception as e:
            answer = f"[Error: {e}]"

        rows["question"].append(q)
        rows["answer"].append(answer)
        rows["contexts"].append(contexts)
        rows["ground_truth"].append(item["answer"])

    dataset = Dataset.from_dict(rows)

    try:
        ragas_llm = LangchainLLMWrapper(llm)
        ragas_emb = LangchainEmbeddingsWrapper(
            GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key,
            )
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=ragas_llm,
            embeddings=ragas_emb,
        )
        return {
            "faithfulness": round(float(result["faithfulness"]), 4),
            "answer_relevancy": round(float(result["answer_relevancy"]), 4),
            "n_samples": len(sample),
        }
    except Exception as e:
        logger.error(f"Ragas evaluation failed: {e}")
        return {"skipped": True, "reason": str(e)}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_results_md(retrieval: Dict, ragas: Dict, run_time: float, out_path: Path):
    k_values = sorted(retrieval["recall_at_k"].keys())
    recall_header = " | ".join(f"Recall@{k}" for k in k_values)
    recall_row = " | ".join(str(retrieval["recall_at_k"][k]) for k in k_values)

    ragas_section = ""
    if ragas.get("skipped"):
        ragas_section = f"> Ragas skipped: {ragas.get('reason', 'unknown')}\n"
    else:
        ragas_section = (
            f"| Metric | Score |\n"
            f"|---|---|\n"
            f"| Faithfulness | {ragas['faithfulness']} |\n"
            f"| Answer Relevancy | {ragas['answer_relevancy']} |\n"
            f"| Samples evaluated | {ragas['n_samples']} |\n"
        )

    md = f"""# Evaluation Results

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  
Questions: {retrieval['n_questions']}  
Run time: {run_time:.1f}s

## Retrieval Metrics (baseline — vector search only)

| {recall_header} | MRR |
|---|---|
| {recall_row} | {retrieval['mrr']} |

## Ragas Metrics (LLM-judged, n={ragas.get('n_samples', 0)})

{ragas_section}

## Per-Question Detail

| # | Question | MRR | R@1 | R@3 | R@5 |
|---|---|---|---|---|---|
"""
    for i, pq in enumerate(retrieval["per_question"], 1):
        q = pq["question"][:60].replace("|", "\\|")
        r1 = pq["recalls"].get(1, "-")
        r3 = pq["recalls"].get(3, "-")
        r5 = pq["recalls"].get(5, "-")
        md += f"| {i} | {q} | {pq['rr']} | {r1} | {r3} | {r5} |\n"

    md += "\n## Notes\n\n"
    md += "- Chunk IDs use format `<filename_stem>_chunk_<index>`\n"
    md += "- Similarity metric: cosine (1 − distance)\n"
    md += "- Embedding model: all-MiniLM-L6-v2 (384-dim)\n"
    md += "- Phase 2 will add BM25 hybrid + cross-encoder re-ranking\n"

    out_path.write_text(md, encoding="utf-8")
    logger.info(f"Results written to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10],
                        help="k values for Recall@k (default: 1 3 5 10)")
    parser.add_argument("--no-ragas", action="store_true",
                        help="Skip Ragas LLM-judged evaluation")
    parser.add_argument("--collection", default=None,
                        help="ChromaDB collection name (overrides config.yaml)")
    args = parser.parse_args()

    cfg = load_config()
    kb_cfg = cfg.get("knowledge_base", {})
    retrieval_cfg = cfg.get("retrieval", {})
    collection = args.collection or cfg.get("specialist", {}).get("collection_name", "climate_docs")
    persist_dir = kb_cfg.get("persist_dir", "./chroma_db")
    embed_model = cfg.get("embedding", {}).get("model_name", "all-MiniLM-L6-v2")

    logger.info(f"Loading vector store: collection={collection}, persist_dir={persist_dir}")

    # Check the knowledge base has been indexed
    vs = VectorStore(
        collection_name=collection,
        persist_dir=persist_dir,
        model_name=embed_model,
    )
    doc_count = vs.collection.count()
    if doc_count == 0:
        logger.error("Knowledge base is empty. Run: python setup_knowledge_base.py")
        sys.exit(1)
    logger.info(f"Vector store ready: {doc_count} chunks indexed")

    # Build retriever (hybrid if enabled in config)
    hybrid = retrieval_cfg.get("hybrid", False)
    rerank = retrieval_cfg.get("rerank", False)
    rrf_k = retrieval_cfg.get("rrf_k", 60)
    logger.info(f"Retriever: hybrid={hybrid}, rerank={rerank}, rrf_k={rrf_k}")
    retriever = HybridRetriever(vs, hybrid=hybrid, rerank=rerank, rrf_k=rrf_k)

    questions = load_golden_qa()
    logger.info(f"Loaded {len(questions)} golden questions")

    t0 = time.time()

    logger.info("Running retrieval evaluation...")
    retrieval = run_retrieval_eval(questions, retriever, k_values=args.k)

    ragas = {"skipped": True, "reason": "disabled via --no-ragas"}
    if not args.no_ragas:
        logger.info("Running Ragas evaluation (n=10)...")
        ragas = run_ragas_eval(questions, vs)

    run_time = time.time() - t0

    # Print summary
    print("\n=== Retrieval Evaluation ===")
    for k in sorted(retrieval["recall_at_k"]):
        print(f"  Recall@{k}: {retrieval['recall_at_k'][k]:.4f}")
    print(f"  MRR:       {retrieval['mrr']:.4f}")
    if not ragas.get("skipped"):
        print(f"\n=== Ragas (n={ragas['n_samples']}) ===")
        print(f"  Faithfulness:     {ragas['faithfulness']:.4f}")
        print(f"  Answer Relevancy: {ragas['answer_relevancy']:.4f}")
    print(f"\nTotal time: {run_time:.1f}s")

    out_path = Path(__file__).parent / "results.md"
    write_results_md(retrieval, ragas, run_time, out_path)
    print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    main()
