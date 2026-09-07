import json
import os
import re
from pathlib import Path

import chromadb
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "chroma_db"
REPORT_PATH = BASE_DIR / "data" / "evaluation_report.json"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
K = 5


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def reciprocal_rank_fusion(result_lists, k=60):
    scores = {}
    for results in result_lists:
        for rank, doc_id in enumerate(results, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def retrieval_metrics(retrieved, relevant, k=5):
    retrieved = retrieved[:k]
    relevant = set(relevant)
    hits = [doc_id for doc_id in retrieved if doc_id in relevant]

    hit = 1 if hits else 0
    recall = len(set(hits)) / len(relevant) if relevant else 0
    precision = len(hits) / len(retrieved) if retrieved else 0

    mrr = 0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            mrr = 1 / rank
            break

    return {
        "hit_at_5": round(hit, 4),
        "recall_at_5": round(recall, 4),
        "precision_at_5": round(precision, 4),
        "mrr_at_5": round(mrr, 4),
    }


def evaluate_retrieval_strategies():
    print("Evaluating retrieval strategies...")

    if not DB_PATH.exists():
        raise FileNotFoundError(f"ChromaDB not found at {DB_PATH}. Run ingest.py first.")

    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(name="travel_guides")

    data = collection.get(include=["documents", "metadatas"])
    documents = data["documents"]
    metadatas = data["metadatas"]
    ids = data["ids"]

    if not documents:
        raise RuntimeError("ChromaDB contains no documents. Run ingest.py first.")

    docs_by_id = dict(zip(ids, documents))
    metadata_by_id = dict(zip(ids, metadatas))
    bm25 = BM25Okapi([tokenize(doc) for doc in documents])
    reranker = CrossEncoder(RERANKER_MODEL)

    # Each query has manually defined relevance criteria based on the
    # known synthetic corpus metadata.
    test_queries = [
        {"query": "best food and dining spots in Paris", "city": "Paris", "categories": ["Food & Dining"]},
        {"query": "Paris public transportation and metro tips", "city": "Paris", "categories": ["Transit & Navigation"]},
        {"query": "Paris food and transit recommendations", "city": "Paris", "categories": ["Food & Dining", "Transit & Navigation"]},
        {"query": "Tokyo local food recommendations", "city": "Tokyo", "categories": ["Food & Dining"]},
        {"query": "Tokyo subway and transportation tips", "city": "Tokyo", "categories": ["Transit & Navigation"]},
        {"query": "Tokyo cultural etiquette for visitors", "city": "Tokyo", "categories": ["Culture & Etiquette"]},
        {"query": "Rome hidden gems and unusual places", "city": "Rome", "categories": ["Hidden Gems"]},
        {"query": "Rome safety advice for travelers", "city": "Rome", "categories": ["Safety & Emergencies"]},
        {"query": "London food and dining guide", "city": "London", "categories": ["Food & Dining"]},
        {"query": "London public transit tips", "city": "London", "categories": ["Transit & Navigation"]},
        {"query": "Kyoto culture and etiquette tips", "city": "Kyoto", "categories": ["Culture & Etiquette"]},
        {"query": "Bangkok hidden gems", "city": "Bangkok", "categories": ["Hidden Gems"]},
        {"query": "Barcelona travel safety advice", "city": "Barcelona", "categories": ["Safety & Emergencies"]},
        {"query": "Cairo local food recommendations", "city": "Cairo", "categories": ["Food & Dining"]},
        {"query": "Reykjavik transportation information", "city": "Reykjavik", "categories": ["Transit & Navigation"]},
    ]

    strategies = ["vector", "bm25", "hybrid_rrf", "hybrid_rrf_reranker"]
    results = {strategy: [] for strategy in strategies}

    for item in test_queries:
        query = item["query"]
        relevant_ids = [
            doc_id for doc_id, metadata in metadata_by_id.items()
            if metadata.get("city") == item["city"]
            and metadata.get("category") in item["categories"]
        ]

        embedding = ollama.embed(
            model=EMBEDDING_MODEL,
            input=query,
        )["embeddings"][0]

        vector_result = collection.query(
            query_embeddings=[embedding],
            n_results=min(10, len(documents)),
            include=["documents"],
        )
        vector_ids = [
            ids[documents.index(doc)]
            for doc in vector_result["documents"][0]
        ]

        bm25_scores = bm25.get_scores(tokenize(query))
        bm25_ids = [
            doc_id for _, doc_id in sorted(
                zip(bm25_scores, ids),
                reverse=True
            )[:10]
        ]

        hybrid_ids = reciprocal_rank_fusion(
            [vector_ids, bm25_ids]
        )[:10]

        rerank_pairs = [
            [query, docs_by_id[doc_id]]
            for doc_id in hybrid_ids
        ]
        rerank_scores = reranker.predict(rerank_pairs)

        reranked_ids = [
            doc_id for _, doc_id in sorted(
                zip(rerank_scores, hybrid_ids),
                reverse=True
            )
        ]

        retrieved = {
            "vector": vector_ids,
            "bm25": bm25_ids,
            "hybrid_rrf": hybrid_ids,
            "hybrid_rrf_reranker": reranked_ids,
        }

        for strategy in strategies:
            metrics = retrieval_metrics(
                retrieved[strategy],
                relevant_ids,
                K,
            )
            results[strategy].append({
                "query": query,
                "relevant_documents": relevant_ids,
                **metrics,
            })

    summary = {}

    for strategy, evaluations in results.items():
        summary[strategy] = {
            "queries": len(evaluations),
            "hit_at_5": round(
                sum(x["hit_at_5"] for x in evaluations) / len(evaluations), 4
            ),
            "recall_at_5": round(
                sum(x["recall_at_5"] for x in evaluations) / len(evaluations), 4
            ),
            "precision_at_5": round(
                sum(x["precision_at_5"] for x in evaluations) / len(evaluations), 4
            ),
            "mrr_at_5": round(
                sum(x["mrr_at_5"] for x in evaluations) / len(evaluations), 4
            ),
        }

    best_strategy = max(
        summary,
        key=lambda strategy: (
            summary[strategy]["mrr_at_5"],
            summary[strategy]["recall_at_5"],
            summary[strategy]["precision_at_5"],
        ),
    )

    report = {
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "k": K,
        "evaluation_queries": len(test_queries),
        "evaluation_method": (
            "Relevance labels are defined from the known city/category "
            "metadata of the synthetic benchmark corpus."
        ),
        "strategies": summary,
        "best_strategy": best_strategy,
        "per_query_results": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nRetrieval evaluation results:")
    for strategy, metrics in summary.items():
        print(
            f"{strategy}: "
            f"Hit@5={metrics['hit_at_5']:.3f}, "
            f"Recall@5={metrics['recall_at_5']:.3f}, "
            f"MRR@5={metrics['mrr_at_5']:.3f}"
        )

    print(f"\n✓ Best strategy: {best_strategy}")
    print(f"✓ Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    evaluate_retrieval_strategies()