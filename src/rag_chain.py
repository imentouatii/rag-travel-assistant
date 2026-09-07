import os
import re
import time
from pathlib import Path
import chromadb
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "chroma_db"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


class TravelRAGChain:
    def __init__(self):
        if not DB_PATH.exists():
            raise FileNotFoundError(f"ChromaDB not found at {DB_PATH}. Run ingest.py first.")

        self.client = chromadb.PersistentClient(path=str(DB_PATH))
        self.collection = self.client.get_or_create_collection(name="travel_guides")

        data = self.collection.get(include=["documents", "metadatas"])
        self.ids = data["ids"]
        self.documents = data["documents"]
        self.metadatas = data["metadatas"]

        if not self.documents:
            raise RuntimeError("ChromaDB contains no documents. Run ingest.py first.")

        self.docs_by_id = dict(zip(self.ids, self.documents))
        self.metadata_by_id = dict(zip(self.ids, self.metadatas))
        self.bm25 = BM25Okapi([tokenize(doc) for doc in self.documents])
        self.reranker = CrossEncoder(RERANKER_MODEL)

    def _rewrite_query(self, query):
        try:
            response = ollama.generate(
                model=OLLAMA_MODEL,
                prompt=(
                    "Rewrite the following travel question into a concise search query. "
                    "Keep the city, topic, and important keywords. Return only the rewritten query.\n\n"
                    f"Question: {query}"
                ),
                options={"temperature": 0},
            )
            rewritten = response["response"].strip()
            return rewritten or query
        except Exception:
            return query

    def _rrf(self, result_lists, k=60):
        scores = {}

        for results in result_lists:
            for rank, doc_id in enumerate(results, start=1):
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

        return [
            doc_id
            for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]

    def query(self, user_query, n_results=3):
        total_start = time.perf_counter()

        rewritten_query = self._rewrite_query(user_query)

        embedding_start = time.perf_counter()
        query_embedding = ollama.embed(
            model=EMBEDDING_MODEL,
            input=rewritten_query,
        )["embeddings"][0]

        vector_result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(10, len(self.documents)),
            include=["documents", "metadatas"],
        )
        embedding_latency = time.perf_counter() - embedding_start

        vector_ids = vector_result["ids"][0]

        bm25_start = time.perf_counter()
        bm25_scores = self.bm25.get_scores(tokenize(rewritten_query))
        bm25_ids = [
            doc_id
            for _, doc_id in sorted(
                zip(bm25_scores, self.ids),
                reverse=True,
            )[:10]
        ]
        bm25_latency = time.perf_counter() - bm25_start

        hybrid_ids = self._rrf([vector_ids, bm25_ids])[:10]

        rerank_start = time.perf_counter()
        rerank_pairs = [[rewritten_query, self.docs_by_id[doc_id]] for doc_id in hybrid_ids]
        rerank_scores = self.reranker.predict(rerank_pairs)

        reranked_ids = [
            doc_id
            for _, doc_id in sorted(
                zip(rerank_scores, hybrid_ids),
                reverse=True,
            )
        ]
        rerank_latency = time.perf_counter() - rerank_start

        final_ids = reranked_ids[:n_results]

        generation_context = []
        sources = []

        for doc_id in final_ids:
            metadata = self.metadata_by_id[doc_id]
            generation_context.append(
                f"Title: {metadata.get('title', '')}\n"
                f"City: {metadata.get('city', '')}\n"
                f"Category: {metadata.get('category', '')}\n"
                f"Content: {self.docs_by_id[doc_id]}"
            )
            sources.append({
                "document_id": doc_id,
                "city": metadata.get("city", ""),
                "category": metadata.get("category", ""),
                "title": metadata.get("title", ""),
            })

        context = "\n\n---\n\n".join(generation_context)

        generation_start = time.perf_counter()

        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=(
                "You are a helpful travel assistant. "
                "Answer the user's question using only the supplied travel context. "
                "If the context does not contain enough information, say so. "
                "Do not invent facts.\n\n"
                f"Travel context:\n{context}\n\n"
                f"User question: {user_query}"
            ),
            options={"temperature": 0.2},
        )

        answer = response["response"].strip()
        generation_latency = time.perf_counter() - generation_start
        total_latency = time.perf_counter() - total_start

        return {
            "answer": answer,
            "original_query": user_query,
            "rewritten_query": rewritten_query,
            "sources": sources,
            "retrieved_documents": len(final_ids),
            "embedding_latency_ms": round(embedding_latency * 1000, 2),
            "bm25_latency_ms": round(bm25_latency * 1000, 2),
            "rerank_latency_ms": round(rerank_latency * 1000, 2),
            "generation_latency_ms": round(generation_latency * 1000, 2),
            "total_latency_ms": round(total_latency * 1000, 2),
            "retrieval_strategy": "Hybrid BM25 + Vector + RRF + CrossEncoder",
        }