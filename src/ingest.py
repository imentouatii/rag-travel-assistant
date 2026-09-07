import json
import os
from pathlib import Path

import chromadb
import dlt
import ollama


BASE_DIR = Path(__file__).resolve().parent.parent
JSON_FILE = BASE_DIR / "data" / "travel_docs_structured.json"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


@dlt.resource(name="travel_docs", write_disposition="replace")
def load_travel_documents():
    if not JSON_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {JSON_FILE}")

    with JSON_FILE.open("r", encoding="utf-8") as f:
        documents = json.load(f)

    for item in documents:
        yield {
            "id": item.get("id"),
            "content": item.get("content", ""),
            "city": item.get("city", "Unknown"),
            "category": item.get("category", "General"),
            "title": item.get("title", "Travel Info"),
            "summary": item.get("summary", ""),
            "keywords": json.dumps(item.get("keywords", []), ensure_ascii=False),
        }


def run_ingestion_pipeline():
    print("Starting dlt ingestion pipeline...")

    pipeline = dlt.pipeline(
        pipeline_name="travel_copilot_ingest",
        destination="duckdb",
        dataset_name="travel_staging",
    )

    load_info = pipeline.run(load_travel_documents())
    print(f"dlt pipeline completed: {load_info}")

    # Read the staged records from DuckDB so dlt is actually part of the pipeline.
    with pipeline.sql_client() as client:
        with client.execute_query("SELECT * FROM travel_docs") as cursor:
            columns = [column[0] for column in cursor.description]
            docs = [dict(zip(columns, row)) for row in cursor.fetchall()]

    if not docs:
        raise RuntimeError("No documents were loaded into the dlt staging table.")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(name="travel_guides")

    ids, documents, embeddings, metadatas = [], [], [], []

    print(f"Generating {EMBEDDING_MODEL} embeddings for {len(docs)} documents...")

    for doc in docs:
        doc_id = str(doc["id"])

        # Include metadata in the embedding text to improve retrieval.
        embedding_text = (
            f"Title: {doc['title']}\n"
            f"City: {doc['city']}\n"
            f"Category: {doc['category']}\n"
            f"Summary: {doc['summary']}\n"
            f"Keywords: {doc['keywords']}\n"
            f"Content: {doc['content']}"
        )

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=embedding_text,
        )

        ids.append(doc_id)
        documents.append(doc["content"])
        embeddings.append(response["embeddings"][0])
        metadatas.append({
            "document_id": doc_id,
            "city": doc["city"],
            "category": doc["category"],
            "title": doc["title"],
            "summary": doc["summary"],
            "keywords": doc["keywords"],
        })

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"✓ Ingested {len(documents)} documents into ChromaDB.")
    print(f"✓ ChromaDB: {CHROMA_DIR}")


if __name__ == "__main__":
    run_ingestion_pipeline()