import json
import os
import time
from pathlib import Path
from typing import List

import ollama
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "data" / "travel_docs_structured.json"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")


class TravelDocumentSchema(BaseModel):
    city: str = Field(min_length=2, description="Name of the city")
    category: str = Field(min_length=2, description="Travel category")
    title: str = Field(min_length=5, description="Descriptive guide title")
    summary: str = Field(min_length=20, description="Brief 1-2 sentence summary")
    content: str = Field(min_length=100, description="Detailed travel information")
    keywords: List[str] = Field(min_length=3, max_length=5, description="3-5 key search terms")


def generate_production_dataset():
    cities = ["Tokyo", "Paris", "Rome", "New York", "Bangkok", "London", "Barcelona", "Kyoto", "Reykjavik", "Cairo"]
    categories = ["Food & Dining", "Transit & Navigation", "Culture & Etiquette", "Hidden Gems", "Safety & Emergencies"]
    dataset, failed = [], []
    total = len(cities) * len(categories)

    print(f"Generating {total} synthetic travel documents with {OLLAMA_MODEL}...")

    doc_id = 1
    for city in cities:
        for category in categories:
            print(f"[{doc_id}/{total}] {city} - {category}")

            prompt = f"""Create a useful travel-guide document about {city}, focusing specifically on "{category}".
This is a synthetic benchmark corpus for a local RAG system.
- Keep information internally consistent.
- Prefer well-known, commonly documented places and practices.
- Do not claim information is independently verified.
- Include practical, information-dense content useful for retrieval.
- Return only the requested JSON object."""

            try:
                response = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    format=TravelDocumentSchema.model_json_schema(),
                    options={"temperature": 0.2},
                )
                data = json.loads(response["message"]["content"])
                document = TravelDocumentSchema.model_validate(data).model_dump()
                document["id"] = f"DOC-{doc_id:04d}"
                dataset.append(document)
                print("  ✓ Validated")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                failed.append({"city": city, "category": category, "error": str(e)})

            doc_id += 1
            time.sleep(0.2)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated: {len(dataset)}/{total}")
    print(f"Saved to: {OUTPUT_FILE}")

    if failed:
        print(f"Failed: {len(failed)} documents")
    else:
        print("✓ Dataset generation completed successfully.")


if __name__ == "__main__":
    generate_production_dataset()