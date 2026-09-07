import json
import os
from pathlib import Path

import ollama


BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_PATH = BASE_DIR / "data" / "llm_evaluation_report.json"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")


def calculate_fact_coverage(response, key_facts):
    response_lower = response.lower()
    matches = sum(1 for fact in key_facts if fact.lower() in response_lower)
    return matches / len(key_facts) if key_facts else 0.0


def evaluate_answer(response, key_facts):
    coverage = calculate_fact_coverage(response, key_facts)
    word_count = len(response.split())

    # Penalize extremely short answers while avoiding a reward for verbosity.
    completeness = min(word_count / 60, 1.0)
    quality_score = (coverage * 0.8) + (completeness * 0.2)

    return {
        "fact_coverage": round(coverage, 4),
        "completeness": round(completeness, 4),
        "quality_score": round(quality_score, 4),
        "word_count": word_count,
    }


def evaluate_llm_outputs():
    print(f"Evaluating LLM prompt strategies with {OLLAMA_MODEL}...")

    test_cases = [
        {
            "query": "What are some hidden gems and local transit tips in Paris?",
            "context": (
                "Paris has lesser-known attractions such as the Petite Ceinture "
                "railway and La Coulée Verte. For public transportation, "
                "travelers can use a Navigo Easy pass for eligible journeys."
            ),
            "facts": ["Petite Ceinture", "La Coulée Verte", "Navigo Easy"],
        },
        {
            "query": "What should I know about food and culture in Tokyo?",
            "context": (
                "Tokyo offers sushi, ramen, and izakaya dining. Visitors should "
                "avoid tipping in most restaurants and should be considerate "
                "when using public spaces."
            ),
            "facts": ["sushi", "ramen", "izakaya", "avoid tipping"],
        },
        {
            "query": "What are some cultural etiquette tips for Kyoto?",
            "context": (
                "In Kyoto, visitors should be respectful at temples and shrines, "
                "keep voices low in traditional areas, and follow local customs "
                "when entering certain buildings."
            ),
            "facts": ["temples", "shrines", "keep voices low", "local customs"],
        },
        {
            "query": "What safety advice is useful for travelers in Rome?",
            "context": (
                "Travelers in Rome should take care of personal belongings in "
                "crowded tourist areas, remain aware of their surroundings, "
                "and use official transportation services."
            ),
            "facts": ["personal belongings", "crowded tourist areas", "official transportation"],
        },
        {
            "query": "What public transportation options are useful in London?",
            "context": (
                "London's public transportation network includes the Underground, "
                "buses, and Overground services. Contactless payment can be used "
                "on many services."
            ),
            "facts": ["Underground", "buses", "Overground", "contactless payment"],
        },
    ]

    strategies = {
        "concise_grounded": {
            "description": "Concise answer focused strictly on the supplied context.",
            "temperature": 0.1,
        },
        "structured_expert": {
            "description": "Structured local-guide response while remaining grounded in context.",
            "temperature": 0.2,
        },
    }

    results = {name: [] for name in strategies}

    for index, case in enumerate(test_cases, 1):
        print(f"[{index}/{len(test_cases)}] Evaluating: {case['query']}")

        for strategy, config in strategies.items():
            if strategy == "concise_grounded":
                prompt = f"""Answer the query using ONLY the provided context.
Be concise, factual, and use short bullet points.
Do not add information that is not supported by the context.

Context:
{case['context']}

Query:
{case['query']}"""
            else:
                prompt = f"""You are a helpful local travel guide.
Answer the query using ONLY the provided context.
Organize the answer clearly and make it useful to a traveler.
Do not invent facts or add information that is not supported by the context.

Context:
{case['context']}

Query:
{case['query']}"""

            response = ollama.generate(
                model=OLLAMA_MODEL,
                prompt=prompt,
                options={"temperature": config["temperature"]},
            )

            output = response["response"].strip()
            metrics = evaluate_answer(output, case["facts"])

            results[strategy].append({
                "query": case["query"],
                **metrics,
                "output": output,
            })

    summary = {}

    for strategy, evaluations in results.items():
        summary[strategy] = {
            "description": strategies[strategy]["description"],
            "queries_evaluated": len(evaluations),
            "average_fact_coverage": round(
                sum(x["fact_coverage"] for x in evaluations) / len(evaluations), 4
            ),
            "average_completeness": round(
                sum(x["completeness"] for x in evaluations) / len(evaluations), 4
            ),
            "average_quality_score": round(
                sum(x["quality_score"] for x in evaluations) / len(evaluations), 4
            ),
            "average_word_count": round(
                sum(x["word_count"] for x in evaluations) / len(evaluations), 1
            ),
        }

    best_approach = max(
        summary,
        key=lambda x: summary[x]["average_quality_score"]
    )

    report = {
        "model": OLLAMA_MODEL,
        "evaluation_method": (
            "Prompt strategies are evaluated across multiple queries using "
            "fact coverage, completeness, and a combined quality score. "
            "Fact coverage measures the proportion of predefined context facts "
            "included in the generated answer."
        ),
        "test_cases": len(test_cases),
        "strategies": summary,
        "best_approach": best_approach,
        "per_query_results": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nLLM evaluation results:")
    for strategy, metrics in summary.items():
        print(
            f"{strategy}: "
            f"fact_coverage={metrics['average_fact_coverage']:.3f}, "
            f"completeness={metrics['average_completeness']:.3f}, "
            f"quality={metrics['average_quality_score']:.3f}"
        )

    print(f"\n✓ Best prompt strategy: {best_approach}")
    print(f"✓ Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    evaluate_llm_outputs()