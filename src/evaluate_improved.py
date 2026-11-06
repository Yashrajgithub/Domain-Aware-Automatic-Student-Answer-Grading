
import csv
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from predict_improved import grade_answer_improved


# Manually defined benchmark cases.
# Required concepts are explicit for this first experiment.

TEST_CASES = [
    {
        "id": "ML1",
        "domain": "ml",
        "type": "Correct",
        "question": "What is the difference between classification and regression?",
        "reference": "Classification predicts discrete labels, while regression predicts continuous numerical values.",
        "student": "Classification assigns discrete labels to data, whereas regression predicts continuous values.",
        "concepts": ["discrete labels", "continuous values"],
    },
    {
        "id": "ML2",
        "domain": "ml",
        "type": "Contradictory",
        "question": "What is the difference between classification and regression?",
        "reference": "Classification predicts discrete labels, while regression predicts continuous numerical values.",
        "student": "Classification predicts continuous numerical values, while regression predicts discrete class labels.",
        "concepts": ["discrete labels", "continuous values"],
    },
    {
        "id": "ML3",
        "domain": "ml",
        "type": "Partial",
        "question": "What is the difference between classification and regression?",
        "reference": "Classification predicts discrete labels, while regression predicts continuous numerical values.",
        "student": "Classification predicts discrete labels.",
        "concepts": ["discrete labels", "continuous values"],
    },
    {
        "id": "ML4",
        "domain": "ml",
        "type": "Irrelevant",
        "question": "What is the difference between classification and regression?",
        "reference": "Classification predicts discrete labels, while regression predicts continuous numerical values.",
        "student": "The weather today is sunny and the temperature is pleasant.",
        "concepts": ["discrete labels", "continuous values"],
    },
    {
        "id": "DE1",
        "domain": "de",
        "type": "Correct",
        "question": "Why can duplicate packets be a problem in a network?",
        "reference": "A receiver may process duplicated packets as valid packets and perform the same action twice.",
        "student": "Duplicate packets may cause the receiver to process the same information twice, potentially repeating a transaction.",
        "concepts": ["receiver", "process", "twice"],
    },
    {
        "id": "DE2",
        "domain": "de",
        "type": "Contradictory",
        "question": "Why can duplicate packets be a problem in a network?",
        "reference": "A receiver may process duplicated packets as valid packets and perform the same action twice.",
        "student": "Duplicate packets never cause repeated processing and always improve network reliability.",
        "concepts": ["receiver", "process", "twice"],
    },
    {
        "id": "DE3",
        "domain": "de",
        "type": "Partial",
        "question": "Why can duplicate packets be a problem in a network?",
        "reference": "A receiver may process duplicated packets as valid packets and perform the same action twice.",
        "student": "Duplicate packets can cause repeated processing.",
        "concepts": ["receiver", "process", "twice"],
    },
    {
        "id": "G1",
        "domain": "general",
        "type": "Correct",
        "question": "Why does ice float on water?",
        "reference": "Ice floats because it is less dense than liquid water.",
        "student": "Ice floats because its density is lower than the density of liquid water.",
        "concepts": ["ice", "less dense", "water"],
    },
    {
        "id": "G2",
        "domain": "general",
        "type": "Contradictory",
        "question": "Why does ice float on water?",
        "reference": "Ice floats because it is less dense than liquid water.",
        "student": "Ice sinks because it is denser than liquid water.",
        "concepts": ["ice", "less dense", "water"],
    },
]



# Load once, outside the evaluation loop
concept_model = SentenceTransformer("all-MiniLM-L6-v2")


def concept_coverage(student_answer, concepts):
    """
    Experimental semantic concept matching.
    A concept counts as covered if its embedding is sufficiently
    similar to at least one sentence in the student answer.
    """

    sentences = [
        s.strip()
        for s in re.split(r"[.!?;\n]+", student_answer)
        if s.strip()
    ]

    if not concepts:
        return 1.0, [], []

    if not sentences:
        return 0.0, [], list(concepts)

    concept_embeddings = concept_model.encode(concepts)
    sentence_embeddings = concept_model.encode(sentences)

    similarities = cosine_similarity(
        concept_embeddings,
        sentence_embeddings
    )

    matched = []
    missing = []

    # Initial experimental threshold; validate before using for scoring.
    threshold = 0.40

    for i, concept in enumerate(concepts):
        best_similarity = float(similarities[i].max())

        if best_similarity >= threshold:
            matched.append(concept)
        else:
            missing.append(concept)

    coverage = len(matched) / len(concepts)

    return coverage, matched, missing


def main():
    rows = []

    print("\nCORRECTNESS-AWARE GRADING EVALUATION")
    print("=" * 78)

    for case in TEST_CASES:
        result = grade_answer_improved(
            case["domain"],
            case["question"],
            case["reference"],
            case["student"],
        )

        coverage, matched, missing = concept_coverage(
            case["student"],
            case["concepts"],
        )

        row = {
            "id": case["id"],
            "domain": case["domain"],
            "type": case["type"],
            "baseline_score": result["baseline_score"],
            "contradiction_probability": result["contradiction_probability"],
            "experimental_score": result["improved_score"],
            "concept_coverage": round(coverage, 3),
            "matched_concepts": "; ".join(matched),
            "missing_concepts": "; ".join(missing),
        }
        rows.append(row)

        print(f"\n{case['id']} | {case['domain']} | {case['type']}")
        print("Baseline:", row["baseline_score"], "/10")
        print("Contradiction probability:", row["contradiction_probability"])
        print("Experimental:", row["experimental_score"], "/10")
        print(f"Concept coverage: {coverage:.0%}")
        print("Matched concepts:", matched)
        print("Missing concepts:", missing)

    output_path = Path(__file__).resolve().parent.parent / "evaluation_results.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 78)
    print("Saved evaluation results to:", output_path)
    print("Remember: phrase-based concept coverage is only a diagnostic.")
    print("Inspect the examples before claiming improved grading accuracy.")


if __name__ == "__main__":
    main()
